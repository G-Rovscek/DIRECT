import argparse
import time
import math
import json
import torch
import os
import numpy as np
from welford import Welford
from streamhist import StreamHist
import math
import glob

from src.data.sst_data import get_dataloaders, MaskSampler
from src.utils.config_utils import deep_update, load_config
from src.model.flow import Direct

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def robust_stats_from_list(values):
    """
    values: 1D numpy array or list
    returns: (median, mad)
    """
    a = np.asarray(values)
    if a.size == 0:
        return 0.0, 0.0
    median = np.median(a)
    mad = np.median(np.abs(a - median))
    return float(median), float(mad)


def compute_global_robust_stats(all_recons, dataloaders, metadata, sample_limit=None):
    """Two-pass helper: collect eps_del across dataset and return median, mad"""
    eps_collect = []

    for i, rec_sample in enumerate(all_recons):
        if (sample_limit is not None) and (i >= sample_limit):
            break

        obs, miss_mask, d_idx = dataloaders['test'].dataset[rec_sample['data_idx']]
        obs = obs[metadata['feat_to_idx']['measurement_0']]
        obs = obs * metadata['data_std'] + metadata['data_mean']
        obs = obs.unsqueeze(0)
        miss_mask = miss_mask.unsqueeze(0)

        _, sampled_mask, _ = dataloaders['test'].dataset[rec_sample['sampled_idx']]
        sampled_mask = sampled_mask.unsqueeze(0)

        recons = rec_sample['recons'] * metadata['data_std'] + metadata['data_mean']
        rec = torch.mean(recons, dim=0)
        var = torch.var(recons, dim=0)

        eps = (obs - rec) / torch.sqrt(var)
        M_all = (miss_mask * metadata['land_mask'])
        M_del = M_all * (1 - sampled_mask)

        # collect deleted eps values (as numpy scalars)
        if (M_del == 1).any():
            eps_del_t = eps[M_del == 1].flatten().cpu().numpy()
            # avoid NaNs/Infs: keep finite
            eps_del_t = eps_del_t[np.isfinite(eps_del_t)]
            if eps_del_t.size > 0:
                eps_collect.append(eps_del_t)

    if len(eps_collect) == 0:
        return 0.0, 0.0
    
    eps_concat = np.concatenate(eps_collect, axis=0)
    median, mad = robust_stats_from_list(eps_concat)
    print(f"[GLOBAL ROBUST STATS] median={median:.4f}, mad={mad:.6e}, n={eps_concat.size}")
    return median, mad


class VarianceBiasAnalysis:
    """Analyse the statistical properties of the estimated variance and reconstruction bias"""

    def __init__(self, maxbins=100):
        # initialize the streaming histograms
        # (See: https://github.com/carsonfarmer/streamhist)
        self.maxbins = maxbins
        self.hist_del = StreamHist()
        self.hist_vis = StreamHist()

        # Use the Welford's algorithm to estimate accurate statistics over a stream of data
        # (See: https://pypi.org/project/welford/)
        self.w_del = Welford()
        self.w_vis = Welford()
        
        self.removed_count = 0
        self.eps_del_count = 0

    def update(self, target, rec, var, land_mask, missing_mask, sampled_mask, sample_idx=None,
               robust_median=None, robust_mad=None, robust_thresh=3.5):
        # compute the scaled difference (eps) and unscaled difference (bias)  
        eps = (target - rec) / torch.sqrt(var)
        bias = target - rec

        # compute deleted and visible masks
        M_all = missing_mask * land_mask
        M_del = M_all * (1 - sampled_mask)
        M_vis = M_all * sampled_mask
    
        # separate into deleted and visible regions
        eps_del_t = eps[M_del == 1]
        eps_vis_t = eps[M_vis == 1]
        
        eps_del_list = eps_del_t.flatten().cpu().numpy().tolist()
        eps_vis_list = eps_vis_t.flatten().cpu().numpy().tolist()

        bias_del_t = bias[M_del == 1]
        bias_vis_t = bias[M_vis == 1]
        
        bias_del_list = bias_del_t.flatten().cpu().numpy().tolist()
        bias_vis_list = bias_vis_t.flatten().cpu().numpy().tolist()
        
        if robust_median is not None and robust_mad is not None and len(eps_del_list) > 0:
            mad = robust_mad
            if mad <= 0:
                # if mad==0 then all values identical -> no outliers
                robust_z = np.zeros(len(eps_del_list))
            else:
                robust_z = 0.6745 * (np.array(eps_del_list) - robust_median) / mad
            
            keep_mask = np.abs(robust_z) <= robust_thresh
            removed_count = int(np.sum(~keep_mask))
            self.removed_count += removed_count
            self.eps_del_count += len(eps_del_list)
            if removed_count > 0:
                print(f"[INFO] sample {sample_idx}: removed {removed_count}/{len(eps_del_list)} deleted eps outliers (robust_z>{robust_thresh})")

            # filter lists
            eps_del_list = list(np.array(eps_del_list)[keep_mask])
            bias_del_list = list(np.array(bias_del_list)[keep_mask])

        
        # update the streaming histogram
        if len(eps_del_list) > 0:
            self.hist_del.update(eps_del_list)
        if len(eps_vis_list) > 0:
            self.hist_vis.update(eps_vis_list)

        # update mu_eps, sigma_eps and the bias 
        for (eps_i, bias_i) in zip(eps_del_list, bias_del_list):
            self.w_del.add(np.array([eps_i, bias_i]))

        for (eps_i, bias_i) in zip(eps_vis_list, bias_vis_list):
            self.w_vis.add(np.array([eps_i, bias_i]))

    def export(self, filename) -> str:
        # store mu_eps, sigma_eps and the bias
        def get_means_counts_widths(hist):
            # compute counts and bin edges
            counts, bins = hist.compute_breaks(self.maxbins)

            means = [(a + b)/2. for a, b in zip(bins[:-1], bins[1:])]
            widths = [a - b for a, b in zip(bins[1:], bins[:-1])]

            # return in dict format
            return {"means": means, "counts": counts, "widths": widths}

        hist_del = get_means_counts_widths(self.hist_del)
        hist_del["mu_eps"] = self.w_del.mean[0]
        hist_del["sigma_eps"] = math.sqrt(self.w_del.var_p[0])
        hist_del["bias"] = self.w_del.mean[1]

        # store mu_eps, sigma_eps and the bias
        hist_vis = get_means_counts_widths(self.hist_vis)
        hist_vis["mu_eps"] = self.w_vis.mean[0]
        hist_vis["sigma_eps"] = math.sqrt(self.w_vis.var_p[0])
        hist_vis["bias"] = self.w_vis.mean[1]   
     
        # save data
        _data = {"deleted_regions": hist_del, "visible_regions": hist_vis}
        with open(filename, "w") as file:
            json.dump(_data, file, indent=4)
        print(f"Results saved to:{filename}", flush=True)


@torch.no_grad()
def eval(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = cfg['testing']['mode']
    file_name = cfg['va']['file']
    sigma0 = cfg['va']['sigma0']

    dataloaders, metadata = get_dataloaders(
        data_path=cfg["data"]["data_path"],
        cloud_coverage_threshold=cfg["data"]["cloud_coverage_threshold"],
        batch_size=1,
        time_win=cfg["data"]["time_win"],
        mask_in_cond=cfg["data"]["mask_in_cond"],
        n_samples=cfg["data"]["n_samples"],
        standardize=cfg["data"]["standardize"],
        use_onehot=cfg["data"]["use_onehot"],
        use_filled=cfg["data"]["use_filled"],
        max_delta=cfg["data"]["max_delta"],
    )

    del dataloaders['train']
    if mode == 'test':
        del dataloaders['val']
    else:
        del dataloaders['test']
    
    mask_sampler = MaskSampler(
        dataset=dataloaders[mode].dataset
    )
    mask_sampler.to(device)

    direct = Direct.load_from_pretrained(cfg['testing']['model'], device=device)
    direct.eval()

    va = VarianceBiasAnalysis()
    
    all_recons = None
    if cfg['va'].get('recons', ''):
        recons_path = glob.glob(os.path.join(cfg['va']['recons'], '*.pt'))[0] if not cfg['va']['recons'].endswith('.pt') else cfg['va']['recons']
        print(f'Loading Reconstructions from: {recons_path}')
        all_recons = torch.load(recons_path)

    ml = metadata['land_mask'].to(device)
    if all_recons == None:
        for _ in range(cfg['testing']['num_runs']):
            for obs, mt, d_idx in dataloaders['test']:
                obs = obs.to(device)
                mt = mt.to(device)
            
                mm, s_idx = mask_sampler.sample_valid_mask(ml, mt)

                m_vis = mt * mm * ml
                recons = direct.reconstruct(obs, m_vis, ml, cfg['inference']['Nk'], cfg['inference']['N'])

                recons = recons * metadata['data_std'] + metadata['data_mean']
                mu_rec = torch.mean(recons, dim=0)
                mu_var = torch.var(recons, dim=0)
                var_safe = mu_var + (sigma0 ** 2)

                target = obs[:, metadata['feat_to_idx']['measurement_0'], :, :] * metadata['data_std'] + metadata['data_mean']
                
                va.update(target, mu_rec, var_safe, ml, mt, mm, sample_idx=0,
                    robust_median=None, robust_mad=None, robust_thresh=3.5)
    
    else:
        for i, rec_sample in enumerate(all_recons):
            obs, mt, _ = dataloaders[mode].dataset[rec_sample['data_idx']]
            obs = obs.to(device)
            mt = mt.unsqueeze(0).to(device)

            target = obs[metadata['feat_to_idx']['measurement_0'], :, :].unsqueeze(0) * metadata['data_std'] + metadata['data_mean']

            _, mm, _ = dataloaders[mode].dataset[rec_sample['sampled_idx']]
            mm = mm.unsqueeze(0).to(device)

            recons = rec_sample['recons'] * metadata['data_std'] + metadata['data_mean']
            recons = recons.to(device)
            mu_rec = torch.mean(recons, dim=0)
            mu_var = torch.var(recons, dim=0)
            
            var_safe = mu_var + (sigma0 ** 2)
            
            va.update(target, mu_rec, var_safe, ml, mt, mm, sample_idx=i,
                robust_median=None, robust_mad=None, robust_thresh=3.5)
    
    export_path = cfg['va']['export_dir']
    print(f"exporting CCE result...", flush=True)
    va.export(os.path.join(export_path, file_name))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_config", type=str, required=True, help="Path to test yaml config file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print(f"Script ran at time: {time.strftime('%d %m %Y %H:%M:%S', time.localtime())}")
    args = parse_arguments()
    cfg = load_config(args.test_config)
    cfg['testing']['model'] = glob.glob(os.path.join(cfg['testing']['model_dir'], '*.pt'))[0]
    t_cfg = load_config(f'{cfg['testing']['model_dir']}/configs/train_config.yaml')
    cfg = deep_update(t_cfg, cfg)
    eval(cfg)
    print(f"Script finished at time: {time.strftime('%d %m %Y %H:%M:%S', time.localtime())}")