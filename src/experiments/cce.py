import torch
import numpy as np
import argparse
import json
import time
import os
import glob
import torch.nn.functional as F

from typing import List

from src.data.sst_data import get_dataloaders, MaskSampler
from src.utils.losses import compute_mse
from src.model.flow import Direct
from src.utils.config_utils import load_config, deep_update

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class CloudCoverageLevelRobustnessAnalysis:
    """Analyse the robustness of the model to clouds with varying coverage levels"""
    
    min_cloud_coverage = float("inf")
    max_cloud_coverage = float("-inf")

    def __init__(
        self, metrics: List[str], group_max_coverage: List[float] = [0.6, 0.75]
    ) -> None:
        self._data = {m: {} for m in metrics}
        self.groups = {}

        # add groups
        group_max_coverage = [0.0] + group_max_coverage + [1.0]
        for idx in range(len(group_max_coverage) - 1):
            # compute group bounds
            lower_bound, upper_bound = (
                group_max_coverage[idx],
                group_max_coverage[idx + 1],
            )
            self.groups[f"g_{idx}"] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

            # add group to all metrics
            for m in metrics:
                self._data[m][f"g_{idx}"] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "vals": [],
                }

        print(f"groups: {self.groups}", flush=True)

    def update(self, data, sampled_mask, missing_mask, land_mask):
        # find the group corresponding to cloud coverage
        coverage = self._compute_cloud_coverage(sampled_mask, missing_mask, land_mask)
        group = self._find_group(coverage)
        assert (
            group != None
        ), f"Coverage:{coverage} out of range! Could not find the appropriate group!"

        # store data in the corresponding group
        for key, val in data.items():
            assert key in data.keys(), f"Unknown metric:{key}"
            self._data[key][group]["vals"] += [val]

        # update cloud coverage
        self.min_cloud_coverage = min(self.min_cloud_coverage, coverage)
        self.max_cloud_coverage = max(self.max_cloud_coverage, coverage)

    def export(self, filename: str) -> None:
        _data = self._data.copy()
        for metric in _data.keys():
            for group in _data[metric].keys():
                _data[metric][group] = {
                    "lower_bound": self._data[metric][group]["lower_bound"],
                    "upper_bound": self._data[metric][group]["upper_bound"],
                    "mean": np.mean(self._data[metric][group]["vals"]),
                    "std": np.std(self._data[metric][group]["vals"]),
                    "n_samples": len(self._data[metric][group]["vals"]),
                }

        # store the minimum and maximum (observed) cloud coverage
        _data["min_cloud_coverage"] = self.min_cloud_coverage
        _data["max_cloud_coverage"] = self.max_cloud_coverage

        with open(filename, "w") as file:
            json.dump(_data, file, indent=4)
        print(f"Results saved to:{filename}", flush=True)

    def _find_group(self, coverage) -> str:
        _group = None
        for group, val in self.groups.items():
            if coverage > val["lower_bound"] and coverage <= val["upper_bound"]:
                _group = group
                break
        return _group

    def _compute_cloud_coverage(self, sampled_mask, missing_mask, land_mask):
        mask = missing_mask * sampled_mask
        assert (
            torch.unique(mask[:, land_mask == 0]) == 1
        ), "Missing mask defined on land!"
        area_cloud = torch.sum(1 - mask)
        area_sea = torch.sum(land_mask)
        return (area_cloud / area_sea).item()


@torch.no_grad()
def eval(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = cfg['testing']['mode']

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

    cclra = CloudCoverageLevelRobustnessAnalysis(
        ["RMSE_all", "RMSE_del", "RMSE_vis"], group_max_coverage=[0.6, 0.75]
    )
    
    all_recons = None
    if cfg['cce'].get('recons', ''):
        recons_path = glob.glob(os.path.join(cfg['cce']['recons'], '*.pt'))[0] if not cfg['cce']['recons'].endswith('.pt') else cfg['cce']['recons']
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

                target = obs[:, metadata['feat_to_idx']['measurement_0'], :, :] * metadata['data_std'] + metadata['data_mean']

                ld = compute_mse(target, mu_rec, mt, ml, mm)

                data = {
                    "RMSE_all": np.sqrt(ld["mse_all"].item()).mean(),
                    "RMSE_del": np.sqrt(ld["mse_hid"].item()).mean(),
                    "RMSE_vis": np.sqrt(ld["mse_vis"].item()).mean(),
                }
            
                cclra.update(data, mm, mt, ml)
    
    else:
        for rec_sample in all_recons:
            obs, mt, _ = dataloaders[mode].dataset[rec_sample['data_idx']]
            obs = obs.to(device)
            mt = mt.unsqueeze(0).to(device)

            target = obs[metadata['feat_to_idx']['measurement_0'], :, :].unsqueeze(0) * metadata['data_std'] + metadata['data_mean']

            _, mm, _ = dataloaders[mode].dataset[rec_sample['sampled_idx']]
            mm = mm.unsqueeze(0).to(device)

            recons = (rec_sample['recons'] * metadata['data_std'] + metadata['data_mean']).to(device)
            mu_rec = torch.mean(recons, dim=0, keepdim=True)
            
            ld = compute_mse(target, mu_rec, mt, ml, mm)

            data = {
                "RMSE_all": np.sqrt(ld["mse_all"].item()).mean(),
                "RMSE_del": np.sqrt(ld["mse_hid"].item()).mean(),
                "RMSE_vis": np.sqrt(ld["mse_vis"].item()).mean(),
            }
        
            cclra.update(data, mm, mt, ml)
    
    export_path = cfg['cce']['export_dir']
    print(f"exporting CCE result...", flush=True)
    cclra.export(os.path.join(export_path, cfg['cce']['file']))


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