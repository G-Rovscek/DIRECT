import torch
import time
import numpy as np
import os
import argparse
import glob
from pathlib import Path

from src.model.flow import Direct
from src.data.sst_data import get_dataloaders, MaskSampler
from src.utils.losses import compute_mse
from src.utils.config_utils import load_config, deep_update

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

    ml = metadata['land_mask'].to(device)
    losses = {"mse_all": [], "mse_hid": [], "mse_vis": []}
    saved_outputs = []
    for _ in range(cfg['testing']['num_runs']):
        for obs, mt, d_idx in dataloaders['test']:
            obs = obs.to(device)
            mt = mt.to(device)
        
            mm, s_idx = mask_sampler.sample_valid_mask(ml, mt)

            m_vis = mt * mm * ml
            recons = direct.reconstruct(obs, m_vis, ml, cfg['inference']['Nk'], cfg['inference']['N'])

            if cfg['testing']['save_recons']:
                saved_outputs.append({
                    "data_idx":d_idx.item(),
                    "sampled_idx":s_idx.item(),
                    "recons": recons.cpu(),
                })

            recons = recons * metadata['data_std'] + metadata['data_mean']
            mu_rec = torch.mean(recons, dim=0, keepdim=True)

            target = obs[:, metadata['feat_to_idx']['measurement_0'], :, :].unsqueeze(1) * metadata['data_std'] + metadata['data_mean']

            ld = compute_mse(target, mu_rec, mt, ml, mm)

            losses["mse_all"] += [ld["mse_all"].item()]
            losses["mse_hid"] += [ld["mse_hid"].item()]
            losses["mse_vis"] += [ld["mse_vis"].item()]
    
    print("-----------------------")
    for k, mse in losses.items():
        print(f'r{k}: {np.sqrt(mse).mean()}')
    
    if cfg['testing']['save_recons']:
        Path(Path(cfg['testing']['model_dir']) / 'recons').mkdir(parents=True, exist_ok=True)
        file_name = f"recons-loop{cfg['testing']['num_runs']}-averaging{cfg['inference']['N']}.pt"
        torch.save(saved_outputs, f'{cfg['testing']['model_dir']}/recons/{file_name}')


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