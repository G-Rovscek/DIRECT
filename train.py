import torch
import time
import argparse
import numpy as np
import os
import wandb
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from pathlib import Path

from src.data.sst_data import get_dataloaders, MaskSampler
from src.utils.config_utils import load_config, save_config
from src.model.flow import Direct

def train_one_epoch(
        direct,
        train_dataloader,
        metadata,
        optimizer,
        mask_sampler,
):
    direct.train()

    land_mask = metadata['land_mask']

    epoch_losses = []
    for batch in train_dataloader:
        loss = direct.training_step(batch, mask_sampler, land_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
    
    return epoch_losses

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, metadata = get_dataloaders(
        data_path=cfg["data"]["data_path"],
        cloud_coverage_threshold=cfg["data"]["cloud_coverage_threshold"],
        batch_size=cfg["data"]["batch_size"],
        time_win=cfg["data"]["time_win"],
        mask_in_cond=cfg["data"]["mask_in_cond"],
        n_samples=cfg["data"]["n_samples"],
        standardize=cfg["data"]["standardize"],
        use_onehot=cfg["data"]["use_onehot"],
        max_delta=cfg["data"]["max_delta"],
    )

    del dataloaders["test"]
    del dataloaders["val"]

    # wandb.init(
    #     project="DIRECT",
    #     name=cfg["model_settings"]["model_name"],
    #     config=cfg,
    # )

    obs, _, _ = next(iter(dataloaders['train']))
    B, C, W, H = obs.shape

    model_params = {
        **cfg['direct_params'],
        'cond_channels': cfg["data"]["time_win"],
        "mask_ch": (cfg["data"]["time_win"] - 1) * cfg["data"]["max_delta"] + 2,
        'w': W,
        'h': H,
    }

    mask_sampler = MaskSampler(
        dataset=dataloaders['train'].dataset,
    )
    mask_sampler.to(device)

    model_config = {
        'model_params': model_params,
        'time_win': cfg["data"]["time_win"],
        'max_delta': cfg["data"]["max_delta"],
        'feat_to_idx': metadata['feat_to_idx'],
    }

    direct = Direct(model_config, device)

    optimizer = torch.optim.AdamW(
        direct.model.parameters(), lr=float(cfg["training"]["learning_rate"]), betas=cfg["training"]["optimizer_betas"]
    )

    epochs = cfg["training"]["epochs"]

    if cfg["training"]["decay_lr"]:
        warmup_epochs = int(0.1 * epochs)
        main_epochs = epochs - warmup_epochs
        print(f"Warmup: {warmup_epochs}, Cosine Decay: {main_epochs}")

        warmup_scheduler = LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-8)

        lr_schedule = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    
    save_dir = Path(cfg["model_settings"]["save_path"])
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(save_dir / 'configs').mkdir(parents=True, exist_ok=True)
    save_name = f"{cfg["model_settings"]["model_name"]}.pt"

    print()
    print("Begining Training:")
    print("----------------------")
    for epoch in range(epochs):
        t0 = time.time()

        epoch_losses = train_one_epoch(
            direct,
            dataloaders['train'],
            metadata,
            optimizer,
            mask_sampler,
        )
        epoch_loss = np.mean(epoch_losses)

        print(
            f"epoch: {epoch} \t dt: {time.time() - t0}[sec] \t train_loss: {epoch_loss}",
            flush=True,
        )

        # current_lr = optimizer.param_groups[0]['lr']
        # wandb.log({
        #     "epoch": epoch,
        #     "train_loss": epoch_loss,
        #     "learning_rate": current_lr
        # })

        if cfg["training"]["decay_lr"]:
            lr_schedule.step()

        if (epoch+1) % cfg["model_settings"]["save_period"] == 0:
            direct.save_pretrained(save_dir / save_name)
            save_config(cfg, save_dir / 'configs/train_config.yaml')
    
    direct.save_pretrained(save_dir / save_name)
    save_config(cfg, save_dir / 'configs/train_config.yaml')
    # wandb.finish()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to .yaml training config file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print(f"Script ran at time: {time.strftime('%d %m %Y %H:%M:%S', time.localtime())}")
    args = parse_arguments()
    cfg = load_config(args.config)
    train(cfg)
    print(f"Script finished at time: {time.strftime('%d %m %Y %H:%M:%S', time.localtime())}")