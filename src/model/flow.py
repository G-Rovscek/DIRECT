import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from ..utils.flow_matching import path_sample
from .backbone import DirectUnet

class Direct(nn.Module):
    def __init__(self, config: Dict, device: torch.device):
        super().__init__()
        self.cfg = config
        self.device = device
        
        self.model = DirectUnet(**config['model_params']).to(device)
        
        self.cond_channels = config['time_win']
        self.max_delta = config.get('max_delta', 3)
        self.feat_map = config['feat_to_idx']
        
        # Define slice indices for one-hot mask once
        self.masks_oh_end = self.cond_channels + (config['time_win'] - 1) * self.max_delta

    def get_one_hot_mask0(self, mask0):
        """Helper to create one-hot mask for the center frame."""
        mask0 = torch.tensor(mask0, dtype=torch.int64)
        one_hot = F.one_hot(mask0, num_classes=2)
        return one_hot.permute(0, 3, 1, 2).float().to(self.device)

    def get_conditioning(self, observation, sampled_mask):
        """
        Builds the dictionary of conditioning tensors.
        Args:
            observation: [B, C, H, W]
            sampled_mask: [B, H, W]
        """
        c_fields = observation[:, :self.cond_channels, :, :].clone()
        c_fields[:, self.feat_map['measurement_0'], :, :] *= sampled_mask

        c_sin = observation[:, self.feat_map['doy_sin'], 0, 0]
        c_cos = observation[:, self.feat_map['doy_cos'], 0, 0]
        c_doy = torch.cat((c_sin[:, None], c_cos[:, None]), dim=-1)

        mask_0 = observation[:, self.feat_map['mask_0']] * sampled_mask
        mask_0_oh = self.get_one_hot_mask0(mask_0)

        oh_masks = observation[:, self.cond_channels:self.masks_oh_end, :, :]

        conditioning = {
            'concat_conditioning': c_fields,
            'sincos_doy': c_doy,
            'one_hot_mask': torch.cat((oh_masks, mask_0_oh), dim=1)
        }
        return conditioning

    def training_step(self, batch, mask_sampler, land_mask):
        """
        Standard Training Step for Batch.
        """
        observation, missing_mask, *_ = batch
        observation = observation.to(self.device)       # [B, C, W, H]
        missing_mask = missing_mask.to(self.device)     # [B, W, H]
        land_mask = land_mask.unsqueeze(0).unsqueeze(0).to(self.device)
        
        B = observation.shape[0]

        sampled_mask, _ = mask_sampler(B)
        sampled_mask = sampled_mask.to(self.device) # [B, H, W]

        conditioning = self.get_conditioning(observation, sampled_mask)
        
        idx_m0 = self.feat_map['measurement_0']
        x1 = observation[:, idx_m0, :, :].unsqueeze(1)

        x0 = torch.randn_like(x1)

        # FUSE #
        vis_mask = land_mask * missing_mask.unsqueeze(1) * sampled_mask.unsqueeze(1)
        x0 = x0 * (1 - vis_mask) + vis_mask * x1
        x0 = x0 * land_mask

        t = torch.rand(B, device=self.device)
        xt, ut = path_sample(x0, x1, t)

        vt = self.model(xt, t, conditioning)

        loss = F.mse_loss(ut, vt, reduction='none')
        loss_mask = (x1 != 0).float() 
        loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        return loss 

    @torch.no_grad()
    def reconstruct(self, observation, vis_mask, land_mask, steps=25, num_avg=16):
        """
        Inference for a SINGLE sample, repeated `num_avg` times.
        
        Args:
            observation: [1, C, H, W]
            vis_mask: [1, H, W] (Visible pixels 1, else 0)
            steps: ODE steps
            num_avg: Number of stochastic noise seeds
        """
        W, H = observation.shape[-2:]

        observation = observation.to(self.device)
        vis_mask = vis_mask.to(self.device)
        land_mask = land_mask.unsqueeze(0).unsqueeze(0).to(self.device)
        
        obs_expanded = observation.expand(num_avg, -1, -1, -1)     # [K, C, H, W]
        vis_mask_expanded = vis_mask.expand(num_avg, -1, -1)      # [K, H, W]

        conditioning = self.get_conditioning(obs_expanded, vis_mask_expanded)
        
        idx_m0 = self.feat_map['measurement_0']

        xt = torch.randn(num_avg, 1, W, H, device=self.device)

        x1 = obs_expanded[:, idx_m0, :, :].unsqueeze(1)
        vis_mask_expanded = vis_mask_expanded.unsqueeze(1) # [K, 1, H, W]

        t_schedule = torch.linspace(0.0, 1.0, steps, device=self.device)
        
        for i in range(len(t_schedule) - 1):
            t = t_schedule[i]
            t_next = t_schedule[i+1]
            dt = t_next - t

            # FUSE #
            xt = xt * (1 - vis_mask_expanded) + vis_mask_expanded * x1
            xt = xt * land_mask
            
            t_batch = t.expand(num_avg)
            pred_v = self.model(xt, t_batch, conditioning)
            
            xt = xt + dt * pred_v

        return xt
    
    # SAVE and LOAD Utils
    def save_pretrained(self, path):
        print(f"Saving model to {path}")
        torch.save({
            'state_dict': self.model.state_dict(),
            'config': self.cfg,
        }, path)

    @classmethod
    def load_from_pretrained(cls, path, device):
        print(f"Loading model from {path}")
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'], device)
        model.model.load_state_dict(checkpoint['state_dict'])
        return model