import math
import torch
from torch import nn, device
import torch.nn.functional as F
from abc import abstractmethod
from typing import Tuple, Optional
from .layers import *


class DirectUnet(nn.Module):
    def __init__(self, in_channels: int, model_channels: int, out_channels: int, h: int, w: int,cond_channels: int = 0, 
                channel_mult: Tuple[int] = (1, 2, 3),
                attention_resolutions: Tuple[int] = (1, 2), num_res_blocks: int = 2, num_mask_enc_blocks: int = 0,
                num_extra_blocks: int = 0, num_input_enc_blocks: int = 0, mask_ch: int = 0, dropout: float = 0.0,
                use_scale_shift_norm: bool = True, num_heads: int = 1, num_head_channels: int = -1,
                num_heads_upsample: int = -1, use_new_attention_order: bool = True, resblock_updown: bool = False,
                conv_resample: bool = False, embed_doy: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.channel_mult = channel_mult
        self.attention_resolutions = attention_resolutions
        self.num_res_blocks = num_res_blocks
        self.num_extra_blocks = num_extra_blocks
        self.num_input_enc_blocks = num_input_enc_blocks
        self.num_mask_enc_blocks = num_mask_enc_blocks
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_new_attention_order = use_new_attention_order
        self.resblock_updown = resblock_updown
        self.conv_resample = conv_resample
        self.embed_doy = embed_doy
        self.mask_ch = mask_ch

        total_in_channels = self.in_channels + self.cond_channels

        if self.num_heads_upsample == -1:
            self.num_heads_upsample = self.num_heads

        self.time_embed_dim = self.model_channels * 4


        self.doy_embed = nn.Sequential(
            nn.Linear(2, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        ) if self.embed_doy else nn.Identity()

        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * self.model_channels)
        
        self.input_encoding = nn.ModuleList([
            TimestepEmbedSequential(
                # nn.Conv2d(total_in_channels, ch, 3, padding=1))
                nn.Conv2d(total_in_channels, ch, 1))
        ])
            

        # Optional extra ResBlocks before any downsampling
        for _ in range(self.num_input_enc_blocks):  
            self.input_encoding.append(
                TimestepEmbedSequential(
                    ResBlock_1x1(
                    # ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=ch,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                )
            )
        
        if self.mask_ch:
            self.mask_encoding = nn.ModuleList([
                TimestepEmbedSequential(
                    nn.Conv2d(mask_ch, ch, 1),
                )
            ])

            for _ in range(self.num_mask_enc_blocks):
                self.mask_encoding.append(
                TimestepEmbedSequential(
                    ResBlock_1x1(
                    # ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=ch,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                )
            )
        
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Identity())
        ])

        for _ in range(self.num_extra_blocks):  
            self.input_blocks.append(
                TimestepEmbedSequential(
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=ch,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                )
            )
        
        # input_block_chans = [ch]
        input_block_chans = [ch] * (1 + self.num_extra_blocks)
        ds = 1

        for level, mult in enumerate(self.channel_mult):
            for i in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                        )
                        if self.resblock_updown
                        else Downsample(
                            ch, self.conv_resample, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(self.model_channels * mult),
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                        )
                        if self.resblock_updown
                        else Upsample(
                            ch, self.conv_resample, out_channels=out_ch
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        for _ in range(self.num_extra_blocks):  # Match encoder additions
            ich = input_block_chans.pop()
            self.output_blocks.append(
                TimestepEmbedSequential(
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=ch,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                )
            )

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, self.out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, extra):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param extra: dict with extra
        :return: an [N x C x ...] Tensor of outputs.
        """

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(x))

        if 'sincos_doy' in extra:
            emb_doy = self.doy_embed(extra['sincos_doy'])
            emb += emb_doy

        hs = []
        h = x
        if "concat_conditioning" in extra:
            h = torch.cat([x, extra["concat_conditioning"]], dim=1)
            
        for module in self.input_encoding:
            h = module(h, emb)
        
        if "one_hot_mask" in extra:
            oh = extra["one_hot_mask"]
            for module in self.mask_encoding:
                oh = module(oh, emb)
            h = h + oh    

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        result = self.out(h)
        return result