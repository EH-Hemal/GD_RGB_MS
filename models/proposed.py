"""Exact six-channel architecture extracted from original notebook cell 40.
No dataset loading or training at import. Checkpoint keys retained.
"""
import random
import torch
from torch import nn
from torch.nn import functional as F
PREFERRED_GN_GROUPS = 8
RGB_CHANNELS = 3
SPECTRAL_CHANNELS = 3
TOTAL_INPUT_CHANNELS = 6
INPUT_CHANNEL_ORDER = ('R','G','B','Green','Red-Edge','NIR')
MODALITY_DROPOUT_PROBABILITY = 0.0

def group_norm(channels):
    groups = min(PREFERRED_GN_GROUPS, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-5, affine=True)

class SCSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel(x) + x * self.spatial(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            group_norm(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                group_norm(out_channels),
            )
        )
        self.attention = SCSE(out_channels)

    def forward(self, x):
        return self.attention(
            F.silu(self.block(x) + self.skip(x), inplace=True)
        )

class SoftGatedFusion(nn.Module):
    """Two learned weights sum to one at every spatial location."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            group_norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            group_norm(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb_feature, spectral_feature):
        weights = torch.softmax(
            self.gate(torch.cat([rgb_feature, spectral_feature], dim=1)),
            dim=1,
        )
        fused = (
            weights[:, 0:1] * rgb_feature
            + weights[:, 1:2] * spectral_feature
        )
        return self.refine(fused), weights

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = ResBlock(
            in_channels + skip_channels, out_channels, dropout
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.conv(torch.cat([x, skip], dim=1))

class AuxiliarySupervisedGatedUNetNoNDRE(nn.Module):
    def __init__(self, base=28):
        super().__init__()
        channels = [base, base * 2, base * 4, base * 8]
        c1, c2, c3, c4 = channels
        self.pool = nn.MaxPool2d(2)

        self.rgb1 = ResBlock(RGB_CHANNELS, c1)
        self.spec1 = ResBlock(SPECTRAL_CHANNELS, c1)
        self.rgb2, self.spec2 = ResBlock(c1, c2), ResBlock(c1, c2)
        self.rgb3, self.spec3 = ResBlock(c2, c3, 0.05), ResBlock(c2, c3, 0.05)
        self.rgb4, self.spec4 = ResBlock(c3, c4, 0.10), ResBlock(c3, c4, 0.10)

        self.fuse1 = SoftGatedFusion(c1)
        self.fuse2 = SoftGatedFusion(c2)
        self.fuse3 = SoftGatedFusion(c3)
        self.fuse4 = SoftGatedFusion(c4)

        # Training-only heads. Their outputs are requested explicitly by
        # return_aux=True; ordinary inference does not execute them.
        self.rgb_auxiliary_head = nn.Conv2d(c4, 1, kernel_size=1)
        self.spectral_auxiliary_head = nn.Conv2d(c4, 1, kernel_size=1)

        self.center = ResBlock(c4, c4 * 2, 0.10)
        self.up4 = UpBlock(c4 * 2, c4, c4, 0.10)
        self.up3 = UpBlock(c4, c3, c3, 0.05)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.output = nn.Conv2d(c1, 1, 1)

    def modality_dropout(self, rgb, spectral):
        if not self.training or MODALITY_DROPOUT_PROBABILITY <= 0:
            return rgb, spectral
        draw = random.random()
        if draw < MODALITY_DROPOUT_PROBABILITY / 2:
            rgb = torch.zeros_like(rgb)
        elif draw < MODALITY_DROPOUT_PROBABILITY:
            spectral = torch.zeros_like(spectral)
        return rgb, spectral

    def forward(self, x, return_gates=False, return_aux=False):
        if x.ndim != 4 or x.shape[1] != TOTAL_INPUT_CHANNELS:
            raise ValueError(
                "Expected input shaped [B, 6, H, W] in the order "
                f"{INPUT_CHANNEL_ORDER}; received {tuple(x.shape)}."
            )
        input_size = x.shape[-2:]
        rgb, spectral = self.modality_dropout(
            x[:, :RGB_CHANNELS], x[:, RGB_CHANNELS:]
        )

        r1, s1 = self.rgb1(rgb), self.spec1(spectral)
        f1, g1 = self.fuse1(r1, s1)

        r2 = self.rgb2(self.pool(r1 + f1))
        s2 = self.spec2(self.pool(s1 + f1))
        f2, g2 = self.fuse2(r2, s2)

        r3 = self.rgb3(self.pool(r2 + f2))
        s3 = self.spec3(self.pool(s2 + f2))
        f3, g3 = self.fuse3(r3, s3)

        r4 = self.rgb4(self.pool(r3 + f3))
        s4 = self.spec4(self.pool(s3 + f3))
        f4, g4 = self.fuse4(r4, s4)

        center = self.center(self.pool(f4))
        d4 = self.up4(center, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        logits = self.output(d1)

        gates = [g1, g2, g3, g4]
        if return_aux:
            rgb_auxiliary_logits = F.interpolate(
                self.rgb_auxiliary_head(r4),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
            spectral_auxiliary_logits = F.interpolate(
                self.spectral_auxiliary_head(s4),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
            if return_gates:
                return (
                    logits,
                    gates,
                    rgb_auxiliary_logits,
                    spectral_auxiliary_logits,
                )
            return logits, rgb_auxiliary_logits, spectral_auxiliary_logits

        return (logits, gates) if return_gates else logits
