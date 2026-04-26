"""
Depth estimation head with UNet-style decoder
Takes FPN features and progressively upsamples with skip connections
Outputs depth in log space
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthDecoder(nn.Module):
    """
    UNet-style decoder for depth estimation
    Takes 3 FPN feature maps and progressively upsamples them
    """
    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (256, 512, 1024),  # FPN output channels (low to high res)
        out_channels: int = 1,  # depth is single channel
        act: str = "relu",
        skip_quarter_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels  # (256, 512, 1024) for /8, /16, /32
        self.skip_quarter_channels = skip_quarter_channels
        
        # Activation function
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {act}")
        
        # Starting from lowest resolution (1024 channels at /32)
        # Upsample to /16
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels[2], 512, 3, padding=1),
            nn.BatchNorm2d(512),
            self.act,
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512 + in_channels[1], 256, 3, padding=1),  # 512 + 512 = 1024
            nn.BatchNorm2d(256),
            self.act,
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            self.act,
        )
        
        # Upsample to /8
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            self.act,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128 + in_channels[0], 128, 3, padding=1),  # 128 + 256 = 384
            nn.BatchNorm2d(128),
            self.act,
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            self.act,
        )
        
        # Upsample to /4
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            self.act,
        )
        conv3_in = 64 + (skip_quarter_channels or 0)
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv3_in, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            self.act,
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            self.act,
        )
        
        # Upsample to /2
        self.up4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            self.act,
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            self.act,
        )
        
        # Final depth prediction heads for multi-scale outputs
        # Output depth at multiple scales for multi-scale loss
        self.depth_head_1 = nn.Conv2d(256, out_channels, 1)  # at /16
        self.depth_head_2 = nn.Conv2d(128, out_channels, 1)  # at /8
        self.depth_head_3 = nn.Conv2d(64, out_channels, 1)   # at /4
        self.depth_head_4 = nn.Conv2d(32, out_channels, 1)   # at /2

        # /2 -> /1：深度可分离卷积 + PixelShuffle，参数量与 FLOPs 增量极小
        self.to_full_res = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 4, 1, bias=False),
            nn.PixelShuffle(2),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        fpn_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        feat_skip_quarter: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            fpn_features: tuple of (feat_low, feat_mid, feat_high)
                         corresponding to /8, /16, /32 resolutions
            feat_skip_quarter: optional backbone stage-1 feature after RNN (/4), concatenated
                               before the /4 refinement convs (same as FPN /8 upsampled to /4).
        Returns:
            depth_outputs: dict with multiple scale depth predictions in log space
        """
        feat_low, feat_mid, feat_high = fpn_features  # /8, /16, /32
        
        # Start from highest level (lowest resolution /32)
        x = self.up1(feat_high)  # 1024 -> 512
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # /32 -> /16
        
        # Skip connection with /16 feature
        x = torch.cat([x, feat_mid], dim=1)  # 512 + 512
        x = self.conv1(x)  # -> 256
        # Network outputs are normalized log-depth in [0, 1] (norm_log)
        depth_16 = torch.sigmoid(self.depth_head_1(x))  # Depth at /16
        
        # Upsample to /8
        x = self.up2(x)  # 256 -> 128
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # /16 -> /8
        
        # Skip connection with /8 feature
        x = torch.cat([x, feat_low], dim=1)  # 128 + 256
        x = self.conv2(x)  # -> 128
        depth_8 = torch.sigmoid(self.depth_head_2(x))  # Depth at /8
        
        # Upsample to /4
        x = self.up3(x)  # 128 -> 64
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # /8 -> /4
        if self.skip_quarter_channels is not None:
            assert feat_skip_quarter is not None, (
                "feat_skip_quarter is required when skip_quarter_channels is set"
            )
            if feat_skip_quarter.shape[-2:] != x.shape[-2:]:
                feat_skip_quarter = F.interpolate(
                    feat_skip_quarter,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat([x, feat_skip_quarter], dim=1)
        x = self.conv3(x)  # -> 64
        depth_4 = torch.sigmoid(self.depth_head_3(x))  # Depth at /4
        
        # Upsample to /2
        x = self.up4(x)  # 64 -> 32
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # /4 -> /2
        x = self.conv4(x)  # -> 32
        depth_2 = torch.sigmoid(self.depth_head_4(x))  # Depth at /2
        depth_1 = self.to_full_res(x)  # (B,1,H,W) full resolution, norm_log

        # Return multi-scale depth predictions (all in normalized log space)
        outputs = {
            'depth_16': depth_16,  # 1/16 resolution
            'depth_8': depth_8,    # 1/8 resolution
            'depth_4': depth_4,    # 1/4 resolution
            'depth_2': depth_2,    # 1/2 resolution
            'depth_1': depth_1,    # 1/1 resolution (finest)
        }
        
        return outputs


def build_depth_head(
    head_cfg,
    in_channels: Tuple[int, int, int],
    skip_quarter_channels: Optional[int] = None,
):
    """Build depth estimation head"""
    return DepthDecoder(
        in_channels=in_channels,
        out_channels=1,
        act=head_cfg.get('act', 'relu'),
        skip_quarter_channels=skip_quarter_channels,
    )

