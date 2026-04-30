"""
Depth estimation head with UNet-style decoder.

Two modes are supported:

1. **FPN mode** (`DepthDecoder`): consumes 3 FPN feature maps at /8, /16, /32
   plus an optional /4 backbone skip connection. Produces multi-scale depth
   predictions at /16, /8, /4, /2, /1 (using a PixelShuffle-based /2→/1 head).

2. **No-FPN mode** (`DepthDecoderNoFPN`): consumes 4 backbone feature maps
   directly at /1, /2, /4, /8 (corresponds to backbone with downsample
   factors [1, 2, 2, 2]). Produces 3 depth predictions at /4, /2, /1 by
   successively upsampling and concatenating with the matching-resolution
   backbone feature.

The build entry point picks one based on ``head_cfg['mode']`` (defaults to
``'fpn'`` for backward compatibility).
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(channels: int, preferred: int = 8) -> int:
    g = min(preferred, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


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


class DepthDecoderNoFPN(nn.Module):
    """
    Decoder consuming backbone features at /1, /2, /4, /8 directly (no FPN).
    Produces 3 depth predictions at /4, /2, /1.

    Forward path::

        f8 ─up─> /4 ─cat(f4)─ conv1 ─head1─> depth_4
                   │
                   up
                   ▼
                  /2 ─cat(f2)─ conv2 ─head2─> depth_2
                   │
                   up
                   ▼
                  /1 ─cat(f1)─ conv3 ─head3─> depth_1
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int, int] = (32, 64, 128, 256),
        out_channels: int = 1,
        act: str = "relu",
    ):
        super().__init__()
        self.in_channels = tuple(in_channels)  # (/1, /2, /4, /8)
        c1, c2, c4, c8 = self.in_channels

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {act}")

        def gn(ch: int) -> nn.GroupNorm:
            return nn.GroupNorm(_gn_groups(ch), ch)

        # /8 -> /4
        self.up1 = nn.Sequential(
            nn.Conv2d(c8, c4, 3, padding=1, bias=False),
            gn(c4),
            self.act,
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c4 + c4, c4, 3, padding=1, bias=False),
            gn(c4),
            self.act,
            nn.Conv2d(c4, c4, 3, padding=1, bias=False),
            gn(c4),
            self.act,
        )
        self.head1 = nn.Conv2d(c4, out_channels, 1)

        # /4 -> /2
        self.up2 = nn.Sequential(
            nn.Conv2d(c4, c2, 3, padding=1, bias=False),
            gn(c2),
            self.act,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, 3, padding=1, bias=False),
            gn(c2),
            self.act,
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            gn(c2),
            self.act,
        )
        self.head2 = nn.Conv2d(c2, out_channels, 1)

        # /2 -> /1
        self.up3 = nn.Sequential(
            nn.Conv2d(c2, c1, 3, padding=1, bias=False),
            gn(c1),
            self.act,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, 3, padding=1, bias=False),
            gn(c1),
            self.act,
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            gn(c1),
            self.act,
        )
        self.head3 = nn.Conv2d(c1, out_channels, 1)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, backbone_features: Dict[int, torch.Tensor]):
        """
        Args:
            backbone_features: dict keyed by 1-based stage index. Stage 1 -> /1,
                               stage 2 -> /2, stage 3 -> /4, stage 4 -> /8.
        Returns:
            dict with 'depth_4', 'depth_2', 'depth_1' (all in norm_log space).
        """
        f1 = backbone_features[1]  # /1
        f2 = backbone_features[2]  # /2
        f4 = backbone_features[3]  # /4
        f8 = backbone_features[4]  # /8

        x = self.up1(f8)
        x = self._up_to(x, f4)
        x = torch.cat([x, f4], dim=1)
        x = self.conv1(x)
        depth_4 = torch.sigmoid(self.head1(x))

        x = self.up2(x)
        x = self._up_to(x, f2)
        x = torch.cat([x, f2], dim=1)
        x = self.conv2(x)
        depth_2 = torch.sigmoid(self.head2(x))

        x = self.up3(x)
        x = self._up_to(x, f1)
        x = torch.cat([x, f1], dim=1)
        x = self.conv3(x)
        depth_1 = torch.sigmoid(self.head3(x))

        return {
            'depth_4': depth_4,
            'depth_2': depth_2,
            'depth_1': depth_1,
        }


def build_depth_head(
    head_cfg,
    in_channels: Tuple[int, ...],
    skip_quarter_channels: Optional[int] = None,
):
    """Build depth estimation head.

    The decoder variant is selected by ``head_cfg['mode']``:
      - ``'fpn'`` (default): expects ``in_channels`` of length 3 and feeds
        the FPN-style decoder. ``skip_quarter_channels`` enables the /4
        backbone skip path.
      - ``'no_fpn'``: expects ``in_channels`` of length 4 corresponding to
        backbone features at (/1, /2, /4, /8); feeds ``DepthDecoderNoFPN``.
    """
    mode = head_cfg.get('mode', 'fpn') if isinstance(head_cfg, dict) else 'fpn'
    act = head_cfg.get('act', 'relu') if isinstance(head_cfg, dict) else 'relu'

    if mode == 'no_fpn':
        assert len(in_channels) == 4, (
            f"no_fpn depth head expects 4 backbone in_channels, got {in_channels}"
        )
        return DepthDecoderNoFPN(
            in_channels=tuple(in_channels),
            out_channels=1,
            act=act,
        )

    assert len(in_channels) == 3, (
        f"fpn depth head expects 3 in_channels, got {in_channels}"
    )
    return DepthDecoder(
        in_channels=in_channels,
        out_channels=1,
        act=act,
        skip_quarter_channels=skip_quarter_channels,
    )
