from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class SafeAvgPool3d(nn.Module):

    def __init__(self, kernel_size: Sequence[int]):
        super().__init__()
        self.kernel_size = tuple(int(v) for v in kernel_size)
        if len(self.kernel_size) != 3:
            raise ValueError("kernel_size must be a 3-element sequence.")

    @staticmethod
    def _safe_odd_kernel(k: int, dim: int) -> int:
        k = int(k)
        dim = int(dim)
        if dim <= 1:
            return 1
        if k <= dim:
            return k if k % 2 == 1 else max(1, k - 1)
        return dim if dim % 2 == 1 else max(1, dim - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d, h, w = x.shape[-3:]
        kd = self._safe_odd_kernel(self.kernel_size[0], d)
        kh = self._safe_odd_kernel(self.kernel_size[1], h)
        kw = self._safe_odd_kernel(self.kernel_size[2], w)
        kernel = (kd, kh, kw)
        padding = (kd // 2, kh // 2, kw // 2)
        return F.avg_pool3d(
            x,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )


class Conv3DBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = bool(residual)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True),
        )
        self.residual_conv = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if self.residual else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return self.conv(x) + self.residual_conv(x)
        return self.conv(x)


class UpBlock(nn.Module):

    def __init__(self, in_channels: int, scale_factor: Tuple[int, int, int]):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="trilinear", align_corners=True)
        self.conv = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class DepthwisePointwiseConv3D(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: Sequence[int] | int = 3,
        padding: Sequence[int] | int = 1,
        norm: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            )
        ]
        if norm:
            layers.append(nn.InstanceNorm3d(channels, affine=True))
        layers.append(nn.ReLU(inplace=True))
        layers.extend([
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AnisotropicCrescentContext3D(nn.Module):
    """Anisotropic context branch for elongated crescent-like hematoma features."""

    def __init__(self, channels: int, xy_kernel: int = 7, z_kernel: int = 3):
        super().__init__()
        xy_pad = xy_kernel // 2
        z_pad = z_kernel // 2
        self.x_context = DepthwisePointwiseConv3D(
            channels,
            kernel_size=(1, 1, xy_kernel),
            padding=(0, 0, xy_pad),
        )
        self.y_context = DepthwisePointwiseConv3D(
            channels,
            kernel_size=(1, xy_kernel, 1),
            padding=(0, xy_pad, 0),
        )
        self.z_context = DepthwisePointwiseConv3D(
            channels,
            kernel_size=(z_kernel, 1, 1),
            padding=(z_pad, 0, 0),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 3, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([
            self.x_context(x),
            self.y_context(x),
            self.z_context(x),
        ], dim=1))


class AFSEBlock3D(nn.Module):

    def __init__(
        self,
        channels: int,
        low_kernels: Sequence[Sequence[int]] = ((1, 3, 3), (1, 5, 5), (1, 7, 7)),
        anisotropic_xy_kernel: int = 7,
        anisotropic_z_kernel: int = 3,
        use_gate: bool = True,
        init_gamma: float = 0.1,
    ):
        super().__init__()
        self.low_pools = nn.ModuleList([SafeAvgPool3d(k) for k in low_kernels])
        self.low_branches = nn.ModuleList([
            DepthwisePointwiseConv3D(channels, kernel_size=3, padding=1)
            for _ in low_kernels
        ])
        self.high_branch = DepthwisePointwiseConv3D(channels, kernel_size=3, padding=1)
        self.spatial_branch = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.aniso_branch = AnisotropicCrescentContext3D(
            channels,
            xy_kernel=anisotropic_xy_kernel,
            z_kernel=anisotropic_z_kernel,
        )

        branch_count = 1 + len(low_kernels) + 1 + 1
        fuse_in_channels = channels * branch_count
        self.fuse = nn.Sequential(
            nn.Conv3d(fuse_in_channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv3d(fuse_in_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        ) if use_gate else None
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lows = [pool(x) for pool in self.low_pools]
        low_features = [branch(low) for branch, low in zip(self.low_branches, lows)]
        high = x - lows[0]
        high_feature = self.high_branch(high)
        spatial_feature = self.spatial_branch(x)
        aniso_feature = self.aniso_branch(x)
        cat = torch.cat([spatial_feature] + low_features + [high_feature, aniso_feature], dim=1)
        fused = self.fuse(cat)
        if self.gate is not None:
            fused = fused * self.gate(cat)
        return x + self.gamma * fused


class CrescentGlobalBlock3D(nn.Module):

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        xy_kernel: int = 7,
        z_kernel: int = 3,
        init_beta: float = 0.05,
    ):
        super().__init__()
        hidden = max(channels // int(reduction), 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.axial_context = AnisotropicCrescentContext3D(channels, xy_kernel=xy_kernel, z_kernel=z_kernel)
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.axial_context(x)
        context = context * self.channel_gate(x)
        context = self.refine(context)
        return x + self.beta * context

class ResidualEdgeEnhancement3D(nn.Module):

    def __init__(
        self,
        channels: int,
        avg_kernel: Sequence[int] = (1, 3, 3),
        init_alpha: float = 0.0,
        use_attention: bool = True,
    ):
        super().__init__()
        self.low_pool = SafeAvgPool3d(avg_kernel)
        self.edge_conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.edge_attention = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        ) if use_attention else None
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low_pool(x)
        high = x - low
        edge = self.edge_conv(high)
        if self.edge_attention is not None:
            edge = edge * self.edge_attention(torch.cat([x, high], dim=1))
        return x + self.alpha * edge


class AlignedEdgeEnhancement3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        fusion_channels: int,
        edge_init_alpha: float = 0.0,
        use_edge: bool = True,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv3d(in_channels, fusion_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(fusion_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.edge_enhancement = ResidualEdgeEnhancement3D(
            fusion_channels,
            avg_kernel=(1, 3, 3),
            init_alpha=edge_init_alpha,
            use_attention=True,
        ) if use_edge else nn.Identity()

    @staticmethod
    def align_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:] == ref.shape[-3:]:
            return x
        return F.interpolate(x, size=ref.shape[-3:], mode="trilinear", align_corners=True)

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.align_like(x, ref)
        x = self.edge_enhancement(x)
        return x


class UncertaintyGate3D(nn.Module):
    def __init__(self, power: float = 1.0, detach: bool = True):
        super().__init__()
        self.power = float(power)
        self.detach = bool(detach)

    def forward(self, base_logits: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(base_logits)
        uncertainty = 4.0 * prob * (1.0 - prob)
        uncertainty = torch.clamp(uncertainty, min=0.0, max=1.0)
        if self.power != 1.0:
            uncertainty = uncertainty.pow(self.power)
        if self.detach:
            uncertainty = uncertainty.detach()
        return uncertainty


class ScaleWeightedFusion3D(nn.Module):
    def __init__(self, fusion_channels: int, num_scales: int):
        super().__init__()
        self.num_scales = int(num_scales)
        self.weight_predictor = nn.Sequential(
            nn.Conv3d(fusion_channels * num_scales, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(fusion_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(fusion_channels, num_scales, kernel_size=1, bias=True),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) == 1:
            return features[0]
        if len(features) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} features, got {len(features)}.")
        cat = torch.cat(list(features), dim=1)
        weights = torch.softmax(self.weight_predictor(cat), dim=1)
        fused = torch.zeros_like(features[0])
        for i, f in enumerate(features):
            fused = fused + weights[:, i:i + 1] * f
        return fused


class UGRBlock3D(nn.Module):
    def __init__(
        self,
        stage_channels: Dict[int, int],
        output_ch: int = 1,
        selected_stages: Sequence[int] = (10, 11),
        fusion_channels: int = 8,
        init_alpha: float = 0.0,
        edge_init_alpha: float = 0.0,
        use_edge: bool = True,
        uncertainty_power: float = 1.0,
        uncertainty_detach: bool = True,
    ):
        super().__init__()
        self.stage_channels = {int(k): int(v) for k, v in stage_channels.items()}
        self.selected_stages = tuple(int(s) for s in selected_stages)
        self.fusion_channels = int(fusion_channels)
        if len(self.selected_stages) < 1:
            raise ValueError("selected_stages must contain at least one stage.")
        for s in self.selected_stages:
            if s not in self.stage_channels:
                raise ValueError(f"Stage {s} is not available. Available stages: {sorted(self.stage_channels)}")

        self.aligned_edge = nn.ModuleDict({
            str(s): AlignedEdgeEnhancement3D(
                in_channels=self.stage_channels[s],
                fusion_channels=self.fusion_channels,
                edge_init_alpha=edge_init_alpha,
                use_edge=use_edge,
            ) for s in self.selected_stages
        })
        self.uncertainty_gate = UncertaintyGate3D(power=uncertainty_power, detach=uncertainty_detach)
        self.scale_fusion = ScaleWeightedFusion3D(
            fusion_channels=self.fusion_channels,
            num_scales=len(self.selected_stages),
        )
        self.refine_head = nn.Sequential(
            nn.Conv3d(self.fusion_channels, self.fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(self.fusion_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.fusion_channels, output_ch, kernel_size=1, bias=True),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))

    @staticmethod
    def _align_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:] == ref.shape[-3:]:
            return x
        return F.interpolate(x, size=ref.shape[-3:], mode="trilinear", align_corners=True)

    def forward(self, base_logits: torch.Tensor, decoder_features: Dict[int, torch.Tensor]) -> torch.Tensor:
        if 11 in decoder_features:
            ref = decoder_features[11]
        else:
            ref = decoder_features[self.selected_stages[-1]]

        aligned_features = []
        for s in self.selected_stages:
            if s not in decoder_features:
                raise KeyError(f"decoder_features missing stage {s}.")
            aligned_features.append(self.aligned_edge[str(s)](decoder_features[s], ref))

        fused = self.scale_fusion(aligned_features)
        refine_logits = self.refine_head(fused)
        refine_logits = self._align_like(refine_logits, base_logits)

        uncertainty = self.uncertainty_gate(base_logits)
        uncertainty = self._align_like(uncertainty, refine_logits)

        return base_logits + self.alpha * uncertainty * refine_logits

class CresUNet(nn.Module):
    def __init__(
        self,
        input_ch: int = 1,
        output_ch: int = 1,
        init_feats: int = 8,
        afse_stages: Sequence[int] = (2, 3, 4, 5),
        afse_low_kernels: Sequence[Sequence[int]] = ((1, 3, 3), (1, 5, 5), (1, 7, 7)),
        afse_init_gamma: float = 0.1,
        aniso_xy_kernel: int = 7,
        aniso_z_kernel: int = 3,
        use_cgb: bool = True,
        cgb_reduction: int = 4,
        cgb_init_beta: float = 0.05,
        ugr_stages: Sequence[int] = (10, 11),
        ugr_fusion_channels: Optional[int] = None,
        ugr_init_alpha: float = 0.0,
        ugr_edge_init_alpha: float = 0.0,
        ugr_use_edge: bool = True,
        ugr_uncertainty_power: float = 1.0,
        ugr_uncertainty_detach: bool = True,
        # Backward-compatible argument names from the original file.
        cma_stages: Optional[Sequence[int]] = None,
        cma_low_kernels: Optional[Sequence[Sequence[int]]] = None,
        cma_init_gamma: Optional[float] = None,
        use_cgcb: Optional[bool] = None,
        cgcb_reduction: Optional[int] = None,
        cgcb_init_beta: Optional[float] = None,
        ugrmsf_stages: Optional[Sequence[int]] = None,
        ugrmsf_fusion_channels: Optional[int] = None,
        ugrmsf_init_alpha: Optional[float] = None,
        ugrmsf_edge_init_alpha: Optional[float] = None,
        ugrmsf_use_edge: Optional[bool] = None,
        ugrmsf_uncertainty_power: Optional[float] = None,
        ugrmsf_uncertainty_detach: Optional[bool] = None,
        **unused_kwargs,
    ):
        super().__init__()

        # Backward-compatible overrides.
        if cma_stages is not None:
            afse_stages = cma_stages
        if cma_low_kernels is not None:
            afse_low_kernels = cma_low_kernels
        if cma_init_gamma is not None:
            afse_init_gamma = cma_init_gamma
        if use_cgcb is not None:
            use_cgb = use_cgcb
        if cgcb_reduction is not None:
            cgb_reduction = cgcb_reduction
        if cgcb_init_beta is not None:
            cgb_init_beta = cgcb_init_beta
        if ugrmsf_stages is not None:
            ugr_stages = ugrmsf_stages
        if ugrmsf_fusion_channels is not None:
            ugr_fusion_channels = ugrmsf_fusion_channels
        if ugrmsf_init_alpha is not None:
            ugr_init_alpha = ugrmsf_init_alpha
        if ugrmsf_edge_init_alpha is not None:
            ugr_edge_init_alpha = ugrmsf_edge_init_alpha
        if ugrmsf_use_edge is not None:
            ugr_use_edge = ugrmsf_use_edge
        if ugrmsf_uncertainty_power is not None:
            ugr_uncertainty_power = ugrmsf_uncertainty_power
        if ugrmsf_uncertainty_detach is not None:
            ugr_uncertainty_detach = ugrmsf_uncertainty_detach

        self.afse_stages = set(int(s) for s in afse_stages)
        self.use_cgb = bool(use_cgb)

        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.up7 = UpBlock(init_feats * 32, scale_factor=(1, 2, 2))
        self.up8 = UpBlock(init_feats * 16, scale_factor=(2, 2, 2))
        self.up9 = UpBlock(init_feats * 8, scale_factor=(1, 2, 2))
        self.up10 = UpBlock(init_feats * 4, scale_factor=(2, 2, 2))
        self.up11 = UpBlock(init_feats * 2, scale_factor=(1, 2, 2))

        self.conv1 = Conv3DBlock(input_ch, init_feats)
        self.conv2 = Conv3DBlock(init_feats, init_feats * 2)
        self.conv3 = Conv3DBlock(init_feats * 2, init_feats * 4)
        self.conv4 = Conv3DBlock(init_feats * 4, init_feats * 8)
        self.conv5 = Conv3DBlock(init_feats * 8, init_feats * 16)
        self.conv6 = Conv3DBlock(init_feats * 16, init_feats * 32)

        self.afse1 = self._make_afse(init_feats, 1, afse_low_kernels, afse_init_gamma, aniso_xy_kernel, aniso_z_kernel)
        self.afse2 = self._make_afse(init_feats * 2, 2, afse_low_kernels, afse_init_gamma, aniso_xy_kernel, aniso_z_kernel)
        self.afse3 = self._make_afse(init_feats * 4, 3, afse_low_kernels, afse_init_gamma, aniso_xy_kernel, aniso_z_kernel)
        self.afse4 = self._make_afse(init_feats * 8, 4, afse_low_kernels, afse_init_gamma, aniso_xy_kernel, aniso_z_kernel)
        self.afse5 = self._make_afse(init_feats * 16, 5, afse_low_kernels, afse_init_gamma, aniso_xy_kernel, aniso_z_kernel)
        self.afse6 = self._make_afse(init_feats * 32, 6, afse_low_kernels, afse_init_gamma, aniso_xy_kernel, aniso_z_kernel)

        self.cgb = CrescentGlobalBlock3D(
            init_feats * 32,
            reduction=cgb_reduction,
            xy_kernel=aniso_xy_kernel,
            z_kernel=aniso_z_kernel,
            init_beta=cgb_init_beta,
        ) if self.use_cgb else nn.Identity()

        self.conv7 = Conv3DBlock(init_feats * 32, init_feats * 16)
        self.conv8 = Conv3DBlock(init_feats * 16, init_feats * 8)
        self.conv9 = Conv3DBlock(init_feats * 8, init_feats * 4)
        self.conv10 = Conv3DBlock(init_feats * 4, init_feats * 2)
        self.conv11 = Conv3DBlock(init_feats * 2, init_feats)

        self.base_head = nn.Conv3d(init_feats, output_ch, kernel_size=1)

        ugr_fusion_channels = init_feats if ugr_fusion_channels is None else int(ugr_fusion_channels)
        self.ugr_block = UGRBlock3D(
            stage_channels={9: init_feats * 4, 10: init_feats * 2, 11: init_feats},
            output_ch=output_ch,
            selected_stages=tuple(ugr_stages),
            fusion_channels=ugr_fusion_channels,
            init_alpha=ugr_init_alpha,
            edge_init_alpha=ugr_edge_init_alpha,
            use_edge=ugr_use_edge,
            uncertainty_power=ugr_uncertainty_power,
            uncertainty_detach=ugr_uncertainty_detach,
        )

        self.conv12 = self.base_head
        self.ugrmsf_decoder = self.ugr_block

    def _make_afse(
        self,
        channels: int,
        stage_id: int,
        low_kernels: Sequence[Sequence[int]],
        init_gamma: float,
        aniso_xy_kernel: int,
        aniso_z_kernel: int,
    ) -> nn.Module:
        if int(stage_id) in self.afse_stages:
            return AFSEBlock3D(
                channels=channels,
                low_kernels=low_kernels,
                anisotropic_xy_kernel=aniso_xy_kernel,
                anisotropic_z_kernel=aniso_z_kernel,
                use_gate=True,
                init_gamma=init_gamma,
            )
        return nn.Identity()

    @staticmethod
    def _cat_skip(skip: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if skip.shape[-3:] == up.shape[-3:]:
            return torch.cat([skip, up], dim=1)

        sd, sh, sw = skip.shape[-3:]
        ud, uh, uw = up.shape[-3:]
        target = (min(sd, ud), min(sh, uh), min(sw, uw))

        def center_crop(t: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
            d, h, w = t.shape[-3:]
            td, th, tw = target_shape
            ds = max((d - td) // 2, 0)
            hs = max((h - th) // 2, 0)
            ws = max((w - tw) // 2, 0)
            return t[..., ds:ds + td, hs:hs + th, ws:ws + tw]

        return torch.cat([center_crop(skip, target), center_crop(up, target)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder.
        conv1 = self.afse1(self.conv1(x))
        pool1 = self.pool1(conv1)

        conv2 = self.afse2(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        conv3 = self.afse3(self.conv3(pool2))
        pool3 = self.pool3(conv3)

        conv4 = self.afse4(self.conv4(pool3))
        pool4 = self.pool4(conv4)

        conv5 = self.afse5(self.conv5(pool4))
        pool5 = self.pool5(conv5)

        conv6 = self.afse6(self.conv6(pool5))
        conv6 = self.cgb(conv6)

        # Decoder.
        up7 = self.up7(conv6)
        conv7 = self.conv7(self._cat_skip(conv5, up7))

        up8 = self.up8(conv7)
        conv8 = self.conv8(self._cat_skip(conv4, up8))

        up9 = self.up9(conv8)
        conv9 = self.conv9(self._cat_skip(conv3, up9))

        up10 = self.up10(conv9)
        conv10 = self.conv10(self._cat_skip(conv2, up10))

        up11 = self.up11(conv10)
        conv11 = self.conv11(self._cat_skip(conv1, up11))

        base_logits = self.base_head(conv11)
        decoder_features = {9: conv9, 10: conv10, 11: conv11}
        final_logits = self.ugr_block(base_logits, decoder_features)
        return final_logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CresUNet(
        input_ch=1,
        output_ch=1,
        init_feats=8,
        afse_stages=(2, 3, 4, 5),
        ugr_stages=(10, 11),
        ugr_init_alpha=0.0,
    ).to(device)

    x = torch.randn(1, 1, 32, 192, 192, device=device)
    with torch.no_grad():
        y = model(x)

    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("params:", sum(p.numel() for p in model.parameters()))
