# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# Modified for the PhysFlowNet research release on 2026-11-26.
"""PhysFlowNet modules for physics-conditioned feature rectification."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .block import C3k2
from ultralytics.nn.modules.head import Detect

def get_safe_groups(channels, default_groups=32):
    if channels % default_groups == 0:
        return default_groups
    candidates = [16, 8, 4, 2]
    for g in candidates:
        if channels % g == 0:
            return g
    return 1

class HPCE(nn.Module):
    def __init__(self, image_size=640, embed_dim=128):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim

        self._cache_key = None
        self._cache_val = None

        self.num_bands = 8
        self.register_buffer(
            'band_masks',
            self._create_band_masks(image_size, self.num_bands)
        )

        self.band_gate = nn.Parameter(
            torch.ones(1, self.num_bands)
        )

        input_dim = self.num_bands + 1 + 3 + 3

        self.phys_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, embed_dim),
            nn.Sigmoid()
        )

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k in ['_cache_key', '_cache_val']:
                setattr(new_obj, k, None)
            else:
                setattr(new_obj, k, deepcopy(v, memo))
        return new_obj

    def _create_band_masks(self, size, n_bands):
        h, w = size, size
        y, x = torch.meshgrid(
            torch.arange(h),
            torch.arange(w // 2 + 1),
            indexing='ij'
        )
        freq_radius = torch.sqrt(x**2 + y**2)
        max_radius = freq_radius.max()
        masks = []
        limits = np.logspace(
            0, np.log10(max_radius.item()), n_bands + 1
        )
        limits[0] = 0
        for i in range(n_bands):
            mask = (freq_radius >= limits[i]) & (freq_radius < limits[i+1])
            mask = mask.float()
            mask = mask / (mask.sum() + 1e-6)
            masks.append(mask)
        return torch.stack(masks)

    def extract_physics_proxies(self, x):
        x = x.float()
        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
            x_small = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )
        else:
            x_small = x

        mean_rgb = torch.mean(x_small, dim=(2, 3))
        std_rgb  = torch.std(x_small, dim=(2, 3), unbiased=False)

        x_gray = x_small.mean(dim=1)
        fft_x = torch.fft.rfft2(x_gray, norm='backward')
        power_spectrum = torch.abs(fft_x) ** 2

        B = x_small.shape[0]
        freq_feats = []
        for i in range(self.num_bands):
            mask = self.band_masks[i].unsqueeze(0).expand(B, -1, -1)
            band_energy = torch.log(
                (power_spectrum * mask).sum(dim=(1, 2)) + 1e-6
            )
            freq_feats.append(band_energy)
        freq_feats = torch.stack(freq_feats, dim=1)

        freq_feats = freq_feats * self.band_gate

        std_gray = torch.std(x_gray, dim=(1, 2), unbiased=False).unsqueeze(1)

        return torch.cat([freq_feats, std_gray, mean_rgb, std_rgb], dim=1)

    def clear_cache(self):
        self._cache_key = None
        self._cache_val = None

    def _hook_clear_cache(self, grad):
        self.clear_cache()
        return None

    def forward(self, x):
        if not self.training:
            with torch.no_grad():
                proxies = self.extract_physics_proxies(x)
            return self.phys_encoder(proxies)

        ver = getattr(x, "_version", 0)
        key = (
            x.untyped_storage().data_ptr(),
            tuple(x.shape),
            ver,
            x.device
        )

        if self._cache_key == key and self._cache_val is not None:
            return self._cache_val

        self._cache_key = key
        with torch.no_grad():
            proxies = self.extract_physics_proxies(x)
        embed = self.phys_encoder(proxies)

        if embed.requires_grad:
            embed.register_hook(self._hook_clear_cache)

        self._cache_val = embed
        return embed

class AdaGN(nn.Module):
    def __init__(self, channels, cond_dim, num_groups=32):
        super().__init__()
        safe_groups = get_safe_groups(channels, num_groups)
        self.group_norm = nn.GroupNorm(
            safe_groups, channels, affine=False
        )
        self.cond_proj = nn.Sequential(
            nn.Conv2d(cond_dim, channels * 2, 1, 1, 0, bias=True),
            nn.SiLU()
        )
        nn.init.constant_(self.cond_proj[0].weight, 0)
        nn.init.constant_(self.cond_proj[0].bias, 0)

    def forward(self, x, condition):
        if condition.dim() == 2:
            condition = condition.view(
                condition.shape[0], condition.shape[1], 1, 1
            ).clone()
        else:
            condition = condition.clone()
        normed_x = self.group_norm(x)
        style = self.cond_proj(condition)
        if style.shape[-2:] != x.shape[-2:]:
             style = style.expand(-1, -1, x.shape[-2], x.shape[-1])
        scale, shift = style.chunk(2, dim=1)
        return normed_x * (1 + scale) + shift

class DiagonalSPDMetricEstimator(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.cond_proj = nn.Conv2d(cond_dim, channels, 1)
        hidden_dim = int(channels * 1.5)
        h_groups = get_safe_groups(hidden_dim, 8)

        self.metric_network = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_dim, 1),
            nn.GroupNorm(h_groups, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(h_groups, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, channels, 1)
        )

        nn.init.constant_(self.metric_network[-1].weight, 0)
        nn.init.constant_(self.metric_network[-1].bias, 0)

    def forward(self, x, condition):
        B, C, H, W = x.shape
        if condition.dim() == 2:
            condition = condition.view(B, -1, 1, 1).clone()
        else:
            condition = condition.clone()
        cond_spatial = self.cond_proj(condition)
        if cond_spatial.shape[-2:] != (H, W):
            cond_spatial = cond_spatial.expand(-1, -1, H, W)
        combined = torch.cat([x, cond_spatial], dim=1)
        h = self.metric_network(combined)

        G_diag = 1.0 + torch.tanh(h) * 0.9
        G_diag = torch.clamp(G_diag, min=0.1, max=10.0)
        return G_diag

class PhysicsRiemannianFlowBlock(nn.Module):
    def __init__(self, channels, cond_dim=128):
        super().__init__()
        self.v_conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.adagn1 = AdaGN(channels, cond_dim)
        self.v_conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.adagn2 = AdaGN(channels, cond_dim)
        self.v_projector = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Tanh()
        )
        self.metric_estimator = DiagonalSPDMetricEstimator(channels, cond_dim)

        self.internal_phys_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, cond_dim),
            nn.LayerNorm(cond_dim)
        )

        self.out_norm = nn.GroupNorm(
            get_safe_groups(channels), channels
        )

    def riemannian_retraction_step(self, x, v, G_diag):
        preconditioner = torch.rsqrt(G_diag + 1e-6)
        return x + v * preconditioner

    def forward(self, x, condition=None):
        if condition is None:
            condition = self.internal_phys_proj(x)

        v = self.v_conv1(x)
        v = self.adagn1(v, condition)
        v = F.silu(v)
        v = self.v_conv2(v)
        v = self.adagn2(v, condition)
        velocity = self.v_projector(v)

        G_diag = self.metric_estimator(x, condition)

        x_rectified = self.riemannian_retraction_step(
            x, velocity, G_diag
        )

        return self.out_norm(x_rectified)

class PPRB(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, phys_ctx=None):
        super().__init__()
        self.main_branch = C3k2(c1, c2, n, c3k, e, g, shortcut)

        if phys_ctx is not None:
            self.sdpe = phys_ctx
        else:
            self.sdpe = HPCE(embed_dim=128)

        self.local_extractor = nn.Sequential(
            nn.Conv2d(c2, c2, 5, 1, 2, groups=c2, bias=False),
            nn.Conv2d(c2, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.riemannian_block = PhysicsRiemannianFlowBlock(
            c2, cond_dim=128 + 64
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, 1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))
        self.loss_proj = nn.Linear(c2, 128) if c2 != 128 else nn.Identity()
        self.manifold_loss = torch.tensor(0.0)

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k == 'manifold_loss':
                setattr(new_obj, k, torch.tensor(0.0))
            else:
                setattr(new_obj, k, deepcopy(v, memo))
        return new_obj

    def forward(self, x, img=None):
        if isinstance(x, list) and len(x) == 2:
            x_main, img = x[0], x[1]
        else:
            x_main = x

        x_out = self.main_branch(x_main)
        B, C, H, W = x_out.shape

        if img is None:
            x_rectified = self.riemannian_block(x_out, condition=None)
            self.manifold_loss = torch.tensor(0.0, device=x_main.device)
        else:
            global_prior = self.sdpe(img)
            global_map = global_prior.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
            local_map = self.local_extractor(x_out)
            spatial_condition = torch.cat([global_map, local_map], dim=1)
            x_rectified = self.riemannian_block(x_out, condition=spatial_condition)

            if self.training:
                z_pool = F.adaptive_avg_pool2d(x_rectified, (1, 1)).flatten(1)
                z_embed = self.loss_proj(z_pool)
                z_norm = F.normalize(z_embed, dim=1)
                p_norm = F.normalize(global_prior.detach(), dim=1)
                sim = (z_norm * p_norm).sum(dim=1)
                self.manifold_loss = 1.0 - sim.mean()
            else:
                self.manifold_loss = torch.tensor(0.0, device=x_main.device)

        displacement = x_rectified - x_out
        gate_input = torch.cat([x_out, x_rectified], dim=1)
        gate = self.gate_conv(gate_input)
        return x_out + self.gamma * gate * displacement

class PhysFlowNetNeckRefinement(nn.Module):
    def __init__(self, c1, c2=None, e=0.5, shortcut=True, phys_ctx=None):
        super().__init__()
        c2 = c2 if c2 else c1
        self.c = int(c1 * e)
        self.ablation_mode = 'full'

        if phys_ctx is not None:
            self.sdpe = phys_ctx
        else:
            self.sdpe = HPCE(embed_dim=128)

        self.flow_block = PhysicsRiemannianFlowBlock(self.c, cond_dim=128)
        self.cv1 = nn.Conv2d(c1, self.c, 1, 1)
        self.cv2 = nn.Conv2d(self.c, c2, 1, 1)
        self.add = shortcut and c1 == c2
        self.dummy_cond = nn.Parameter(torch.randn(1, 128))
        self.loss_proj = nn.Linear(self.c, 128) if self.c != 128 else nn.Identity()
        self.manifold_loss = torch.tensor(0.0)

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k == 'manifold_loss':
                setattr(new_obj, k, torch.tensor(0.0))
            else:
                setattr(new_obj, k, deepcopy(v, memo))
        return new_obj

    def forward(self, x, img=None):
        if isinstance(x, list) and len(x) == 2:
            x_main, img = x[0], x[1]
        else:
            x_main = x

        identity = x_main
        y = self.cv1(x_main)
        current_loss = torch.tensor(0.0, device=x_main.device)

        if img is None:
            condition = self.dummy_cond.expand(y.shape[0], -1)
        else:
            condition = self.sdpe(img)

        if self.ablation_mode == 'full':
            y_refined = self.flow_block(y, condition)
            if self.training and img is not None:
                y_pool = F.adaptive_avg_pool2d(y_refined, (1, 1)).flatten(1)
                y_embed = self.loss_proj(y_pool)
                z_norm = F.normalize(y_embed, dim=1)
                p_norm = F.normalize(condition.detach(), dim=1)
                sim = (z_norm * p_norm).sum(dim=1)
                current_loss = 1.0 - sim.mean()
            y = y_refined
        elif self.ablation_mode == 'euclidean':
            y = self.flow_block(y, condition)
        elif self.ablation_mode == 'diffusion_only':
            dummy_c = self.dummy_cond.expand(y.shape[0], -1)
            y = self.flow_block(y, dummy_c)

        self.manifold_loss = current_loss

        H0, W0 = y.shape[-2:]
        if H0 % 2 != 0 or W0 % 2 != 0:
            y = F.pad(y, (0, W0 % 2, 0, H0 % 2), mode='reflect')
        out = self.cv2(y)
        if out.shape[-2:] != (H0, W0):
            out = out[..., :H0, :W0]
        if self.add:
            return identity + out
        return out

class PMD(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.main_conv = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.bn_main = nn.BatchNorm2d(c2)
        g_c1 = get_safe_groups(c1 // 4, 4)
        self.phys_estimator = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, 1),
            nn.GroupNorm(g_c1, c1 // 4),
            nn.SiLU(),
            nn.Conv2d(c1 // 4, c1, 1),
            nn.Sigmoid()
        )
        self.manifold_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.geo_proj = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        x_geom = self.bn_main(self.main_conv(x))
        phys_weight = self.phys_estimator(x)
        x_enhanced = x * (1.0 + phys_weight)
        x_phys = self.geo_proj(self.manifold_pool(x_enhanced))
        return self.act(x_geom + x_phys)

class PhysFlowNetDetect(Detect):
    """
    PhysFlowNet detection head.
    继承原生 Detect。
    训练时返回 (logits, raw_features) 供额外损失使用。
    推理时保持原生输出接口。
    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)

    def forward(self, x):
        x_raw = [xi.clone() for xi in x]

        if self.end2end:
            return self.forward_end2end(x, x_raw)

        for i in range(self.nl):
            x[i] = torch.cat(
                (self.cv2[i](x[i]), self.cv3[i](x[i])), 1
            )

        if self.training:
            return (x, x_raw)

        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x, x_raw):
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat(
                (self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1
            )
            for i in range(self.nl)
        ]

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return {
                "one2many": x,
                "one2one": one2one,
                "raw_features": x_raw
            }

        y = self._inference(one2one)
        y = self.postprocess(
            y.permute(0, 2, 1), self.max_det, self.nc
        )
        return y if self.export else (y, {"one2many": x, "one2one": one2one})
