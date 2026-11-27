# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# Modified for the PhysFlowNet research release on 2026-11-26.
"""PhysFlowNet modules for physics-conditioned feature rectification."""

import math                                  # 数学库 用于标量数学运算 (scalar math)
import torch                                 # PyTorch 主库 (PyTorch core)
import torch.nn as nn                        # 神经网络模块 (nn modules)
import torch.nn.functional as F              # 函数式算子 如插值/激活 (functional ops)
import numpy as np                           # NumPy 用于生成对数分带边界等 (NumPy utilities)
from copy import deepcopy                    # 深拷贝 用于 deepcopy 修复 (deep copy)
from .block import C3k2                      # 本地模块 C3k2 主干块 (local block)
from ultralytics.nn.modules.head import Detect  # YOLO 原生 Detect Head (YOLO detect head)

# ==============================================================================
# 0. 辅助函数
# ==============================================================================
def get_safe_groups(channels, default_groups=32):   # 选择可整除的 GroupNorm 分组数 (choose GN groups)
    if channels % default_groups == 0:              # 若 channels 能被 default_groups 整除 (divisible)
        return default_groups                       # 返回默认组数 标量 int (return int)
    candidates = [16, 8, 4, 2]                      # 备选组数 从大到小 (candidates)
    for g in candidates:                            # 遍历候选组数 (loop)
        if channels % g == 0:                       # 若可整除 (divisible)
            return g                                # 返回该组数 标量 int (return int)
    return 1                                        # 最坏情况退化为 1 组 (i.e., LayerNorm-like over channels)

# ==============================================================================
# 1. 物理上下文管理器 (HPCE)
#    [Critical Fix]
#    1. 训练时：注册 backward hook 自动清理缓存，防止跨 Batch 复用死图 (clear cache on backward)
#    2. 推理时：禁用缓存，防止内存地址复用导致的特征错乱 (disable cache in eval)
# ==============================================================================
class HPCE(nn.Module):  # 定义模块 用输入图像提取“物理代理特征”并编码为 embedding (image->physics proxies->embedding)
    def __init__(self, image_size=640, embed_dim=128):  # image_size: 统一频域尺寸 embed_dim: 输出维度 (embedding dim)
        super().__init__()                              # 初始化父类 完成参数注册 (init nn.Module)
        self.image_size = image_size                    # 记录统一处理尺寸 标量 int (scalar)
        self.embed_dim = embed_dim                      # 记录 embedding 维度 标量 int (scalar)

        # Instance-level cache：同一 batch 内复用 embedding (reuse within-batch)
        self._cache_key = None                          # 缓存键 初始 None (cache key)
        self._cache_val = None                          # 缓存值 初始 None (cache value)

        # Physics: Frequency Domain 频域分带统计 (frequency bands)
        self.num_bands = 8                              # 频带数 标量 int (num bands)
        self.register_buffer(                           # 注册 buffer 不参与梯度 但随 device 迁移 (non-trainable buffer)
            'band_masks',
            self._create_band_masks(image_size, self.num_bands)
        )                                               # band_masks 形状 (num_bands, H, W//2+1) where H=W=image_size

        # Learnable scattering filter：频带门控参数 (learnable band gate)
        self.band_gate = nn.Parameter(                  # 可训练参数 (trainable)
            torch.ones(1, self.num_bands)
        )                                               # 形状 (1, num_bands) 广播到 (B, num_bands)

        input_dim = self.num_bands + 1 + 3 + 3          # 代理特征维度 = bands + gray_std + RGB_mean + RGB_std (num_bands+7)

        self.phys_encoder = nn.Sequential(              # MLP 将 (B,input_dim) -> (B,embed_dim) (MLP encoder)
            nn.Linear(input_dim, 64),                   # (B,input_dim) -> (B,64)
            nn.LayerNorm(64),                           # (B,64) -> (B,64) 归一化最后一维 (normalize last dim)
            nn.SiLU(),                                  # (B,64) -> (B,64) 激活 (activation)
            nn.Linear(64, embed_dim),                   # (B,64) -> (B,embed_dim)
            nn.Sigmoid()                                # (B,embed_dim) -> (B,embed_dim) 限幅到 (0,1)
        )

    # [Deepcopy Fix]：避免拷贝缓存张量导致复制旧计算图或触发 deepcopy 限制 (avoid copying cached tensors/graphs)
    def __deepcopy__(self, memo):                       # memo: deepcopy 内部备忘录字典 (deepcopy memo)
        cls = self.__class__                            # 当前类对象 (class handle)
        new_obj = cls.__new__(cls)                      # 只分配对象 不跑 __init__ (allocate only)
        memo[id(self)] = new_obj                        # 防止递归重复构造 (memoize)
        for k, v in self.__dict__.items():              # 遍历实例字段 (iterate attributes)
            if k in ['_cache_key', '_cache_val']:       # 如果是缓存字段 (cache fields)
                setattr(new_obj, k, None)               # 强制清空缓存 (reset cache)
            else:
                setattr(new_obj, k, deepcopy(v, memo))  # 其余字段标准深拷贝 (deepcopy others)
        return new_obj                                  # 返回新对象 (return)

    def _create_band_masks(self, size, n_bands):        # 构造 rfft2 频域分带掩膜 (create band masks in rfft space)
        h, w = size, size                               # 空间尺寸 H=W=size (spatial size)
        y, x = torch.meshgrid(                          # 频域坐标网格 (frequency grid)
            torch.arange(h),
            torch.arange(w // 2 + 1),
            indexing='ij'
        )                                               # y,x 形状 (h, w//2+1)
        freq_radius = torch.sqrt(x**2 + y**2)           # 半径 r= sqrt(x^2+y^2) 形状 (h, w//2+1)
        max_radius = freq_radius.max()                  # 最大半径 标量张量 (scalar tensor)
        masks = []                                      # 掩膜列表 (list of masks)
        limits = np.logspace(                           # 对数间隔的半径边界 (log-spaced radii)
            0, np.log10(max_radius.item()), n_bands + 1
        )                                               # numpy 数组 长度 n_bands+1
        limits[0] = 0                                   # 最低边界设为 0 覆盖 DC (include DC)
        for i in range(n_bands):                        # 对每个频带 (for each band)
            mask = (freq_radius >= limits[i]) & (freq_radius < limits[i+1])  # bool mask (h,w//2+1)
            mask = mask.float()                         # 转 float (0/1) 形状不变 (same shape)
            mask = mask / (mask.sum() + 1e-6)           # 归一化 使 mask 总和约为 1 (normalize)
            masks.append(mask)                          # 追加 (append)
        return torch.stack(masks)                       # (n_bands,h,w//2+1)

    def extract_physics_proxies(self, x):               # 从图像提取代理特征 (extract proxies)
        x = x.float()                                   # 输入转 float32 形状通常 (B,C,H,W) (cast to float)
        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:  # 若 H/W 非统一尺寸 (resize if needed)
            x_small = F.interpolate(                    # 插值到 (image_size,image_size) (interpolate)
                x, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )                                           # x_small: (B,C,image_size,image_size)
        else:
            x_small = x                                 # 已是统一尺寸 (B,C,image_size,image_size)

        mean_rgb = torch.mean(x_small, dim=(2, 3))       # RGB 空间均值 (B,C) typically (B,3)
        std_rgb  = torch.std(x_small, dim=(2, 3), unbiased=False)  # RGB 空间标准差 (B,C) typically (B,3)

        x_gray = x_small.mean(dim=1)                     # 灰度 (按通道均值) (B,image_size,image_size)
        fft_x = torch.fft.rfft2(x_gray, norm='backward') # 实 FFT 频谱 复数 (B,H,W//2+1)
        power_spectrum = torch.abs(fft_x) ** 2           # 功率谱 |F|^2 实数 (B,H,W//2+1)

        B = x_small.shape[0]                             # batch 大小 标量 int (batch size)
        freq_feats = []                                  # 各频带能量特征列表 (list)
        for i in range(self.num_bands):                  # 遍历频带 (iterate bands)
            mask = self.band_masks[i].unsqueeze(0).expand(B, -1, -1)  # (H,W//2+1)->(1,H,W//2+1)->(B,H,W//2+1)
            band_energy = torch.log(                     # 对能量取 log 稳定尺度 (log-energy)
                (power_spectrum * mask).sum(dim=(1, 2)) + 1e-6
            )                                            # (B,H,W//2+1)->(B,)
            freq_feats.append(band_energy)               # 追加 (append)
        freq_feats = torch.stack(freq_feats, dim=1)      # (list of B,) -> (B,num_bands)

        freq_feats = freq_feats * self.band_gate         # 频带门控 广播 (B,num_bands)*(1,num_bands)->(B,num_bands)

        std_gray = torch.std(x_gray, dim=(1, 2), unbiased=False).unsqueeze(1)  # gray std (B,)->(B,1)

        return torch.cat([freq_feats, std_gray, mean_rgb, std_rgb], dim=1)    # 拼接 -> (B,num_bands+1+C+C) typically (B,15)

    def clear_cache(self):                               # 手动清缓存 (manual cache clear)
        self._cache_key = None                           # 清 key (reset key)
        self._cache_val = None                           # 清 val (reset value)

    def _hook_clear_cache(self, grad):                   # backward hook: grad 形状 (B,embed_dim) (gradient tensor)
        self.clear_cache()                               # 反传开始即清缓存 防止下一 batch 误命中 (clear on backward)
        return None                                      # 不修改梯度 仅副作用 (no grad replacement)

    def forward(self, x):                                # x: (B,C,H,W) -> embed: (B,embed_dim)
        # 1) 推理模式：禁用缓存 并用 no_grad 避免构建图 (eval: disable cache, no_grad)
        if not self.training:                            # eval / inference branch
            with torch.no_grad():                        # 不追踪梯度 (no autograd)
                proxies = self.extract_physics_proxies(x)  # (B,input_dim)
            return self.phys_encoder(proxies)            # (B,input_dim)->(B,embed_dim)

        # 2) 训练模式：启用缓存 仅在同一 batch 内共享 embedding 计算图 (train: cache within-batch)
        ver = getattr(x, "_version", 0)                  # 读取 Tensor 版本号 检测 in-place 修改 (tensor version)
        key = (                                          # 构造缓存键 (cache key)
            x.untyped_storage().data_ptr(),              # 存储地址 标量 int (storage pointer)
            tuple(x.shape),                              # 形状元组 (B,C,H,W)
            ver,                                         # 版本号 int
            x.device                                     # 设备 (cpu/cuda)
        )

        if self._cache_key == key and self._cache_val is not None:  # cache hit
            return self._cache_val                        # 直接返回 (B,embed_dim)

        # cache miss: 重新计算 (recompute)
        self._cache_key = key                             # 更新 key (update key)
        with torch.no_grad():                             # 代理特征提取不回传到 x (no grad through proxies)
            proxies = self.extract_physics_proxies(x)     # (B,input_dim)
        embed = self.phys_encoder(proxies)                # (B,input_dim)->(B,embed_dim) 对 encoder 参数可求导 (grad for encoder params)

        if embed.requires_grad:                           # 通常 True (needs grad)
            embed.register_hook(self._hook_clear_cache)   # 注册 hook 在 backward 时清缓存 (register hook)

        self._cache_val = embed                           # 写入缓存值 (store embedding)
        return embed                                      # 返回 (B,embed_dim)

# ==============================================================================
# 2. 黎曼几何组件
# ==============================================================================
class AdaGN(nn.Module):                                  # 条件自适应 GroupNorm (adaptive GN via scale/shift)
    def __init__(self, channels, cond_dim, num_groups=32):
        super().__init__()                               # init (register params)
        safe_groups = get_safe_groups(channels, num_groups)  # 选择能整除 channels 的组数 (safe groups)
        self.group_norm = nn.GroupNorm(                  # GN 不使用 affine 因为用 condition 提供 scale/shift (affine=False)
            safe_groups, channels, affine=False
        )
        self.cond_proj = nn.Sequential(                  # condition -> (scale,shift) 映射 (condition projection)
            nn.Conv2d(cond_dim, channels * 2, 1, 1, 0, bias=True),  # (B,cond_dim,1,1)->(B,2C,1,1) 或 (B,cond_dim,H,W)->(B,2C,H,W)
            nn.SiLU()                                    # 激活 保持形状 (activation)
        )
        nn.init.constant_(self.cond_proj[0].weight, 0)   # 初始化为 0 -> 初始 scale/shift=0 (zero init)
        nn.init.constant_(self.cond_proj[0].bias, 0)     # 同上 (zero init)

    def forward(self, x, condition):                     # x: (B,C,H,W) condition: (B,cond_dim) or (B,cond_dim,H,W)
        if condition.dim() == 2:                         # 若 condition 是向量 (B,cond_dim)
            condition = condition.view(                  # reshape -> (B,cond_dim,1,1)
                condition.shape[0], condition.shape[1], 1, 1
            ).clone()                                    # clone 防止潜在共享/就地修改 (clone)
        else:
            condition = condition.clone()                # 若已是 (B,cond_dim,H,W) 仍 clone (clone)
        normed_x = self.group_norm(x)                    # GN: (B,C,H,W)->(B,C,H,W)
        style = self.cond_proj(condition)                # (B,cond_dim,*,*)->(B,2C,*,*)
        if style.shape[-2:] != x.shape[-2:]:             # 若 spatial 不匹配 (spatial mismatch)
             style = style.expand(-1, -1, x.shape[-2], x.shape[-1])  # broadcast 到 (B,2C,H,W)
        scale, shift = style.chunk(2, dim=1)             # 按通道分为两半: (B,C,H,W) + (B,C,H,W)
        return normed_x * (1 + scale) + shift            # 仿射调制 输出 (B,C,H,W)

class DiagonalSPDMetricEstimator(nn.Module):             # 对角 SPD 度量估计器 G=diag(g) (diagonal metric)
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.cond_proj = nn.Conv2d(cond_dim, channels, 1)  # condition -> (B,C,*,*) (project cond to channels)
        hidden_dim = int(channels * 1.5)                  # hidden 通道数 标量 int (hidden channels)
        h_groups = get_safe_groups(hidden_dim, 8)         # hidden 的 GN 分组数 (GN groups)

        self.metric_network = nn.Sequential(              # 输入 concat([x,cond]) -> 输出 h (B,C,H,W) (metric net)
            nn.Conv2d(channels * 2, hidden_dim, 1),       # (B,2C,H,W)->(B,hidden,H,W)
            nn.GroupNorm(h_groups, hidden_dim),           # ->(B,hidden,H,W)
            nn.SiLU(),                                    # ->(B,hidden,H,W)
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),  # ->(B,hidden,H,W)
            nn.GroupNorm(h_groups, hidden_dim),           # ->(B,hidden,H,W)
            nn.SiLU(),                                    # ->(B,hidden,H,W)
            nn.Conv2d(hidden_dim, channels, 1)            # ->(B,C,H,W) 输出 h (logit-like)
        )

        nn.init.constant_(self.metric_network[-1].weight, 0)  # 末层置零 初始 h=0 (zero init last layer)
        nn.init.constant_(self.metric_network[-1].bias, 0)    # 同上 (zero init)

    def forward(self, x, condition):                      # x: (B,C,H,W) condition: (B,cond_dim) or (B,cond_dim,H,W)
        B, C, H, W = x.shape                              # 取形状 (B,C,H,W)
        if condition.dim() == 2:                          # 若 condition 为 (B,cond_dim)
            condition = condition.view(B, -1, 1, 1).clone()# ->(B,cond_dim,1,1)
        else:
            condition = condition.clone()                 # (B,cond_dim,H,W)
        cond_spatial = self.cond_proj(condition)          # (B,cond_dim,*,*)->(B,C,*,*)
        if cond_spatial.shape[-2:] != (H, W):             # 若 spatial 不匹配 (spatial mismatch)
            cond_spatial = cond_spatial.expand(-1, -1, H, W)  # ->(B,C,H,W)
        combined = torch.cat([x, cond_spatial], dim=1)    # concat 通道: (B,C,H,W)+(B,C,H,W)->(B,2C,H,W)
        h = self.metric_network(combined)                 # (B,2C,H,W)->(B,C,H,W)

        G_diag = 1.0 + torch.tanh(h) * 0.9                # g in [0.1,1.9] before clamp 形状 (B,C,H,W)
        G_diag = torch.clamp(G_diag, min=0.1, max=10.0)   # 数值稳定 clamp 形状不变 (B,C,H,W)
        return G_diag                                     # 返回对角度量 (B,C,H,W)

# ==============================================================================
# 2. 黎曼回缩流 (PhysicsRiemannianFlowBlock) - [Fix: Re-normalization]
# ==============================================================================
class PhysicsRiemannianFlowBlock(nn.Module):             # 黎曼流形修正块 (Riemannian flow rectification)
    def __init__(self, channels, cond_dim=128):
        super().__init__()
        self.v_conv1 = nn.Conv2d(channels, channels, 3, 1, 1)  # 速度场卷积1: (B,C,H,W)->(B,C,H,W)
        self.adagn1 = AdaGN(channels, cond_dim)                 # 条件 GN1: (B,C,H,W) conditioned on cond
        self.v_conv2 = nn.Conv2d(channels, channels, 3, 1, 1)   # 速度场卷积2: (B,C,H,W)->(B,C,H,W)
        self.adagn2 = AdaGN(channels, cond_dim)                 # 条件 GN2
        self.v_projector = nn.Sequential(                        # 投影到切空间速度 并限幅 (project & bound)
            nn.Conv2d(channels, channels, 1),                   # (B,C,H,W)->(B,C,H,W)
            nn.Tanh()                                           # 限制速度幅值 (B,C,H,W)
        )
        self.metric_estimator = DiagonalSPDMetricEstimator(channels, cond_dim)  # 估计 G_diag: (B,C,H,W)

        self.internal_phys_proj = nn.Sequential(                 # 当 condition=None 时 从 x 自行生成 cond 向量 (self-conditioned)
            nn.AdaptiveAvgPool2d(1),                             # (B,C,H,W)->(B,C,1,1)
            nn.Flatten(),                                        # ->(B,C)
            nn.Linear(channels, cond_dim),                       # (B,C)->(B,cond_dim)
            nn.LayerNorm(cond_dim)                               # ->(B,cond_dim)
        )

        self.out_norm = nn.GroupNorm(                            # 输出再归一化 纠正分布漂移 (renorm)
            get_safe_groups(channels), channels
        )                                                        # (B,C,H,W)->(B,C,H,W)

    def riemannian_retraction_step(self, x, v, G_diag):           # 回缩步 x_{t+1}=x+G^{-1/2} v (retraction)
        preconditioner = torch.rsqrt(G_diag + 1e-6)               # 计算 1/sqrt(g) 形状 (B,C,H,W)
        return x + v * preconditioner                             # 广播逐点缩放 输出 (B,C,H,W)

    def forward(self, x, condition=None):                         # x: (B,C,H,W) condition: (B,cond_dim) or (B,cond_dim,H,W)
        if condition is None:                                     # 若未提供 condition (no condition)
            condition = self.internal_phys_proj(x)                # (B,C,H,W)->(B,cond_dim)

        v = self.v_conv1(x)                                       # (B,C,H,W)->(B,C,H,W)
        v = self.adagn1(v, condition)                             # ->(B,C,H,W) 施加条件调制
        v = F.silu(v)                                             # ->(B,C,H,W)
        v = self.v_conv2(v)                                       # ->(B,C,H,W)
        v = self.adagn2(v, condition)                             # ->(B,C,H,W)
        velocity = self.v_projector(v)                            # ->(B,C,H,W) 并通过 tanh 限幅

        G_diag = self.metric_estimator(x, condition)              # ->(B,C,H,W)

        x_rectified = self.riemannian_retraction_step(            # ->(B,C,H,W)
            x, velocity, G_diag
        )

        return self.out_norm(x_rectified)                         # ->(B,C,H,W) 输出分布拉回稳定区间

# ==============================================================================
# 3. 业务模块
# ==============================================================================
class PPRB(nn.Module):                                   # C3k2 + patch-wise 物理条件 + 黎曼流形修正 (patch block)
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, phys_ctx=None):
        super().__init__()
        self.main_branch = C3k2(c1, c2, n, c3k, e, g, shortcut)   # 主干块: (B,c1,H,W)->(B,c2,H,W)

        if phys_ctx is not None:                                  # 若外部共享 HPCE (shared context)
            self.sdpe = phys_ctx                                  # 复用同一实例 (reuse)
        else:
            self.sdpe = HPCE(embed_dim=128)             # 新建 HPCE: img->(B,128)

        self.local_extractor = nn.Sequential(                      # 从特征提取局部纹理条件 (local condition)
            nn.Conv2d(c2, c2, 5, 1, 2, groups=c2, bias=False),     # depthwise 5x5: (B,c2,H,W)->(B,c2,H,W)
            nn.Conv2d(c2, 64, 1, 1, 0),                            # 1x1: (B,c2,H,W)->(B,64,H,W)
            nn.BatchNorm2d(64),                                    # ->(B,64,H,W)
            nn.SiLU()                                              # ->(B,64,H,W)
        )
        self.riemannian_block = PhysicsRiemannianFlowBlock(        # cond_dim=128+64=192 (global+local)
            c2, cond_dim=128 + 64
        )
        self.gate_conv = nn.Sequential(                             # 门控融合 (gating)
            nn.Conv2d(c2 * 2, c2, 1),                               # concat 后: (B,2c2,H,W)->(B,c2,H,W)
            nn.Sigmoid()                                           # ->(B,c2,H,W) gate in (0,1)
        )
        self.gamma = nn.Parameter(torch.zeros(1, c2, 1, 1))         # 可学习尺度 (1,c2,1,1) broadcast 到 (B,c2,H,W)
        self.loss_proj = nn.Linear(c2, 128) if c2 != 128 else nn.Identity()  # (B,c2)->(B,128) or identity
        self.manifold_loss = torch.tensor(0.0)                      # 存储辅助损失 标量张量 (scalar tensor)

    def __deepcopy__(self, memo):                                   # deepcopy 修复 避免复制旧 manifold_loss 图 (deepcopy fix)
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k == 'manifold_loss':                                # 不复制旧张量图 (reset loss tensor)
                setattr(new_obj, k, torch.tensor(0.0))
            else:
                setattr(new_obj, k, deepcopy(v, memo))
        return new_obj

    def forward(self, x, img=None):                                  # x: 特征 (B,c1,H,W) 或 [feat,img]  img: (B,3,H0,W0)
        # [Zero-Intrusion Fix]：兼容 YOLO 将 [feature, image] 作为 list 传入 (intercept list input)
        if isinstance(x, list) and len(x) == 2:                      # 若 x 是 [feat,img] (list input)
            x_main, img = x[0], x[1]                                 # x_main: (B,c1,H,W) img: (B,3,H0,W0)
        else:
            x_main = x                                               # x_main: (B,c1,H,W)

        x_out = self.main_branch(x_main)                             # 主干输出: (B,c2,H,W)
        B, C, H, W = x_out.shape                                     # 记录形状 (B=c batch, C=c2)

        if img is None:                                              # 若没有原图条件 (no image condition)
            x_rectified = self.riemannian_block(x_out, condition=None)  # self-conditioned: (B,c2,H,W)->(B,c2,H,W)
            self.manifold_loss = torch.tensor(0.0, device=x_main.device)  # 损失置零 标量 (scalar on device)
        else:
            global_prior = self.sdpe(img)                             # HPCE: img->(B,128)
            global_map = global_prior.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)  # (B,128)->(B,128,1,1)->(B,128,H,W)
            local_map = self.local_extractor(x_out)                   # local: (B,c2,H,W)->(B,64,H,W)
            spatial_condition = torch.cat([global_map, local_map], dim=1)  # concat: (B,128,H,W)+(B,64,H,W)->(B,192,H,W)
            x_rectified = self.riemannian_block(x_out, condition=spatial_condition)  # cond flow: (B,c2,H,W)->(B,c2,H,W)

            if self.training:                                         # 仅训练时计算辅助损失 (aux loss only in train)
                z_pool = F.adaptive_avg_pool2d(x_rectified, (1, 1)).flatten(1)  # (B,c2,H,W)->(B,c2,1,1)->(B,c2)
                z_embed = self.loss_proj(z_pool)                      # (B,c2)->(B,128) 或 identity
                z_norm = F.normalize(z_embed, dim=1)                  # L2 归一化 (B,128)
                p_norm = F.normalize(global_prior.detach(), dim=1)    # 条件向量 detach 防止回传到 HPCE 图 (B,128)
                sim = (z_norm * p_norm).sum(dim=1)                    # 余弦相似度 (B,128)->(B,)
                self.manifold_loss = 1.0 - sim.mean()                 # 标量 loss (scalar)
            else:
                self.manifold_loss = torch.tensor(0.0, device=x_main.device)  # eval 下置零

        displacement = x_rectified - x_out                            # 位移/修正量 (B,c2,H,W)
        gate_input = torch.cat([x_out, x_rectified], dim=1)           # concat: (B,2c2,H,W)
        gate = self.gate_conv(gate_input)                             # gate: (B,c2,H,W)
        return x_out + self.gamma * gate * displacement               # 融合输出: (B,c2,H,W)

class PhysFlowNetNeckRefinement(nn.Module):                               # wrapper：通道压缩 + 流形修正 + 残差/投影 (wrapper block)
    def __init__(self, c1, c2=None, e=0.5, shortcut=True, phys_ctx=None):
        super().__init__()
        c2 = c2 if c2 else c1                                         # 若未给 c2 则 c2=c1 (default)
        self.c = int(c1 * e)                                          # 中间通道数 c=int(c1*e) (bottleneck channels)
        self.ablation_mode = 'full'                                   # 消融模式字符串 (mode flag)

        if phys_ctx is not None:
            self.sdpe = phys_ctx                                      # 共享 HPCE (shared)
        else:
            self.sdpe = HPCE(embed_dim=128)                 # 新建 HPCE (img->(B,128))

        self.flow_block = PhysicsRiemannianFlowBlock(self.c, cond_dim=128)  # 流形块: (B,c,H,W)->(B,c,H,W)
        self.cv1 = nn.Conv2d(c1, self.c, 1, 1)                        # 1x1 压缩: (B,c1,H,W)->(B,c,H,W)
        self.cv2 = nn.Conv2d(self.c, c2, 1, 1)                        # 1x1 还原/变换: (B,c,H,W)->(B,c2,H,W)
        self.add = shortcut and c1 == c2                              # 是否残差相加 (residual add flag)
        self.dummy_cond = nn.Parameter(torch.randn(1, 128))           # 无图时的 dummy condition: (1,128)
        self.loss_proj = nn.Linear(self.c, 128) if self.c != 128 else nn.Identity()  # (B,c)->(B,128) 或 identity
        self.manifold_loss = torch.tensor(0.0)                        # 存储辅助损失 标量 (scalar)

    def __deepcopy__(self, memo):                                     # deepcopy 修复 (deepcopy fix)
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k == 'manifold_loss':
                setattr(new_obj, k, torch.tensor(0.0))                # loss 重置 (reset)
            else:
                setattr(new_obj, k, deepcopy(v, memo))                # 其余深拷贝 (deepcopy)
        return new_obj

    def forward(self, x, img=None):                                   # x: (B,c1,H,W) 或 [feat,img]
        # [Zero-Intrusion Fix]：兼容 list 输入 (intercept list input)
        if isinstance(x, list) and len(x) == 2:
            x_main, img = x[0], x[1]                                  # x_main: (B,c1,H,W) img:(B,3,H0,W0)
        else:
            x_main = x                                                # x_main: (B,c1,H,W)

        identity = x_main                                             # 残差支路 (B,c1,H,W)
        y = self.cv1(x_main)                                          # 压缩: (B,c1,H,W)->(B,c,H,W)
        current_loss = torch.tensor(0.0, device=x_main.device)        # 当前 loss 标量 (scalar on device)

        if img is None:                                               # 若无图 (no image)
            condition = self.dummy_cond.expand(y.shape[0], -1)        # (1,128)->(B,128)
        else:
            condition = self.sdpe(img)                                # (B,128)

        if self.ablation_mode == 'full':                              # 完整模式 (full)
            y_refined = self.flow_block(y, condition)                 # 流形修正: (B,c,H,W)->(B,c,H,W)
            if self.training and img is not None:                     # 训练且有图时计算辅助 loss (aux loss)
                y_pool = F.adaptive_avg_pool2d(y_refined, (1, 1)).flatten(1)  # (B,c,H,W)->(B,c)
                y_embed = self.loss_proj(y_pool)                      # (B,c)->(B,128)
                z_norm = F.normalize(y_embed, dim=1)                  # (B,128)
                p_norm = F.normalize(condition.detach(), dim=1)       # (B,128) detach
                sim = (z_norm * p_norm).sum(dim=1)                    # (B,)
                current_loss = 1.0 - sim.mean()                       # 标量 loss
            y = y_refined                                             # 更新 y (B,c,H,W)
        elif self.ablation_mode == 'euclidean':                       # 注意：这里仍调用 flow_block 但可在内部固定度量/条件实现“欧氏”等价 (mode placeholder)
            y = self.flow_block(y, condition)                         # (B,c,H,W)->(B,c,H,W)
        elif self.ablation_mode == 'diffusion_only':                  # 仅 diffusion/无条件 (dummy condition)
            dummy_c = self.dummy_cond.expand(y.shape[0], -1)          # (B,128)
            y = self.flow_block(y, dummy_c)                           # (B,c,H,W)->(B,c,H,W)

        self.manifold_loss = current_loss                             # 写入模块成员 标量 (store scalar)

        H0, W0 = y.shape[-2:]                                         # 记录原 spatial (H0,W0)
        if H0 % 2 != 0 or W0 % 2 != 0:                                # 若奇数尺寸 (odd size)
            y = F.pad(y, (0, W0 % 2, 0, H0 % 2), mode='reflect')      # pad 到偶数: ->(B,c,H0+H0%2,W0+W0%2)
        out = self.cv2(y)                                             # 1x1: (B,c,Hp,Wp)->(B,c2,Hp,Wp)
        if out.shape[-2:] != (H0, W0):                                # 若 pad 过 需裁回 (crop back)
            out = out[..., :H0, :W0]                                  # ->(B,c2,H0,W0)
        if self.add:                                                  # 若可残差相加 (residual)
            return identity + out                                     # (B,c1,H,W)+(B,c2,H,W)->(B,c1,H,W) 这里要求 c1=c2
        return out                                                    # 否则直接输出 (B,c2,H,W)

class PMD(nn.Module):                                # 下采样模块：几何 conv + 物理权重增强并融合 (downsample fusion)
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.main_conv = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)       # stride=2 下采样: (B,c1,H,W)->(B,c2,H/2,W/2)
        self.bn_main = nn.BatchNorm2d(c2)                             # BN: (B,c2,H/2,W/2)->同形
        g_c1 = get_safe_groups(c1 // 4, 4)                            # 为 c1//4 选择 GN 分组数 (GN groups)
        self.phys_estimator = nn.Sequential(                          # 估计物理权重 (estimate phys weight)
            nn.Conv2d(c1, c1 // 4, 1),                                # (B,c1,H,W)->(B,c1/4,H,W)
            nn.GroupNorm(g_c1, c1 // 4),                              # ->(B,c1/4,H,W)
            nn.SiLU(),                                                # ->(B,c1/4,H,W)
            nn.Conv2d(c1 // 4, c1, 1),                                # ->(B,c1,H,W)
            nn.Sigmoid()                                              # ->(B,c1,H,W) 权重在 (0,1)
        )
        self.manifold_pool = nn.MaxPool2d(kernel_size=2, stride=2)    # 池化下采样: (B,c1,H,W)->(B,c1,H/2,W/2)
        self.geo_proj = nn.Sequential(                                # 将物理分支投影到 c2 (project to c2)
            nn.Conv2d(c1, c2, 1, bias=False),                         # (B,c1,H/2,W/2)->(B,c2,H/2,W/2)
            nn.BatchNorm2d(c2)                                        # ->(B,c2,H/2,W/2)
        )
        self.act = nn.SiLU()                                          # 激活 (activation)

    def forward(self, x):                                             # x: (B,c1,H,W)
        x_geom = self.bn_main(self.main_conv(x))                      # 几何分支: (B,c1,H,W)->(B,c2,H/2,W/2)->BN
        phys_weight = self.phys_estimator(x)                          # 物理权重: (B,c1,H,W)->(B,c1,H,W)
        x_enhanced = x * (1.0 + phys_weight)                          # 增强: (B,c1,H,W) 按通道逐点缩放
        x_phys = self.geo_proj(self.manifold_pool(x_enhanced))        # 物理分支: pool->(B,c1,H/2,W/2)->proj->(B,c2,H/2,W/2)
        return self.act(x_geom + x_phys)                              # 融合并激活: (B,c2,H/2,W/2)

class PhysFlowNetDetect(Detect):                                             # 自定义 Detect Head 扩展返回 raw features (custom detect head)
    """
    PhysFlowNet detection head.
    继承原生 Detect。
    训练时返回 (logits, raw_features) 供额外损失使用。
    推理时保持原生输出接口。
    """
    def __init__(self, nc=80, ch=()):                                 # nc: 类别数 ch: 输入特征通道列表 (channels per level)
        super().__init__(nc, ch)                                      # 初始化 Detect 内部层 (init base detect)

    def forward(self, x):                                             # x: list[Tensor] 每层特征 (B,ch[i],Hi,Wi)
        # 1) 拦截并保存 Raw Features (clone raw features)
        x_raw = [xi.clone() for xi in x]                              # 逐层 clone 保持 (B,ch[i],Hi,Wi) 注意 clone 会保留梯度图 (keeps graph)

        # 2) 兼容端到端模式 (YOLOv10-style end2end)
        if self.end2end:                                              # 若启用 end2end 分支 (end2end flag)
            return self.forward_end2end(x, x_raw)                # 进入 end2end 逻辑 返回 dict/tuple

        # 3) 原生预测逻辑：每层 concat(reg,cls)
        for i in range(self.nl):                                      # nl: 检测层数 (num levels)
            x[i] = torch.cat(                                         # 在通道维拼接 (concat on channel)
                (self.cv2[i](x[i]), self.cv3[i](x[i])), 1
            )                                                         # cv2/cv3 输出形状依赖 Detect 定义 通常 (B,no,Hi,Wi) 其中 no=reg+cls

        # 4) 训练阶段：返回 (logits_list, raw_features_list)
        if self.training:                                             # train branch
            return (x, x_raw)                                         # x: list logits, x_raw: list raw features

        # 5) 推理阶段：保持原生一致
        y = self._inference(x)                                        # 推理解码: -> (B, num_preds, no) 或内部格式 (depends on ultralytics)
        return y if self.export else (y, x)                           # export: 仅 y 否则 (y, x_logits)

    def forward_end2end(self, x, x_raw):                         # end2end 逻辑 透传 raw_features
        x_detach = [xi.detach() for xi in x]                          # one2one 用 detach 特征 防止梯度耦合 (detach for one2one)
        one2one = [                                                   # 构造 one2one logits 列表 (build one2one logits)
            torch.cat(
                (self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1
            )
            for i in range(self.nl)
        ]                                                             # 每个元素形状通常 (B,no,Hi,Wi)

        for i in range(self.nl):                                      # one2many 分支 logits (one2many logits)
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # ->(B,no,Hi,Wi)

        if self.training:                                             # 训练：返回 dict 并加入 raw_features (add raw features)
            return {
                "one2many": x,                                        # list logits
                "one2one": one2one,                                   # list logits
                "raw_features": x_raw                                 # list raw features (B,ch[i],Hi,Wi)
            }

        y = self._inference(one2one)                                  # 推理用 one2one (inference on one2one)
        y = self.postprocess(                                         # 后处理: NMS/筛选 (postprocess)
            y.permute(0, 2, 1), self.max_det, self.nc
        )                                                             # 维度通常从 (B,no,N)->(B,N,no) 再 postprocess -> 检测结果结构
        return y if self.export else (y, {"one2many": x, "one2one": one2one})  # 返回与原生一致的封装
