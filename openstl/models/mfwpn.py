import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import numbers
from typing import Iterable, Optional, Sequence, Tuple
from einops import rearrange
from openstl.modules import (ConvSC, GASubBlock)


def _validate_turbine_coords(coords: Iterable[Tuple[int, int]], height: int, width: int) -> Sequence[Tuple[int, int]]:
    validated = []
    for idx, coord in enumerate(coords):
        if len(coord) != 2:
            raise ValueError(f"Turbine coordinate at index {idx} must contain two values (h, w), got {coord}.")
        h, w = int(coord[0]), int(coord[1])
        if not (0 <= h < height and 0 <= w < width):
            raise ValueError(
                f"Turbine coordinate {(h, w)} is outside the feature map with shape ({height}, {width}).")
        validated.append((h, w))
    return validated


class WindCorrectionHead(nn.Module):
    """增强版风速校正头：残差学习 + 丰富输入特征
    
    改进点:
    1. 残差学习: MLP只学习修正量Δv，最终输出 = 插值风速 + Δv
    2. 增强输入: 添加u/v分量、风速梯度、邻域信息
    3. 更深网络: 增加层数 + LayerNorm + Dropout
    """

    def __init__(self, feature_dim: int, turbine_coords: Sequence[Tuple[int, int]], 
                 roi_size: int = 5, hidden_dim: int = 128, feature_hw: Tuple[int, int] = (64, 80),
                 use_residual: bool = True, dropout: float = 0.1):
        super().__init__()
        if roi_size % 2 == 0 or roi_size <= 0:
            raise ValueError(f"roi_size must be a positive odd number, got {roi_size}.")

        self.use_residual = use_residual
        self.roi_size = roi_size
        self.pad = roi_size // 2
        self.feature_hw = feature_hw
        self.turbine_coords = _validate_turbine_coords(turbine_coords, *feature_hw)
        self.num_turbines = len(self.turbine_coords)
        self.hidden_dim = hidden_dim

        # 初始化采样网格
        self._init_sample_grids()

        # 局部特征提取
        local_feat_dim = hidden_dim // 2
        self.local_conv = nn.Sequential(
            nn.Conv2d(feature_dim, local_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(local_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 增强输入维度:
        # - 插值风速: 1
        # - u, v 分量: 2
        # - 风速梯度 (上下左右): 4
        # - 3x3邻域 (去中心): 8
        # - 局部特征: local_feat_dim
        enhanced_input_dim = 1 + 2 + 4 + 8 + local_feat_dim

        # 深层 MLP with 正则化
        self.correction_mlp = nn.Sequential(
            nn.Linear(enhanced_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1)
        )

        # 残差缩放因子 (可学习)
        if use_residual:
            self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _init_sample_grids(self):
        """初始化采样网格: 中心点 + 3x3邻域"""
        feature_h, feature_w = self.feature_hw
        
        # 中心点采样网格
        norm_coords = []
        for h_idx, w_idx in self.turbine_coords:
            y = 2.0 * h_idx / (feature_h - 1) - 1.0
            x = 2.0 * w_idx / (feature_w - 1) - 1.0
            norm_coords.append([x, y])
        grid = torch.tensor(norm_coords, dtype=torch.float32).view(1, self.num_turbines, 1, 2)
        self.register_buffer('sample_grid', grid, persistent=False)
        
        # 3x3 邻域采样网格 (去掉中心点)
        neighbor_coords = []
        for h_idx, w_idx in self.turbine_coords:
            for dh in [-1, 0, 1]:
                for dw in [-1, 0, 1]:
                    if dh == 0 and dw == 0:
                        continue  # 跳过中心点
                    nh = max(0, min(h_idx + dh, feature_h - 1))
                    nw = max(0, min(w_idx + dw, feature_w - 1))
                    y = 2.0 * nh / (feature_h - 1) - 1.0
                    x = 2.0 * nw / (feature_w - 1) - 1.0
                    neighbor_coords.append([x, y])
        neighbor_grid = torch.tensor(neighbor_coords, dtype=torch.float32)
        neighbor_grid = neighbor_grid.view(1, self.num_turbines * 8, 1, 2)
        self.register_buffer('neighbor_grid', neighbor_grid, persistent=False)

    def _compute_gradients(self, speed_field: torch.Tensor, h_idx: int, w_idx: int) -> torch.Tensor:
        """计算指定位置的风速梯度"""
        b, t, _, h, w = speed_field.shape
        
        def safe_get(hi, wi):
            hi = max(0, min(hi, h-1))
            wi = max(0, min(wi, w-1))
            return speed_field[:, :, 0, hi, wi]
        
        center = safe_get(h_idx, w_idx)
        grad_up = center - safe_get(h_idx - 1, w_idx)
        grad_down = safe_get(h_idx + 1, w_idx) - center
        grad_left = center - safe_get(h_idx, w_idx - 1)
        grad_right = safe_get(h_idx, w_idx + 1) - center
        
        return torch.stack([grad_up, grad_down, grad_left, grad_right], dim=-1)

    def forward(self, features: torch.Tensor, wind_field: torch.Tensor) -> dict:
        """计算校正后的精确点风速。

        Args:
            features: 时空特征 [B, T, C, H, W]
            wind_field: 预测风场 (u, v) [B, T, 2, H, W]

        Returns:
            dict: {
                'interp_speed': 插值风速 [B, T, N_turbines],
                'corrected_speed': 校正后风速 [B, T, N_turbines]
            }
        """
        b, t, c, h, w = features.shape
        device = features.device

        # 1. 计算风速场
        speed_field = torch.linalg.norm(wind_field, dim=2, keepdim=True)  # [B,T,1,H,W]
        
        # 2. 插值 u, v, speed
        wind_flat = wind_field.reshape(b * t, 2, h, w)
        speed_flat = speed_field.reshape(b * t, 1, h, w)
        grid = self.sample_grid.to(device).expand(b * t, -1, -1, -1)
        
        u_interp = F.grid_sample(wind_flat[:, 0:1], grid, align_corners=True, mode='bilinear')
        v_interp = F.grid_sample(wind_flat[:, 1:2], grid, align_corners=True, mode='bilinear')
        speed_interp = F.grid_sample(speed_flat, grid, align_corners=True, mode='bilinear')
        
        u_interp = u_interp.view(b, t, self.num_turbines)
        v_interp = v_interp.view(b, t, self.num_turbines)
        speed_interp = speed_interp.view(b, t, self.num_turbines)
        
        # 3. 采样邻域风速
        neighbor_grid = self.neighbor_grid.to(device).expand(b * t, -1, -1, -1)
        neighbors = F.grid_sample(speed_flat, neighbor_grid, align_corners=True, mode='bilinear')
        neighbors = neighbors.view(b, t, self.num_turbines, 8)
        
        # 4. 提取局部特征
        x = features.reshape(b * t, c, h, w)
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='replicate')
        
        corrected_outputs = []
        for idx, (h_idx, w_idx) in enumerate(self.turbine_coords):
            # 局部特征
            center_h = h_idx + self.pad
            center_w = w_idx + self.pad
            patch = x[:, :, center_h - self.pad:center_h + self.pad + 1,
                      center_w - self.pad:center_w + self.pad + 1]
            local_feat = self.local_conv(patch)
            pooled = self.pool(local_feat).reshape(b, t, -1)
            
            # 计算梯度
            gradients = self._compute_gradients(speed_field, h_idx, w_idx)
            
            # 组合特征
            point_features = torch.cat([
                speed_interp[:, :, idx:idx+1],     # 插值风速 (1)
                u_interp[:, :, idx:idx+1],         # u 分量 (1)
                v_interp[:, :, idx:idx+1],         # v 分量 (1)
                gradients,                          # 梯度 (4)
                neighbors[:, :, idx, :],           # 邻域 (8)
                pooled                              # 局部特征 (hidden_dim//2)
            ], dim=-1)
            
            # MLP 校正
            delta = self.correction_mlp(point_features.reshape(b * t, -1))
            delta = delta.reshape(b, t, 1)
            
            # 残差输出
            if self.use_residual:
                corrected = speed_interp[:, :, idx:idx+1] + self.residual_scale * delta
            else:
                corrected = delta
            
            corrected_outputs.append(corrected)
        
        corrected_speed = torch.cat(corrected_outputs, dim=2)

        return {
            'interp_speed': speed_interp,
            'corrected_speed': corrected_speed
        }


class TemporalWindCorrectionHead(nn.Module):
    """时序风速校正头：使用双向LSTM建模时间依赖
    
    改进点:
    1. 双向LSTM: 同时利用过去和未来的时序信息
    2. 时间嵌入: 编码小时信息，学习日周期模式
    3. 残差学习: 保持残差输出结构
    4. 增强输入: 保留u/v分量、梯度、邻域特征
    """

    def __init__(self, feature_dim: int, turbine_coords: Sequence[Tuple[int, int]], 
                 roi_size: int = 5, hidden_dim: int = 128, 
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = True,
                 use_time_embedding: bool = True,
                 feature_hw: Tuple[int, int] = (64, 80)):
        super().__init__()
        
        self.roi_size = roi_size
        self.pad = roi_size // 2
        self.feature_hw = feature_hw
        self.turbine_coords = _validate_turbine_coords(turbine_coords, *feature_hw)
        self.num_turbines = len(self.turbine_coords)
        self.use_time_embedding = use_time_embedding
        self.bidirectional = bidirectional

        # 初始化采样网格
        self._init_sample_grids()

        # 局部特征提取
        local_feat_dim = hidden_dim // 2
        self.local_conv = nn.Sequential(
            nn.Conv2d(feature_dim, local_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(local_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 增强输入维度: speed(1) + u,v(2) + grad(4) + neighbors(8) + local_feat
        enhanced_input_dim = 1 + 2 + 4 + 8 + local_feat_dim

        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(enhanced_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 时间嵌入 (24小时)
        if use_time_embedding:
            self.hour_embedding = nn.Embedding(24, hidden_dim // 4)
            lstm_input_dim = hidden_dim + hidden_dim // 4
        else:
            lstm_input_dim = hidden_dim

        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 输出层
        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)
        self.output_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 残差缩放因子 (可学习)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _init_sample_grids(self):
        """初始化采样网格: 中心点 + 3x3邻域"""
        feature_h, feature_w = self.feature_hw
        
        # 中心点采样网格
        norm_coords = []
        for h_idx, w_idx in self.turbine_coords:
            y = 2.0 * h_idx / (feature_h - 1) - 1.0
            x = 2.0 * w_idx / (feature_w - 1) - 1.0
            norm_coords.append([x, y])
        grid = torch.tensor(norm_coords, dtype=torch.float32).view(1, self.num_turbines, 1, 2)
        self.register_buffer('sample_grid', grid, persistent=False)
        
        # 3x3 邻域采样网格 (去掉中心点)
        neighbor_coords = []
        for h_idx, w_idx in self.turbine_coords:
            for dh in [-1, 0, 1]:
                for dw in [-1, 0, 1]:
                    if dh == 0 and dw == 0:
                        continue
                    nh = max(0, min(h_idx + dh, feature_h - 1))
                    nw = max(0, min(w_idx + dw, feature_w - 1))
                    y = 2.0 * nh / (feature_h - 1) - 1.0
                    x = 2.0 * nw / (feature_w - 1) - 1.0
                    neighbor_coords.append([x, y])
        neighbor_grid = torch.tensor(neighbor_coords, dtype=torch.float32)
        neighbor_grid = neighbor_grid.view(1, self.num_turbines * 8, 1, 2)
        self.register_buffer('neighbor_grid', neighbor_grid, persistent=False)

    def _compute_gradients(self, speed_field: torch.Tensor, h_idx: int, w_idx: int) -> torch.Tensor:
        """计算指定位置的风速梯度"""
        b, t, _, h, w = speed_field.shape
        
        def safe_get(hi, wi):
            hi = max(0, min(hi, h-1))
            wi = max(0, min(wi, w-1))
            return speed_field[:, :, 0, hi, wi]
        
        center = safe_get(h_idx, w_idx)
        grad_up = center - safe_get(h_idx - 1, w_idx)
        grad_down = safe_get(h_idx + 1, w_idx) - center
        grad_left = center - safe_get(h_idx, w_idx - 1)
        grad_right = safe_get(h_idx, w_idx + 1) - center
        
        return torch.stack([grad_up, grad_down, grad_left, grad_right], dim=-1)

    def forward(self, features: torch.Tensor, wind_field: torch.Tensor) -> dict:
        """计算校正后的精确点风速（时序版本）

        Args:
            features: 时空特征 [B, T, C, H, W]
            wind_field: 预测风场 (u, v) [B, T, 2, H, W]

        Returns:
            dict: {
                'interp_speed': 插值风速 [B, T, N_turbines],
                'corrected_speed': 校正后风速 [B, T, N_turbines]
            }
        """
        b, t, c, h, w = features.shape
        device = features.device

        # 1. 计算风速场
        speed_field = torch.linalg.norm(wind_field, dim=2, keepdim=True)  # [B,T,1,H,W]
        
        # 2. 插值 u, v, speed
        wind_flat = wind_field.reshape(b * t, 2, h, w)
        speed_flat = speed_field.reshape(b * t, 1, h, w)
        grid = self.sample_grid.to(device).expand(b * t, -1, -1, -1)
        
        u_interp = F.grid_sample(wind_flat[:, 0:1], grid, align_corners=True, mode='bilinear')
        v_interp = F.grid_sample(wind_flat[:, 1:2], grid, align_corners=True, mode='bilinear')
        speed_interp = F.grid_sample(speed_flat, grid, align_corners=True, mode='bilinear')
        
        u_interp = u_interp.view(b, t, self.num_turbines)
        v_interp = v_interp.view(b, t, self.num_turbines)
        speed_interp = speed_interp.view(b, t, self.num_turbines)
        
        # 3. 采样邻域风速
        neighbor_grid = self.neighbor_grid.to(device).expand(b * t, -1, -1, -1)
        neighbors = F.grid_sample(speed_flat, neighbor_grid, align_corners=True, mode='bilinear')
        neighbors = neighbors.view(b, t, self.num_turbines, 8)
        
        # 4. 提取局部特征
        x = features.reshape(b * t, c, h, w)
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='replicate')
        
        corrected_outputs = []
        for idx, (h_idx, w_idx) in enumerate(self.turbine_coords):
            # 局部特征
            center_h = h_idx + self.pad
            center_w = w_idx + self.pad
            patch = x[:, :, center_h - self.pad:center_h + self.pad + 1,
                      center_w - self.pad:center_w + self.pad + 1]
            local_feat = self.local_conv(patch)
            pooled = self.pool(local_feat).reshape(b, t, -1)
            
            # 计算梯度
            gradients = self._compute_gradients(speed_field, h_idx, w_idx)
            
            # 组合特征 [B, T, D]
            point_features = torch.cat([
                speed_interp[:, :, idx:idx+1],     # 插值风速 (1)
                u_interp[:, :, idx:idx+1],         # u 分量 (1)
                v_interp[:, :, idx:idx+1],         # v 分量 (1)
                gradients,                          # 梯度 (4)
                neighbors[:, :, idx, :],           # 邻域 (8)
                pooled                              # 局部特征 (hidden_dim//2)
            ], dim=-1)
            
            # 5. 输入投影
            proj_feat = self.input_proj(point_features)  # [B, T, hidden_dim]
            
            # 6. 添加时间嵌入
            if self.use_time_embedding:
                hours = torch.arange(t, device=device)
                hour_emb = self.hour_embedding(hours)  # [T, hidden_dim//4]
                hour_emb = hour_emb.unsqueeze(0).expand(b, -1, -1)  # [B, T, hidden_dim//4]
                lstm_input = torch.cat([proj_feat, hour_emb], dim=-1)
            else:
                lstm_input = proj_feat
            
            # 7. LSTM 时序建模
            lstm_out, _ = self.lstm(lstm_input)  # [B, T, lstm_hidden*2]
            
            # 8. 输出修正量
            delta = self.output_head(lstm_out)  # [B, T, 1]
            
            # 9. 残差输出
            corrected = speed_interp[:, :, idx:idx+1] + self.residual_scale * delta
            corrected_outputs.append(corrected)
        
        corrected_speed = torch.cat(corrected_outputs, dim=2)  # [B, T, N]

        return {
            'interp_speed': speed_interp,
            'corrected_speed': corrected_speed
        }


class TurbinePowerHead(nn.Module):
    """Predict turbine power from shared spatiotemporal features."""

    def __init__(self, in_channels: int, turbine_coords: Sequence[Tuple[int, int]], roi_size: int = 5,
                 conv_channels: int = 32, mlp_hidden_dim: int = 64, feature_hw: Tuple[int, int] = (64, 80),
                 include_wind_speed: bool = True, use_corrected_speed: bool = False):
        super().__init__()
        if roi_size % 2 == 0 or roi_size <= 0:
            raise ValueError(f"roi_size must be a positive odd number, got {roi_size}.")

        self.roi_size = roi_size
        self.pad = roi_size // 2
        self.feature_hw = feature_hw
        self.turbine_coords = _validate_turbine_coords(turbine_coords, *feature_hw)
        self.num_turbines = len(self.turbine_coords)
        self.include_wind_speed = include_wind_speed
        self.use_corrected_speed = use_corrected_speed

        conv_hidden = max(conv_channels, in_channels)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_hidden, conv_hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        linear_in_dim = conv_hidden + (1 if self.include_wind_speed else 0)
        self.regressor = nn.Sequential(
            nn.Linear(linear_in_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 1)
        )

        if self.include_wind_speed:
            feature_h, feature_w = feature_hw
            if feature_h <= 1 or feature_w <= 1:
                raise ValueError("feature_hw must both be greater than 1 when using wind speed sampling.")

            norm_coords = []
            for h_idx, w_idx in self.turbine_coords:
                y = 2.0 * h_idx / (feature_h - 1) - 1.0
                x = 2.0 * w_idx / (feature_w - 1) - 1.0
                norm_coords.append([x, y])
            grid = torch.tensor(norm_coords, dtype=torch.float32).view(1, self.num_turbines, 1, 2)
            self.register_buffer('sample_grid', grid, persistent=False)
        else:
            self.register_buffer('sample_grid', None, persistent=False)

    def forward(self, features: torch.Tensor, wind_field: Optional[torch.Tensor] = None,
                corrected_speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute power predictions.

        Args:
            features: Tensor of shape [B, T, C, H, W].
            wind_field: Optional tensor of predicted wind vectors shaped [B, T, 2, H, W].
            corrected_speed: Optional tensor of corrected wind speed [B, T, N_turbines].
                             If use_corrected_speed=True, this will be used instead of interpolated speed.

        Returns:
            Power predictions shaped [B, T, N_turbines, 1].
        """
        b, t, c, h, w = features.shape
        if (h, w) != self.feature_hw:
            raise ValueError(f"Unexpected feature map size {(h, w)}; expected {self.feature_hw} for turbine decoding.")

        sampled_speed = None
        if self.include_wind_speed:
            if wind_field is None:
                raise ValueError("wind_field must be provided when include_wind_speed=True.")
            if wind_field.shape != (b, t, 2, h, w):
                raise ValueError(
                    f"wind_field must have shape {(b, t, 2, h, w)}, got {wind_field.shape}.")

            wind_speed = torch.linalg.norm(wind_field, dim=2, keepdim=True)
            wind_speed = wind_speed.reshape(b * t, 1, h, w)
            grid = self.sample_grid.to(wind_speed.device)
            sampled = F.grid_sample(wind_speed, grid.expand(b * t, -1, -1, -1),
                                    align_corners=True, mode='bilinear')
            sampled_speed = sampled.view(b, t, self.num_turbines)

        # 如果启用校正风速且提供了校正值，则使用校正后的风速
        if self.use_corrected_speed and corrected_speed is not None:
            sampled_speed = corrected_speed

        x = features.reshape(b * t, c, h, w)
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='replicate')

        power_outputs = []
        for idx, (h_idx, w_idx) in enumerate(self.turbine_coords):
            center_h = h_idx + self.pad
            center_w = w_idx + self.pad
            patch = x[:, :, center_h - self.pad:center_h + self.pad + 1,
                      center_w - self.pad:center_w + self.pad + 1]
            local_feat = self.local_conv(patch)
            pooled = self.pool(local_feat).reshape(b, t, -1)
            if sampled_speed is not None:
                reg_input = torch.cat([pooled, sampled_speed[:, :, idx:idx + 1]], dim=-1)
            else:
                reg_input = pooled
            regressed = self.regressor(reg_input.reshape(b * t, -1)).reshape(b, t, 1)
            power_outputs.append(regressed)

        return torch.stack(power_outputs, dim=2)


class MFWPN_Model(nn.Module):
    def __init__(self, hid_S=2, hid_T=256, N_S=2, N_T=8, model_type='gsta',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, turbine_coords: Optional[Sequence[Tuple[int, int]]] = None,
                 power_roi_size: int = 5, power_conv_channels: int = 32, power_mlp_hidden: int = 64,
                 feature_hw: Tuple[int, int] = (64, 80), 
                 enable_wind_correction: bool = False, wind_correction_hidden: int = 128,
                 wind_correction_dropout: float = 0.1, use_residual: bool = True,
                 use_temporal_correction: bool = False,  # 使用时序LSTM校正头
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 use_time_embedding: bool = True,
                 use_corrected_speed_for_power: bool = False, **kwargs):
        super(MFWPN_Model, self).__init__()
        T, C, H, W = 24, 2, 64, 80  # T is pre_seq_length
        act_inplace = False
        self.enc = Encoder(spatio_kernel=spatio_kernel_enc, act_inplace=True)
        self.dec = Decoder(spatio_kernel=spatio_kernel_dec, act_inplace=True)
        self.hid_w = MidMetaNet(T*hid_S, hid_T, N_T, input_resolution=(64, 80), model_type=model_type,
                                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        self.hid_tz = MidMetaNet(T*hid_S, hid_T, N_T, input_resolution=(64, 80), model_type=model_type,
                                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        self.channel_f = CAM(channel = 48)
        self.channel_i = CAM(channel = 48)
        self.gate = nn.Tanh()
        self.ele_conv = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(48, 48, kernel_size=1, stride=1))
        self.wind_correction_head: Optional[WindCorrectionHead] = None
        self.power_head: Optional[TurbinePowerHead] = None
        self._turbine_coords = turbine_coords
        self._feature_hw = feature_hw
        
        if turbine_coords:
            # 精确点风速校正头
            if enable_wind_correction:
                if use_temporal_correction:
                    # 时序 LSTM 版本
                    self.wind_correction_head = TemporalWindCorrectionHead(
                        feature_dim=hid_S,
                        turbine_coords=turbine_coords,
                        roi_size=power_roi_size,
                        hidden_dim=wind_correction_hidden,
                        lstm_hidden=lstm_hidden,
                        lstm_layers=lstm_layers,
                        dropout=wind_correction_dropout,
                        bidirectional=True,
                        use_time_embedding=use_time_embedding,
                        feature_hw=feature_hw
                    )
                else:
                    # MLP 版本
                    self.wind_correction_head = WindCorrectionHead(
                        feature_dim=hid_S,
                        turbine_coords=turbine_coords,
                        roi_size=power_roi_size,
                        hidden_dim=wind_correction_hidden,
                        feature_hw=feature_hw,
                        use_residual=use_residual,
                        dropout=wind_correction_dropout
                    )
            # 功率预测头
            self.power_head = TurbinePowerHead(
                in_channels=hid_S,
                turbine_coords=turbine_coords,
                roi_size=power_roi_size,
                conv_channels=power_conv_channels,
                mlp_hidden_dim=power_mlp_hidden,
                feature_hw=feature_hw,
                use_corrected_speed=use_corrected_speed_for_power
            )

    def forward(self, x_raw, ele, **kwargs):
        B, T, C, H, W = x_raw.shape

        ele = ele[None, None, :, :]
        ele_h = ele
        ele = ele.repeat(B*T,1,1,1)
        x = x_raw.view(B*T, C, H, W)  
        wcs, tzcs = self.enc(x, ele)
        
        _, C_w, H_, W_ = wcs.shape
        _, C_tz, H_, W_ = tzcs.shape

        w = wcs.view(B, T, C_w, H_, W_)
        tz = tzcs.view(B, T, C_tz, H_, W_)

        hid_w = self.hid_w(w)  
        hid_tz = self.hid_tz(tz)
        #ele_h = ele_h.repeat(B, 48, 1, 1)
        ####time fusion
        hid_w = hid_w * self.channel_f(hid_tz) + self.channel_i(hid_tz) * self.gate(hid_tz)  
        hid_w = hid_w.reshape(B , T, C_w, H_, W_)

        dec_input = hid_w.reshape(B * T, C_w, H_, W_)
        Y = self.dec(dec_input)
        Y = Y.reshape(B, T, 2, H, W)

        outputs = {'wind': Y}
        
        corrected_speed = None
        if self.wind_correction_head is not None:
            wind_corr_out = self.wind_correction_head(hid_w, wind_field=Y)
            outputs['interp_speed'] = wind_corr_out['interp_speed']
            outputs['corrected_speed'] = wind_corr_out['corrected_speed']
            corrected_speed = wind_corr_out['corrected_speed']
        
        if self.power_head is not None:
            power_pred = self.power_head(hid_w, wind_field=Y, corrected_speed=corrected_speed)
            outputs['power'] = power_pred
        
        return outputs
    
class SAM(nn.Module):
    def __init__(self, spatial_kernel=7):
        super(SAM, self).__init__()
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)      
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        return spatial_out

class CAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAM, self).__init__()
   
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(       
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),   
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )        
        self.sigmoid = nn.Sigmoid()
                                   
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        eca = max_out + avg_out
        channel_out = self.sigmoid(eca)                           
        return channel_out
                                  
def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    """3D Encoder for SimVP"""
    def __init__(self, spatio_kernel, act_inplace=True):
        super(Encoder, self).__init__()  
        
        self.wscale = nn.Parameter(torch.ones(2))
        self.tzscale = nn.Parameter(torch.ones(2))
         ####Conv branch
        self.wce =  INN_all(num_layers=1)
        self.tzce = INN_all(num_layers=1)
        ####Self-attention branch
        self.wse = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False,
                                     LayerNorm_type='WithBias')
        
        self.tzse = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False,
                                     LayerNorm_type='WithBias')   
        self.sa_f = SAM()
        self.sa_i = SAM()
        self.gate_1 = nn.Sigmoid()
        self.gate_2 = nn.Tanh()
        ####ele
        self.ele_conv = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(2, 2, kernel_size=1, stride=1))
        
    def forward(self, x, ele):
        w = x[:, 0:2]
        tz = x[:, 2:4]
        
        wce_x = self.wce(w)
        wse_x = self.wse(w)
        wcs = wse_x * self.wscale[0] + wce_x * self.wscale[1]

        tzce_x = self.tzce(tz)
        tzse_x = self.tzse(tz)
        tzcs = tzse_x * self.tzscale[0] + tzce_x * self.tzscale[1]
        ####Fusion module
        tz_gate_2 = self.gate_2(tzcs)
        wcs = wcs * self.sa_f(tzcs) + self.sa_i(tzcs) * self.gate_2(tzcs) + wcs * self.gate_1(self.ele_conv(ele))

        return wcs, tzcs

class Decoder(nn.Module):
    """3D Encoder for SimVP"""
    def __init__(self, spatio_kernel, act_inplace=True):
        super(Decoder, self).__init__()
        
        self.dwscale = nn.Parameter(torch.ones(2))
        ####Conv branch
        self.dwce =  INN_all(num_layers=1)                   
        ####Self-attention branch
        self.dwse = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False,
                                     LayerNorm_type='WithBias')
    def forward(self, wdcs):
        
        dwse_x = self.dwse(wdcs)
        dwce_x = self.dwce(wdcs)
        dwcs = dwse_x * self.dwscale[0] + dwce_x * self.dwscale[1]
        
        return dwcs

class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'
        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"
            
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)
    
class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]
        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1)) 
        self.enc_1 = nn.Sequential(*enc_layers)
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)
        z = x
        for i in range(self.N2):
            z = self.enc_1[i](z)
        return z



class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out 


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
    
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
        
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
        
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class INN(nn.Module):
    def __init__(self, input, output, ratio):
        super(INN, self).__init__()
        hidden_dim = int(input * ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(input, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, output, 1, bias=False),
            nn.BatchNorm2d(output),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.phi = INN(input=32, output=32, ratio=2)
        self.seta = INN(input=32, output=32, ratio=2)
        
    def forward(self, f1, f2):
        f2 = f2 + self.phi(f1)
        f1 = f1 + self.seta(f2)
        return f1, f2

class INN_all(nn.Module):
    def __init__(self, num_layers=None):
        super(INN_all, self).__init__()
        INN_layers = [Feature() for _ in range(num_layers)]
        self.net = nn.Sequential(*INN_layers)
        
        self.shffle = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.fusion = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        
    def separate(self, x):
        f1, f2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return f1, f2
    
    def forward(self, x):
        f1, f2 = self.separate(self.shffle(x))
        
        for layer in self.net: 
            f1, f2 = layer(f1, f2)
        f_out = self.fusion(torch.cat((f1, f2), dim=1))
        return f_out

