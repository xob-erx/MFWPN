#!/usr/bin/env python3
"""
阶段2训练脚本：训练精确点风速校正头和功率预测头
- 加载预训练主干并冻结
- 时间对齐网格数据与精确点数据
- 归一化处理
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math

from openstl.models.mfwpn import MFWPN_Model
from config import configs


# ==================== 归一化工具 ====================

class Normalizer:
    """数据归一化器"""
    
    def __init__(self):
        self.stats = {}
    
    def fit_zscore(self, data: np.ndarray, name: str):
        """Z-score 归一化参数"""
        self.stats[name] = {
            'type': 'zscore',
            'mean': data.mean(),
            'std': data.std()
        }
        return self
    
    def fit_minmax(self, data: np.ndarray, name: str, feature_range=(0, 1)):
        """Min-Max 归一化参数"""
        self.stats[name] = {
            'type': 'minmax',
            'min': data.min(),
            'max': data.max(),
            'range': feature_range
        }
        return self
    
    def transform(self, data: np.ndarray, name: str) -> np.ndarray:
        """应用归一化"""
        s = self.stats[name]
        if s['type'] == 'zscore':
            return (data - s['mean']) / (s['std'] + 1e-8)
        else:  # minmax
            a, b = s['range']
            return a + (data - s['min']) / (s['max'] - s['min'] + 1e-8) * (b - a)
    
    def inverse_transform(self, data: np.ndarray, name: str) -> np.ndarray:
        """反归一化"""
        s = self.stats[name]
        if s['type'] == 'zscore':
            return data * s['std'] + s['mean']
        else:  # minmax
            a, b = s['range']
            return (data - a) / (b - a) * (s['max'] - s['min']) + s['min']
    
    def save(self, path: str):
        np.save(path, self.stats)
        print(f"Normalizer saved to {path}")
    
    def load(self, path: str):
        self.stats = np.load(path, allow_pickle=True).item()
        print(f"Normalizer loaded from {path}")
        return self


# ==================== 时间对齐 ====================

def align_grid_with_turbine(
    grid_data: np.ndarray,
    grid_start_date: str,
    turbine_dates: np.ndarray,
    hours_per_day: int = 24,
    turbine_timezone_offset: int = 8,  # 精确点数据时区偏移（北京时间=UTC+8）
    use_test_data: bool = False,
    test_grid_data: np.ndarray = None,
    test_start_date: str = None
) -> tuple:
    """
    将网格数据与精确点数据按时间对齐
    
    Args:
        grid_data: 训练网格数据 [N_hours, C, H, W] (UTC时间)
        grid_start_date: 网格数据起始日期 (e.g., "2020-01-01", UTC)
        turbine_dates: 精确点日期数组 [N_days] (北京时间)
        hours_per_day: 每天小时数
        turbine_timezone_offset: 精确点数据的时区偏移（北京时间=8）
        use_test_data: 是否使用测试集网格数据
        test_grid_data: 测试网格数据
        test_start_date: 测试网格数据起始日期
    
    Returns:
        aligned_grid: 对齐后的网格数据 [N_turbine_days * 24, C, H, W]
        valid_days: 有效日期索引
    """
    # 合并训练和测试网格数据（如果有）
    if use_test_data and test_grid_data is not None:
        full_grid_data = np.concatenate([grid_data, test_grid_data], axis=0)
        print(f"  Combined grid data: {full_grid_data.shape}")
    else:
        full_grid_data = grid_data
    
    grid_start = datetime.strptime(grid_start_date, "%Y-%m-%d")
    
    aligned_indices = []
    valid_days = []
    
    for i, date_str in enumerate(turbine_dates):
        # 精确点日期是北京时间，需要转换为UTC
        # 北京时间的一天 00:00-23:59 对应 UTC 的前一天 16:00 到当天 15:59
        # 但为了简化，我们假设精确点数据是日均值，对应UTC当天的数据
        turbine_date = datetime.strptime(str(date_str), "%Y-%m-%d")
        
        # 考虑时区：北京时间比UTC早8小时
        # 北京时间的第N天 00:00 = UTC第N-1天 16:00
        # 为简化，我们用北京时间日期对应的UTC同一日期（忽略8小时偏移）
        # 如果需要精确对齐，可以调整 hour_start
        days_offset = (turbine_date - grid_start).days
        
        # UTC小时索引
        hour_start = days_offset * hours_per_day
        hour_end = hour_start + hours_per_day
        
        if hour_start >= 0 and hour_end <= len(full_grid_data):
            aligned_indices.extend(range(hour_start, hour_end))
            valid_days.append(i)
        else:
            print(f"Warning: Date {date_str} out of grid range, skipping")
    
    if len(aligned_indices) == 0:
        raise ValueError("No overlapping dates between grid and turbine data!")
    
    aligned_grid = full_grid_data[aligned_indices]
    valid_days = np.array(valid_days)
    
    print(f"Aligned {len(valid_days)} days ({len(aligned_indices)} hours)")
    return aligned_grid, valid_days


# ==================== 数据集 ====================

class TurbineDataset(Dataset):
    """精确点训练数据集"""
    
    def __init__(self, grid_input, grid_target, wind_speed, power=None, indices=None):
        """
        Args:
            grid_input: 输入网格数据 [N, T_in, C, H, W]
            grid_target: 目标网格数据 [N, T_out, C, H, W]
            wind_speed: 精确点风速 [N, T_out]
            power: 精确点功率 [N, T_out]
            indices: 样本索引（用于追踪）
        """
        self.grid_input = grid_input
        self.grid_target = grid_target
        self.wind_speed = wind_speed
        self.power = power
        self.indices = indices
    
    def __len__(self):
        return len(self.grid_input)
    
    def __getitem__(self, idx):
        if self.power is None:
            return (
                self.grid_input[idx],
                self.grid_target[idx],
                self.wind_speed[idx]
            )
        return (
            self.grid_input[idx],
            self.grid_target[idx],
            self.wind_speed[idx],
            self.power[idx]
        )


def create_sequences(
    grid_hourly: np.ndarray,
    wind_speed_hourly: np.ndarray,
    power_hourly: np.ndarray = None,
    input_len: int = 24,
    output_len: int = 24,
    stride: int = 24
) -> tuple:
    """
    创建输入-输出序列
    
    Args:
        grid_hourly: [N_hours, C, H, W]
        wind_speed_hourly: [N_days, 24]
        power_hourly: [N_days, 24]
        input_len: 输入序列长度
        output_len: 输出序列长度
        stride: 滑动步长 (24 = 按天滑动)
    
    Returns:
        grid_input, grid_target, wind_speed, power
    """
    n_hours = len(grid_hourly)
    total_len = input_len + output_len
    
    grid_inputs = []
    grid_targets = []
    wind_speeds = []
    powers = [] if power_hourly is not None else None
    
    # 按天滑动
    for day_idx in range(len(wind_speed_hourly) - 1):  # -1 因为需要下一天作为目标
        hour_start = day_idx * 24
        
        if hour_start + total_len > n_hours:
            break
        
        # 输入: 当天 24 小时
        grid_in = grid_hourly[hour_start:hour_start + input_len]
        # 目标: 下一天 24 小时 (或同一天，取决于预测任务)
        grid_out = grid_hourly[hour_start + input_len:hour_start + total_len]
        
        # 精确点目标: 下一天
        ws = wind_speed_hourly[day_idx + 1]
        pw = power_hourly[day_idx + 1] if power_hourly is not None else None
        
        grid_inputs.append(grid_in)
        grid_targets.append(grid_out)
        wind_speeds.append(ws)
        if powers is not None:
            powers.append(pw)
    
    return (
        np.array(grid_inputs),
        np.array(grid_targets),
        np.array(wind_speeds),
        np.array(powers) if powers is not None else None
    )


# ==================== 训练器 ====================

class Stage2Trainer:
    """阶段2训练器：冻结主干，训练校正头"""
    
    def __init__(self, configs, turbine_coord: tuple, turbine_sample_coord: tuple = None,
                 checkpoint_path: str = None, enable_power_training: bool = True):
        self.configs = configs
        self.device = configs.device
        self.enable_power_training = enable_power_training
        
        # 初始化模型（启用时序LSTM风速校正）
        sample_coords = [turbine_sample_coord] if turbine_sample_coord is not None else None
        self.model = MFWPN_Model(
            turbine_coords=[turbine_coord],
            turbine_sample_coords=sample_coords,
            enable_wind_correction=True,
            use_temporal_correction=True,    # 使用时序LSTM
            wind_correction_hidden=128,      # 隐藏层维度
            lstm_hidden=128,                 # LSTM隐藏层
            lstm_layers=2,                   # LSTM层数
            wind_correction_dropout=0.2,     # Dropout
            use_time_embedding=True,         # 时间嵌入
            use_corrected_speed_for_power=self.enable_power_training,
            power_roi_size=5,
            power_conv_channels=32,
            power_mlp_hidden=64,
        ).to(self.device)
        
        # 加载预训练权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_pretrained(checkpoint_path)
        
        # 冻结主干
        self._freeze_backbone()
        
        # 只优化校正头和功率头
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.mse_loss = nn.MSELoss()
        self.normalizer = None
    
    def _load_pretrained(self, path: str):
        """加载预训练权重"""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get('net', checkpoint)
        
        # 只加载匹配的权重
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained weights from {path}")
    
    def _freeze_backbone(self):
        """冻结主干网络"""
        frozen_modules = ['enc', 'dec', 'hid_w', 'hid_tz', 'channel_f', 'channel_i', 'gate', 'ele_conv']
        
        for name, param in self.model.named_parameters():
            should_freeze = any(name.startswith(m) for m in frozen_modules)
            if should_freeze:
                param.requires_grad = False
        
        # 统计
        total = sum(p.numel() for p in self.model.parameters())
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"Frozen {frozen}/{total} parameters ({frozen/total*100:.1f}%)")
    
    def set_normalizer(self, normalizer: Normalizer):
        self.normalizer = normalizer

    def _compute_power_loss(self, power_pred, power_true):
        if (not self.enable_power_training) or power_true is None or power_pred is None:
            return None

        pred = power_pred.squeeze(-1).squeeze(-1)
        target = power_true
        valid = torch.isfinite(target)

        if not torch.any(valid):
            return None

        return torch.mean((pred[valid] - target[valid]) ** 2)
    
    def train_epoch(self, dataloader, ele):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        wind_loss_sum = 0
        power_loss_sum = 0
        power_batches = 0
        
        ele_tensor = torch.tensor(ele, dtype=torch.float32, device=self.device)
        
        for batch in dataloader:
            if len(batch) == 4:
                grid_input, grid_target, wind_speed_true, power_true = batch
            else:
                grid_input, grid_target, wind_speed_true = batch
                power_true = None
            
            grid_input = grid_input.float().to(self.device)
            wind_speed_true = wind_speed_true.float().to(self.device)
            if power_true is not None:
                power_true = power_true.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(grid_input, ele_tensor)
            
            # 风速校正损失
            corrected_speed = outputs['corrected_speed']  # [B, T, 1]
            wind_loss = self.mse_loss(corrected_speed.squeeze(-1), wind_speed_true)
            
            # 功率预测损失
            power_pred = outputs.get('power')
            power_loss = self._compute_power_loss(power_pred, power_true)
            
            # 总损失
            loss = wind_loss if power_loss is None else (wind_loss + power_loss)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            wind_loss_sum += wind_loss.item()
            if power_loss is not None:
                power_loss_sum += power_loss.item()
                power_batches += 1
        
        n = len(dataloader)
        avg_power_loss = (power_loss_sum / power_batches) if power_batches > 0 else None
        return total_loss / n, wind_loss_sum / n, avg_power_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader, ele):
        """评估"""
        self.model.eval()
        total_loss = 0
        wind_loss_sum = 0
        power_loss_sum = 0
        power_batches = 0
        
        ele_tensor = torch.tensor(ele, dtype=torch.float32, device=self.device)
        
        for batch in dataloader:
            if len(batch) == 4:
                grid_input, grid_target, wind_speed_true, power_true = batch
            else:
                grid_input, grid_target, wind_speed_true = batch
                power_true = None
            
            grid_input = grid_input.float().to(self.device)
            wind_speed_true = wind_speed_true.float().to(self.device)
            if power_true is not None:
                power_true = power_true.float().to(self.device)
            
            outputs = self.model(grid_input, ele_tensor)
            
            corrected_speed = outputs['corrected_speed']
            wind_loss = self.mse_loss(corrected_speed.squeeze(-1), wind_speed_true)
            
            power_pred = outputs.get('power')
            power_loss = self._compute_power_loss(power_pred, power_true)
            
            loss = wind_loss if power_loss is None else (wind_loss + power_loss)
            
            total_loss += loss.item()
            wind_loss_sum += wind_loss.item()
            if power_loss is not None:
                power_loss_sum += power_loss.item()
                power_batches += 1
        
        n = len(dataloader)
        avg_power_loss = (power_loss_sum / power_batches) if power_batches > 0 else None
        return total_loss / n, wind_loss_sum / n, avg_power_loss
    
    def train(self, train_loader, val_loader, ele, num_epochs: int, save_path: str):
        """完整训练流程"""
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_wind, train_power = self.train_epoch(train_loader, ele)
            val_loss, val_wind, val_power = self.evaluate(val_loader, ele)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_msg = f"  Train - Total: {train_loss:.4f}, Wind: {train_wind:.4f}"
            val_msg = f"  Val   - Total: {val_loss:.4f}, Wind: {val_wind:.4f}"
            train_msg += f", Power: {train_power:.4f}" if train_power is not None else ", Power: N/A"
            val_msg += f", Power: {val_power:.4f}" if val_power is not None else ", Power: N/A"
            print(train_msg)
            print(val_msg)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.save_model(save_path)
                print(f"  -> Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("Early stopping!")
                    break
        
        print(f"Training finished. Best val loss: {best_loss:.4f}")
    
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    @torch.no_grad()
    def predict(self, grid_input: np.ndarray, ele: np.ndarray) -> dict:
        """预测"""
        self.model.eval()
        
        grid_tensor = torch.tensor(grid_input, dtype=torch.float32, device=self.device)
        ele_tensor = torch.tensor(ele, dtype=torch.float32, device=self.device)
        
        if grid_tensor.dim() == 4:
            grid_tensor = grid_tensor.unsqueeze(0)
        
        outputs = self.model(grid_tensor, ele_tensor)

        result = {
            'wind_field': outputs['wind'].cpu().numpy(),
            'interp_speed': outputs['interp_speed'].cpu().numpy(),
            'corrected_speed': outputs['corrected_speed'].cpu().numpy(),
        }
        if outputs.get('power') is not None:
            result['power'] = outputs['power'].cpu().numpy()
        
        # 反归一化
        if self.normalizer:
            result['corrected_speed_original'] = self.normalizer.inverse_transform(
                result['corrected_speed'], 'wind_speed'
            )
            if 'power' in result and 'power' in self.normalizer.stats:
                result['power_original'] = self.normalizer.inverse_transform(
                    result['power'], 'power'
                )
        
        return result


# ==================== 主函数 ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Stage2 training with optional power supervision')
    parser.add_argument('--wind-only', action='store_true', help='Train only wind correction head without power supervision')
    parser.add_argument('--allow-missing-power', action='store_true', help='Allow NaN values in power labels and mask them in loss')
    parser.add_argument('--turbine-dir', type=str, default='data/turbine_points', help='Directory containing turbine_wind_speed.npy, turbine_dates.npy and optional turbine_power.npy')
    parser.add_argument('--power-path', type=str, default=None, help='Path to turbine power npy; default is <turbine-dir>/turbine_power.npy')
    parser.add_argument('--turbine-coord', type=str, default=None, help='Grid coordinate as "h,w" (e.g. "32,34"). If not set, try turbine_meta.json then fallback.')
    parser.add_argument('--turbine-lat', type=float, default=None, help='Turbine latitude in decimal degrees')
    parser.add_argument('--turbine-lon', type=float, default=None, help='Turbine longitude in decimal degrees')
    parser.add_argument('--lat-min', type=float, default=38.25, help='Grid south boundary latitude')
    parser.add_argument('--lat-max', type=float, default=54.0, help='Grid north boundary latitude')
    parser.add_argument('--lon-min', type=float, default=116.0, help='Grid west boundary longitude')
    parser.add_argument('--lon-max', type=float, default=135.75, help='Grid east boundary longitude')
    parser.add_argument('--coord-rounding', choices=['round', 'floor', 'ceil'], default='round',
                        help='Rounding strategy when converting decimal lat/lon to integer grid indices')
    parser.add_argument('--test-start-date', type=str, default='2025-11-01',
                        help='Test split start date (inclusive), format YYYY-MM-DD')
    parser.add_argument('--split-mode', choices=['date', 'ratio'], default='date',
                        help='Dataset split mode: by date threshold or by ratios')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train ratio when --split-mode ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation ratio when --split-mode ratio')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Test ratio when --split-mode ratio')
    parser.add_argument('--ratio-split-strategy', choices=['chronological', 'random'], default='chronological',
                        help='How to split when using ratio mode')
    parser.add_argument('--split-random-state', type=int, default=42,
                        help='Random seed for random ratio split')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Training device selection')
    parser.add_argument('--result-dir', type=str, default='result/exp', help='Directory to save stage2 experiment outputs')
    return parser.parse_args()


def latlon_to_grid(lat: float, lon: float, lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                   grid_h: int, grid_w: int, rounding: str):
    if not (lat_min < lat_max and lon_min < lon_max):
        raise ValueError('Invalid grid extent for lat/lon conversion.')
    if grid_h <= 1 or grid_w <= 1:
        raise ValueError('Grid shape must be greater than 1 for lat/lon conversion.')

    frac_h = (lat_max - lat) / (lat_max - lat_min) * (grid_h - 1)
    frac_w = (lon - lon_min) / (lon_max - lon_min) * (grid_w - 1)

    if rounding == 'floor':
        h_idx = int(np.floor(frac_h))
        w_idx = int(np.floor(frac_w))
    elif rounding == 'ceil':
        h_idx = int(np.ceil(frac_h))
        w_idx = int(np.ceil(frac_w))
    else:
        h_idx = int(np.round(frac_h))
        w_idx = int(np.round(frac_w))

    h_idx = max(0, min(grid_h - 1, h_idx))
    w_idx = max(0, min(grid_w - 1, w_idx))
    return (h_idx, w_idx), (float(frac_h), float(frac_w))


def resolve_turbine_coord(coord_arg: str, turbine_dir: str, default_coord: tuple,
                          turbine_lat: float, turbine_lon: float,
                          lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                          grid_h: int, grid_w: int, coord_rounding: str):
    if coord_arg:
        h_str, w_str = coord_arg.split(',')
        return (int(h_str), int(w_str)), None

    if turbine_lat is not None and turbine_lon is not None:
        coord, frac = latlon_to_grid(
            lat=turbine_lat,
            lon=turbine_lon,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            grid_h=grid_h,
            grid_w=grid_w,
            rounding=coord_rounding,
        )
        return coord, {
            'lat': turbine_lat,
            'lon': turbine_lon,
            'frac_h': frac[0],
            'frac_w': frac[1],
            'rounding': coord_rounding,
        }

    meta_path = os.path.join(turbine_dir, 'turbine_meta.json')
    if os.path.exists(meta_path):
        import json
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if turbine_lat is None and turbine_lon is None and 'latitude' in meta and 'longitude' in meta:
            coord, frac = latlon_to_grid(
                lat=float(meta['latitude']),
                lon=float(meta['longitude']),
                lat_min=float(meta.get('grid_extent', {}).get('lat_min', lat_min)),
                lat_max=float(meta.get('grid_extent', {}).get('lat_max', lat_max)),
                lon_min=float(meta.get('grid_extent', {}).get('lon_min', lon_min)),
                lon_max=float(meta.get('grid_extent', {}).get('lon_max', lon_max)),
                grid_h=grid_h,
                grid_w=grid_w,
                rounding=coord_rounding,
            )
            return coord, {
                'lat': float(meta['latitude']),
                'lon': float(meta['longitude']),
                'frac_h': frac[0],
                'frac_w': frac[1],
                'rounding': coord_rounding,
            }
        grid_coord = meta.get('grid_coord')
        if isinstance(grid_coord, list) and len(grid_coord) == 2:
            return (int(grid_coord[0]), int(grid_coord[1])), None

    return default_coord, None


def main():
    args = parse_args()
    if args.device == 'cpu':
        configs.device = torch.device('cpu')
    elif args.device == 'cuda':
        configs.device = torch.device('cuda:0')
    else:
        configs.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Stage 2 Training: Wind Speed Correction + Power Prediction")
    print("=" * 60)
    print(f"Device: {configs.device}")
    
    # ========== 1. 配置 ==========
    # 精确点坐标 (需要根据实际位置计算)
    # 辰阳风电场坐标需要转换为网格索引
    DEFAULT_TURBINE_COORD = (32, 34)
    TURBINE_COORD, coord_detail = resolve_turbine_coord(
        args.turbine_coord,
        args.turbine_dir,
        DEFAULT_TURBINE_COORD,
        args.turbine_lat,
        args.turbine_lon,
        args.lat_min,
        args.lat_max,
        args.lon_min,
        args.lon_max,
        64,
        80,
        args.coord_rounding,
    )
    TURBINE_SAMPLE_COORD = None
    if coord_detail is not None:
        TURBINE_SAMPLE_COORD = (coord_detail['frac_h'], coord_detail['frac_w'])
    
    GRID_START_DATE = "2020-01-01"  # UTC时间
    TEST_START_DATE = args.test_start_date
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    
    # ========== 2. 加载数据 ==========
    print("\n[1] Loading data...")
    
    # 训练网格数据 (UTC: 2020-01-01 ~ 2024-12-30)
    uv_train = np.load("data/Northeast/uv100_train.npy").astype(np.float32)
    zt_train = np.load("data/Northeast/1000zt_train.npy").astype(np.float32)
    grid_train = np.concatenate((uv_train, zt_train), axis=1)  # [N, 4, H, W]
    del uv_train, zt_train
    print(f"  Train grid data: {grid_train.shape}")
    
    # 测试网格数据 (UTC: 2024-12-31 ~ 2025-11-01)
    uv_test = np.load("data/Northeast/uv100_test.npy").astype(np.float32)
    zt_test = np.load("data/Northeast/1000zt_test.npy").astype(np.float32)
    grid_test = np.concatenate((uv_test, zt_test), axis=1)
    del uv_test, zt_test
    print(f"  Test grid data: {grid_test.shape}")
    
    # 高程
    ele = np.load('data/Northeast/DEM_northeast.npy').astype(np.float32)
    ele[ele < 0] = 0
    ele = (ele - ele.mean()) / ele.std()
    
    # 精确点数据
    turbine_dir = args.turbine_dir
    baseline_name = os.path.basename(os.path.normpath(turbine_dir)) or 'default'
    result_dir = os.path.join(args.result_dir, baseline_name)
    os.makedirs(result_dir, exist_ok=True)

    wind_path = os.path.join(turbine_dir, 'turbine_wind_speed.npy')
    dates_path = os.path.join(turbine_dir, 'turbine_dates.npy')
    power_path = args.power_path if args.power_path is not None else os.path.join(turbine_dir, 'turbine_power.npy')

    if not os.path.exists(wind_path):
        raise FileNotFoundError(f'Turbine wind file not found: {wind_path}')
    if not os.path.exists(dates_path):
        raise FileNotFoundError(f'Turbine dates file not found: {dates_path}')

    wind_speed = np.load(wind_path)
    power = None
    if not args.wind_only:
        if os.path.exists(power_path):
            power = np.load(power_path)
        else:
            raise FileNotFoundError(f"Power data file not found: {power_path}")
    dates = np.load(dates_path, allow_pickle=True)
    print(f"  Turbine wind speed: {wind_speed.shape}")
    print(f"  Turbine power: {power.shape}" if power is not None else "  Turbine power: disabled (wind-only)")
    print(f"  Date range: {dates[0]} ~ {dates[-1]}")
    print(f"  Turbine coord (grid): {TURBINE_COORD}")
    if coord_detail is not None:
        print(
            f"  Coord source lat/lon=({coord_detail['lat']:.6f}, {coord_detail['lon']:.6f}), "
            f"frac_idx=({coord_detail['frac_h']:.4f}, {coord_detail['frac_w']:.4f}), "
            f"rounding={coord_detail['rounding']}"
        )
    print(f"  Result dir: {result_dir}")
    
    # ========== 3. 时间对齐 ==========
    print("\n[2] Aligning grid data with turbine data...")
    # 合并训练和测试网格数据
    aligned_grid, valid_days = align_grid_with_turbine(
        grid_train, GRID_START_DATE, dates,
        use_test_data=True,
        test_grid_data=grid_test,
        test_start_date="2024-12-31"  # 测试集起始日期
    )
    del grid_train, grid_test
    
    # 筛选有效的精确点数据
    valid_dates = dates[valid_days]
    wind_speed = wind_speed[valid_days]
    if power is not None:
        power = power[valid_days]
    print(f"  Aligned grid: {aligned_grid.shape}")
    print(f"  Valid turbine days: {len(valid_days)}")
    print(f"  Valid date range: {valid_dates[0]} ~ {valid_dates[-1]}")
    
    # ========== 4. 归一化 ==========
    print("\n[3] Normalizing data...")
    normalizer = Normalizer()
    
    # 风速: Z-score
    normalizer.fit_zscore(wind_speed, 'wind_speed')
    wind_speed_norm = normalizer.transform(wind_speed, 'wind_speed')
    print(f"  Wind speed: mean={wind_speed.mean():.2f}, std={wind_speed.std():.2f}")
    print(f"  -> Normalized: mean={wind_speed_norm.mean():.4f}, std={wind_speed_norm.std():.4f}")
    
    power_norm = None
    if power is not None:
        if np.isfinite(power).all():
            normalizer.fit_minmax(power, 'power', feature_range=(0, 1))
            power_norm = normalizer.transform(power, 'power')
            print(f"  Power: min={power.min():.2f}, max={power.max():.2f}")
            print(f"  -> Normalized: min={power_norm.min():.4f}, max={power_norm.max():.4f}")
        else:
            if not args.allow_missing_power:
                raise ValueError("Power data contains missing values (NaN/Inf). Use --allow-missing-power to continue.")
            valid_mask = np.isfinite(power)
            if not np.any(valid_mask):
                print("  Power: no valid values found, fallback to wind-only training for this run.")
                power = None
                power_norm = None
            else:
                pmin = power[valid_mask].min()
                pmax = power[valid_mask].max()
                normalizer.stats['power'] = {
                    'type': 'minmax',
                    'min': pmin,
                    'max': pmax,
                    'range': (0, 1)
                }
                power_norm = np.full(power.shape, np.nan, dtype=np.float32)
                power_norm[valid_mask] = normalizer.transform(power[valid_mask], 'power').astype(np.float32)
                print(f"  Power: valid={valid_mask.sum()}, missing={(~valid_mask).sum()}")
                print(f"  -> Normalized valid range: min={np.nanmin(power_norm):.4f}, max={np.nanmax(power_norm):.4f}")
    
    # 保存归一化参数
    normalizer.save(f"{turbine_dir}/normalizer.npy")
    
    # ========== 5. 创建序列 ==========
    print("\n[4] Creating sequences...")
    grid_input, grid_target, ws_seq, pw_seq = create_sequences(
        aligned_grid, wind_speed_norm, power_norm,
        input_len=24, output_len=24, stride=24
    )
    print(f"  Grid input: {grid_input.shape}")
    print(f"  Grid target: {grid_target.shape}")
    print(f"  Wind speed: {ws_seq.shape}")
    print(f"  Power: {pw_seq.shape}" if pw_seq is not None else "  Power: N/A")
    
    # ========== 6. 划分数据集 ==========
    print("\n[5] Splitting dataset by date...")
    seq_target_dates = valid_dates[1:1 + len(grid_input)]

    if args.split_mode == 'ratio':
        ratios = np.array([args.train_ratio, args.val_ratio, args.test_ratio], dtype=np.float64)
        if np.any(ratios <= 0):
            raise ValueError('train/val/test ratios must all be > 0 in ratio split mode.')
        if not np.isclose(ratios.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f'Ratios must sum to 1.0, got train+val+test={ratios.sum():.6f}.')

        n_seq = len(grid_input)
        if n_seq < 5:
            raise ValueError(f'Not enough sequences for ratio split: {n_seq}')

        if args.ratio_split_strategy == 'chronological':
            train_count = int(np.floor(n_seq * args.train_ratio))
            val_count = int(np.floor(n_seq * args.val_ratio))
            test_count = n_seq - train_count - val_count

            if min(train_count, val_count, test_count) <= 0:
                raise ValueError(
                    f'Invalid split counts train={train_count}, val={val_count}, test={test_count}.')

            train_idx = np.arange(0, train_count)
            val_idx = np.arange(train_count, train_count + val_count)
            test_indices = np.arange(train_count + val_count, n_seq)
        else:
            all_idx = np.arange(n_seq)
            train_val_idx, test_indices = train_test_split(
                all_idx,
                test_size=args.test_ratio,
                random_state=args.split_random_state,
                shuffle=True,
            )
            rel_val_ratio = args.val_ratio / (args.train_ratio + args.val_ratio)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=rel_val_ratio,
                random_state=args.split_random_state,
                shuffle=True,
            )

        print(
            f"  Split mode: ratio ({args.ratio_split_strategy}) | "
            f"train={args.train_ratio:.2f}, val={args.val_ratio:.2f}, test={args.test_ratio:.2f}"
        )
    else:
        test_start = datetime.strptime(TEST_START_DATE, "%Y-%m-%d")
        test_start_idx = None

        for i, date_str in enumerate(valid_dates):
            date = datetime.strptime(str(date_str), "%Y-%m-%d")
            if date >= test_start:
                test_start_idx = i
                break

        if test_start_idx is None:
            print(f"  Warning: No data found after {TEST_START_DATE}, using last 10% as test")
            test_start_idx = int(len(valid_dates) * 0.9)

        print(f"  Test start date: {valid_dates[test_start_idx]} (index {test_start_idx})")
        print(f"  Test end date: {valid_dates[-1]}")

        # 序列索引对应关系：序列i的输出对应valid_dates[i+1]
        # 所以测试集序列索引应该从 test_start_idx - 1 开始
        test_seq_start = max(0, test_start_idx - 1)
        train_val_indices = np.arange(test_seq_start)
        test_indices = np.arange(test_seq_start, len(grid_input))

        if len(train_val_indices) > 1:
            train_idx, val_idx = train_test_split(
                train_val_indices,
                test_size=0.1,
                random_state=args.split_random_state,
                shuffle=True,
            )
        else:
            train_idx = train_val_indices
            val_idx = np.array([], dtype=int)
    
    if pw_seq is not None:
        train_dataset = TurbineDataset(
            grid_input[train_idx], grid_target[train_idx],
            ws_seq[train_idx], pw_seq[train_idx],
            indices=train_idx
        )
        val_dataset = TurbineDataset(
            grid_input[val_idx], grid_target[val_idx],
            ws_seq[val_idx], pw_seq[val_idx],
            indices=val_idx
        ) if len(val_idx) > 0 else None
        test_dataset = TurbineDataset(
            grid_input[test_indices], grid_target[test_indices],
            ws_seq[test_indices], pw_seq[test_indices],
            indices=test_indices
        )
    else:
        train_dataset = TurbineDataset(
            grid_input[train_idx], grid_target[train_idx],
            ws_seq[train_idx], None,
            indices=train_idx
        )
        val_dataset = TurbineDataset(
            grid_input[val_idx], grid_target[val_idx],
            ws_seq[val_idx], None,
            indices=val_idx
        ) if len(val_idx) > 0 else None
        test_dataset = TurbineDataset(
            grid_input[test_indices], grid_target[test_indices],
            ws_seq[test_indices], None,
            indices=test_indices
        )
    
    test_dates = seq_target_dates[test_indices] if len(test_indices) > 0 else np.array([], dtype=object)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset) if val_dataset else 0} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Test dates: {test_dates[0] if len(test_dates) > 0 else 'N/A'} ~ {test_dates[-1] if len(test_dates) > 0 else 'N/A'}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # batch_size=1便于按日计算
    
    # ========== 7. 训练 ==========
    print("\n[6] Initializing trainer...")
    try:
        trainer = Stage2Trainer(
            configs,
            turbine_coord=TURBINE_COORD,
            turbine_sample_coord=TURBINE_SAMPLE_COORD,
            checkpoint_path="chkfile/checkpoint_mfwpn.chk",  # 预训练权重
            enable_power_training=(pw_seq is not None and (not args.wind_only))
        )
    except RuntimeError as e:
        msg = str(e).lower()
        if configs.device.type == 'cuda' and ('no kernel image is available' in msg or 'cuda error' in msg):
            print('CUDA initialization failed, fallback to CPU...')
            configs.device = torch.device('cpu')
            trainer = Stage2Trainer(
                configs,
                turbine_coord=TURBINE_COORD,
                turbine_sample_coord=TURBINE_SAMPLE_COORD,
                checkpoint_path="chkfile/checkpoint_mfwpn.chk",  # 预训练权重
                enable_power_training=(pw_seq is not None and (not args.wind_only))
            )
        else:
            raise
    trainer.set_normalizer(normalizer)
    
    print("\n[7] Training...")
    if val_loader:
        trainer.train(
            train_loader, val_loader, ele,
            num_epochs=NUM_EPOCHS,
            save_path=os.path.join(result_dir, "checkpoint_stage2.chk")
        )
    else:
        print("  Warning: No validation data, training with train data only")
        # 简化训练，不做验证
        for epoch in range(NUM_EPOCHS):
            train_loss, train_wind, train_power = trainer.train_epoch(train_loader, ele)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {train_loss:.4f}")
        trainer.save_model(os.path.join(result_dir, "checkpoint_stage2.chk"))
    
    # ========== 8. 测试 ==========
    print("\n[8] Testing...")
    trainer.load_model(os.path.join(result_dir, "checkpoint_stage2.chk"))
    test_loss, test_wind, test_power = trainer.evaluate(test_loader, ele)
    test_msg = f"  Test - Total: {test_loss:.4f}, Wind: {test_wind:.4f}"
    test_msg += f", Power: {test_power:.4f}" if test_power is not None else ", Power: N/A"
    print(test_msg)
    
    # ========== 9. 保存测试预测结果 ==========
    print("\n[9] Saving predictions...")
    all_preds = []
    all_true_wind = []
    all_true_power = []
    
    trainer.model.eval()
    ele_tensor = torch.tensor(ele, dtype=torch.float32, device=configs.device)
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                grid_input, _, wind_true, power_true = batch
            else:
                grid_input, _, wind_true = batch
                power_true = None
            grid_input = grid_input.float().to(configs.device)
            
            outputs = trainer.model(grid_input, ele_tensor)

            pred_item = {'corrected_speed': outputs['corrected_speed'].cpu().numpy()}
            if trainer.enable_power_training and outputs.get('power') is not None:
                pred_item['power'] = outputs['power'].cpu().numpy()
            all_preds.append(pred_item)
            all_true_wind.append(wind_true.numpy())
            if power_true is not None:
                all_true_power.append(power_true.numpy())
    
    # 合并预测结果
    pred_wind = np.concatenate([p['corrected_speed'] for p in all_preds], axis=0)
    true_wind = np.concatenate(all_true_wind, axis=0)
    has_power_eval = trainer.enable_power_training and len(all_true_power) > 0 and all('power' in p for p in all_preds)
    pred_power = np.concatenate([p['power'] for p in all_preds], axis=0) if has_power_eval else None
    true_power = np.concatenate(all_true_power, axis=0) if has_power_eval else None
    
    # 反归一化
    pred_wind_orig = normalizer.inverse_transform(pred_wind.squeeze(), 'wind_speed')
    true_wind_orig = normalizer.inverse_transform(true_wind, 'wind_speed')
    pred_power_orig = normalizer.inverse_transform(pred_power.squeeze(), 'power') if pred_power is not None else None
    true_power_orig = normalizer.inverse_transform(true_power, 'power') if true_power is not None else None
    
    # ========== 10. 计算每小时的MAE和RMSE ==========
    print("\n=== Test Results by Hour (Original Scale) ===")
    print("\n--- Wind Speed ---")
    print(f"{'Hour':<6} {'MAE (m/s)':<12} {'RMSE (m/s)':<12}")
    print("-" * 30)
    
    wind_mae_hourly = []
    wind_rmse_hourly = []
    
    for h in range(24):
        # 每小时的误差
        if pred_wind_orig.ndim == 1:
            # 如果数据被压缩了
            h_pred = pred_wind_orig[h::24]
            h_true = true_wind_orig[:, h] if true_wind_orig.ndim > 1 else true_wind_orig[h::24]
        else:
            h_pred = pred_wind_orig[:, h]
            h_true = true_wind_orig[:, h]
        
        mae = np.abs(h_pred - h_true).mean()
        rmse = np.sqrt(((h_pred - h_true) ** 2).mean())
        wind_mae_hourly.append(mae)
        wind_rmse_hourly.append(rmse)
        print(f"{h+1:<6} {mae:<12.3f} {rmse:<12.3f}")
    
    print("-" * 30)
    print(f"{'Avg':<6} {np.mean(wind_mae_hourly):<12.3f} {np.mean(wind_rmse_hourly):<12.3f}")
    
    power_mae_hourly = []
    power_rmse_hourly = []
    if pred_power_orig is not None and true_power_orig is not None:
        print("\n--- Power ---")
        print(f"{'Hour':<6} {'MAE (MW)':<12} {'RMSE (MW)':<12}")
        print("-" * 30)

        for h in range(24):
            if pred_power_orig.ndim == 1:
                h_pred = pred_power_orig[h::24]
                h_true = true_power_orig[:, h] if true_power_orig.ndim > 1 else true_power_orig[h::24]
            else:
                h_pred = pred_power_orig[:, h]
                h_true = true_power_orig[:, h]

            mae = np.abs(h_pred - h_true).mean()
            rmse = np.sqrt(((h_pred - h_true) ** 2).mean())
            power_mae_hourly.append(mae)
            power_rmse_hourly.append(rmse)
            print(f"{h+1:<6} {mae:<12.3f} {rmse:<12.3f}")

        print("-" * 30)
        print(f"{'Avg':<6} {np.mean(power_mae_hourly):<12.3f} {np.mean(power_rmse_hourly):<12.3f}")
    else:
        print("\n--- Power ---")
        print("Power evaluation skipped (wind-only mode or missing power labels).")
    
    # 总体统计
    wind_mae = np.mean(wind_mae_hourly)
    wind_rmse = np.mean(wind_rmse_hourly)
    power_mae = np.mean(power_mae_hourly) if len(power_mae_hourly) > 0 else None
    power_rmse = np.mean(power_rmse_hourly) if len(power_rmse_hourly) > 0 else None
    
    print(f"\n=== Overall Test Results ===")
    print(f"  Wind Speed - MAE: {wind_mae:.3f} m/s, RMSE: {wind_rmse:.3f} m/s")
    if power_mae is not None and power_rmse is not None:
        print(f"  Power      - MAE: {power_mae:.3f} MW,  RMSE: {power_rmse:.3f} MW")
    else:
        print("  Power      - skipped")
    
    # 保存预测结果
    save_payload = {
        'pred_wind': pred_wind_orig,
        'true_wind': true_wind_orig,
        'test_dates': test_dates,
        'wind_mae_hourly': np.array(wind_mae_hourly),
        'wind_rmse_hourly': np.array(wind_rmse_hourly),
    }
    if pred_power_orig is not None and true_power_orig is not None:
        save_payload.update({
            'pred_power': pred_power_orig,
            'true_power': true_power_orig,
            'power_mae_hourly': np.array(power_mae_hourly),
            'power_rmse_hourly': np.array(power_rmse_hourly),
        })

    pred_out_path = os.path.join(result_dir, 'test_predictions.npz')
    np.savez(pred_out_path, **save_payload)
    print(f"\nPredictions saved to {pred_out_path}")
    
    print("\n" + "=" * 60)
    print("Stage 2 Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
