#!/usr/bin/env python3
"""
阶段2训练脚本：训练精确点风速校正头和功率预测头
- 加载预训练主干并冻结
- 时间对齐网格数据与精确点数据
- 归一化处理
"""

import os
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
    
    def __init__(self, grid_input, grid_target, wind_speed, power, indices=None):
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
        return (
            self.grid_input[idx],
            self.grid_target[idx],
            self.wind_speed[idx],
            self.power[idx]
        )


def create_sequences(
    grid_hourly: np.ndarray,
    wind_speed_hourly: np.ndarray,
    power_hourly: np.ndarray,
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
    powers = []
    
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
        pw = power_hourly[day_idx + 1]
        
        grid_inputs.append(grid_in)
        grid_targets.append(grid_out)
        wind_speeds.append(ws)
        powers.append(pw)
    
    return (
        np.array(grid_inputs),
        np.array(grid_targets),
        np.array(wind_speeds),
        np.array(powers)
    )


# ==================== 训练器 ====================

class Stage2Trainer:
    """阶段2训练器：冻结主干，训练校正头"""
    
    def __init__(self, configs, turbine_coord: tuple, checkpoint_path: str = None):
        self.configs = configs
        self.device = configs.device
        
        # 初始化模型（启用风速校正）
        self.model = MFWPN_Model(
            turbine_coords=[turbine_coord],
            enable_wind_correction=True,
            wind_correction_hidden=64,
            use_corrected_speed_for_power=True,
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
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
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
    
    def train_epoch(self, dataloader, ele):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        wind_loss_sum = 0
        power_loss_sum = 0
        
        ele_tensor = torch.tensor(ele, dtype=torch.float32, device=self.device)
        
        for batch in dataloader:
            grid_input, grid_target, wind_speed_true, power_true = batch
            
            grid_input = grid_input.float().to(self.device)
            wind_speed_true = wind_speed_true.float().to(self.device)
            power_true = power_true.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(grid_input, ele_tensor)
            
            # 风速校正损失
            corrected_speed = outputs['corrected_speed']  # [B, T, 1]
            wind_loss = self.mse_loss(corrected_speed.squeeze(-1), wind_speed_true)
            
            # 功率预测损失
            power_pred = outputs['power']  # [B, T, 1, 1]
            power_loss = self.mse_loss(power_pred.squeeze(-1).squeeze(-1), power_true)
            
            # 总损失
            loss = wind_loss + power_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            wind_loss_sum += wind_loss.item()
            power_loss_sum += power_loss.item()
        
        n = len(dataloader)
        return total_loss / n, wind_loss_sum / n, power_loss_sum / n
    
    @torch.no_grad()
    def evaluate(self, dataloader, ele):
        """评估"""
        self.model.eval()
        total_loss = 0
        wind_loss_sum = 0
        power_loss_sum = 0
        
        ele_tensor = torch.tensor(ele, dtype=torch.float32, device=self.device)
        
        for batch in dataloader:
            grid_input, grid_target, wind_speed_true, power_true = batch
            
            grid_input = grid_input.float().to(self.device)
            wind_speed_true = wind_speed_true.float().to(self.device)
            power_true = power_true.float().to(self.device)
            
            outputs = self.model(grid_input, ele_tensor)
            
            corrected_speed = outputs['corrected_speed']
            wind_loss = self.mse_loss(corrected_speed.squeeze(-1), wind_speed_true)
            
            power_pred = outputs['power']
            power_loss = self.mse_loss(power_pred.squeeze(-1).squeeze(-1), power_true)
            
            loss = wind_loss + power_loss
            
            total_loss += loss.item()
            wind_loss_sum += wind_loss.item()
            power_loss_sum += power_loss.item()
        
        n = len(dataloader)
        return total_loss / n, wind_loss_sum / n, power_loss_sum / n
    
    def train(self, train_loader, val_loader, ele, num_epochs: int, save_path: str):
        """完整训练流程"""
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_wind, train_power = self.train_epoch(train_loader, ele)
            val_loss, val_wind, val_power = self.evaluate(val_loader, ele)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Total: {train_loss:.4f}, Wind: {train_wind:.4f}, Power: {train_power:.4f}")
            print(f"  Val   - Total: {val_loss:.4f}, Wind: {val_wind:.4f}, Power: {val_power:.4f}")
            
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
            'power': outputs['power'].cpu().numpy(),
        }
        
        # 反归一化
        if self.normalizer:
            result['corrected_speed_original'] = self.normalizer.inverse_transform(
                result['corrected_speed'], 'wind_speed'
            )
            result['power_original'] = self.normalizer.inverse_transform(
                result['power'], 'power'
            )
        
        return result


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("Stage 2 Training: Wind Speed Correction + Power Prediction")
    print("=" * 60)
    
    # ========== 1. 配置 ==========
    # 精确点坐标 (需要根据实际位置计算)
    # 辰阳风电场坐标需要转换为网格索引
    TURBINE_COORD = (32, 34)  # 辰阳风电场: 124.6138°E, 45.8470°N
    
    GRID_START_DATE = "2020-01-01"  # UTC时间
    TEST_START_DATE = "2025-08-01"  # 测试集起始日期（北京时间）
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
    turbine_dir = "data/turbine_points"
    wind_speed = np.load(f"{turbine_dir}/turbine_wind_speed.npy")
    power = np.load(f"{turbine_dir}/turbine_power.npy")
    dates = np.load(f"{turbine_dir}/turbine_dates.npy", allow_pickle=True)
    print(f"  Turbine wind speed: {wind_speed.shape}")
    print(f"  Turbine power: {power.shape}")
    print(f"  Date range: {dates[0]} ~ {dates[-1]}")
    
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
    
    # 功率: Min-Max [0, 1]
    normalizer.fit_minmax(power, 'power', feature_range=(0, 1))
    power_norm = normalizer.transform(power, 'power')
    print(f"  Power: min={power.min():.2f}, max={power.max():.2f}")
    print(f"  -> Normalized: min={power_norm.min():.4f}, max={power_norm.max():.4f}")
    
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
    print(f"  Power: {pw_seq.shape}")
    
    # ========== 6. 划分数据集 ==========
    # 测试集：2025-08-01 之后的数据
    # 训练集和验证集：2025-08-01 之前的数据
    print("\n[5] Splitting dataset by date...")
    
    # 找到测试集起始索引
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
    
    # 划分索引
    train_val_indices = np.arange(test_seq_start)
    test_indices = np.arange(test_seq_start, len(grid_input))
    
    # 从训练+验证集中划分出验证集 (10%)
    if len(train_val_indices) > 1:
        train_idx, val_idx = train_test_split(train_val_indices, test_size=0.1, random_state=42)
    else:
        train_idx = train_val_indices
        val_idx = np.array([], dtype=int)
    
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
    
    # 保存测试集日期信息
    test_dates = valid_dates[test_seq_start + 1:]  # +1因为输出对应下一天
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset) if val_dataset else 0} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Test dates: {test_dates[0] if len(test_dates) > 0 else 'N/A'} ~ {test_dates[-1] if len(test_dates) > 0 else 'N/A'}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # batch_size=1便于按日计算
    
    # ========== 7. 训练 ==========
    print("\n[6] Initializing trainer...")
    trainer = Stage2Trainer(
        configs,
        turbine_coord=TURBINE_COORD,
        checkpoint_path="chkfile/checkpoint_mfwpn.chk"  # 预训练权重
    )
    trainer.set_normalizer(normalizer)
    
    print("\n[7] Training...")
    if val_loader:
        trainer.train(
            train_loader, val_loader, ele,
            num_epochs=NUM_EPOCHS,
            save_path="chkfile/checkpoint_stage2.chk"
        )
    else:
        print("  Warning: No validation data, training with train data only")
        # 简化训练，不做验证
        for epoch in range(NUM_EPOCHS):
            train_loss, train_wind, train_power = trainer.train_epoch(train_loader, ele)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {train_loss:.4f}")
        trainer.save_model("chkfile/checkpoint_stage2.chk")
    
    # ========== 8. 测试 ==========
    print("\n[8] Testing...")
    trainer.load_model("chkfile/checkpoint_stage2.chk")
    test_loss, test_wind, test_power = trainer.evaluate(test_loader, ele)
    print(f"  Test - Total: {test_loss:.4f}, Wind: {test_wind:.4f}, Power: {test_power:.4f}")
    
    # ========== 9. 保存测试预测结果 ==========
    print("\n[9] Saving predictions...")
    all_preds = []
    all_true_wind = []
    all_true_power = []
    
    trainer.model.eval()
    ele_tensor = torch.tensor(ele, dtype=torch.float32, device=configs.device)
    
    with torch.no_grad():
        for batch in test_loader:
            grid_input, _, wind_true, power_true = batch
            grid_input = grid_input.float().to(configs.device)
            
            outputs = trainer.model(grid_input, ele_tensor)
            
            all_preds.append({
                'corrected_speed': outputs['corrected_speed'].cpu().numpy(),
                'power': outputs['power'].cpu().numpy()
            })
            all_true_wind.append(wind_true.numpy())
            all_true_power.append(power_true.numpy())
    
    # 合并预测结果
    pred_wind = np.concatenate([p['corrected_speed'] for p in all_preds], axis=0)
    pred_power = np.concatenate([p['power'] for p in all_preds], axis=0)
    true_wind = np.concatenate(all_true_wind, axis=0)
    true_power = np.concatenate(all_true_power, axis=0)
    
    # 反归一化
    pred_wind_orig = normalizer.inverse_transform(pred_wind.squeeze(), 'wind_speed')
    pred_power_orig = normalizer.inverse_transform(pred_power.squeeze(), 'power')
    true_wind_orig = normalizer.inverse_transform(true_wind, 'wind_speed')
    true_power_orig = normalizer.inverse_transform(true_power, 'power')
    
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
    
    print("\n--- Power ---")
    print(f"{'Hour':<6} {'MAE (MW)':<12} {'RMSE (MW)':<12}")
    print("-" * 30)
    
    power_mae_hourly = []
    power_rmse_hourly = []
    
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
    
    # 总体统计
    wind_mae = np.mean(wind_mae_hourly)
    wind_rmse = np.mean(wind_rmse_hourly)
    power_mae = np.mean(power_mae_hourly)
    power_rmse = np.mean(power_rmse_hourly)
    
    print(f"\n=== Overall Test Results ===")
    print(f"  Wind Speed - MAE: {wind_mae:.3f} m/s, RMSE: {wind_rmse:.3f} m/s")
    print(f"  Power      - MAE: {power_mae:.3f} MW,  RMSE: {power_rmse:.3f} MW")
    
    # 保存预测结果
    np.savez(
        f"{turbine_dir}/test_predictions.npz",
        pred_wind=pred_wind_orig,
        pred_power=pred_power_orig,
        true_wind=true_wind_orig,
        true_power=true_power_orig,
        test_dates=test_dates,
        wind_mae_hourly=np.array(wind_mae_hourly),
        wind_rmse_hourly=np.array(wind_rmse_hourly),
        power_mae_hourly=np.array(power_mae_hourly),
        power_rmse_hourly=np.array(power_rmse_hourly)
    )
    print(f"\nPredictions saved to {turbine_dir}/test_predictions.npz")
    
    print("\n" + "=" * 60)
    print("Stage 2 Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
