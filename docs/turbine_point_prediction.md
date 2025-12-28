# 精确点风速与功率预测技术文档

## 1. 需求概述

### 1.1 背景
原有 MFWPN 模型仅预测网格风场 (u, v 分量)，现需扩展为：
1. **精确点风速预测**：预测风机所在精确位置的真实风速（非简单插值）
2. **功率预测**：基于风速预测风机发电功率

### 1.2 核心挑战
| 问题 | 原因 |
|------|------|
| 插值误差大 | 网格分辨率有限，无法捕捉局部地形/湍流效应 |
| 数据量不匹配 | 精确点数据仅 13 个月，网格数据 5 年+ |

### 1.3 解决方案
采用**两阶段训练 + 校正头**架构：
- 阶段1：用全量网格数据预训练时空特征提取器
- 阶段2：冻结主干，用精确点数据训练校正头

---

## 2. 模型架构

### 2.1 整体结构
```
输入数据 [B, T, 4, H, W]
    │
    ▼
┌───────────────────────────────────┐
│         Encoder (冻结)            │
│   风场分支 + 温度气压分支          │
└───────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────┐
│       MidMetaNet (冻结)           │
│        时空特征融合               │
└───────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────┐
│         Decoder (冻结)            │
│       输出网格风场 Y              │
└───────────────────────────────────┘
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│   WindCorrectionHead    │          │    TurbinePowerHead     │
│      (可训练)           │────────▶│       (可训练)          │
│   精确点风速校正         │          │      功率回归           │
└─────────────────────────┘          └─────────────────────────┘
    │                                          │
    ▼                                          ▼
  corrected_speed                           power
  [B, T, N_turbines]                    [B, T, N_turbines, 1]
```

### 2.2 新增模块

#### WindCorrectionHead
```python
class WindCorrectionHead(nn.Module):
    """校正插值风速，预测精确位置的真实风速"""
    
    # 输入
    features: [B, T, C, H, W]      # 时空特征
    wind_field: [B, T, 2, H, W]    # 预测风场 (u, v)
    
    # 输出
    {
        'interp_speed': [B, T, N],      # 双线性插值风速
        'corrected_speed': [B, T, N]    # 校正后风速
    }
```

**工作流程**：
1. 从预测风场计算风速场 `√(u² + v²)`
2. 使用 `grid_sample` 双线性插值到精确点
3. 提取精确点周围 ROI 区域的局部特征
4. 拼接插值风速 + 局部特征，送入 MLP 得到校正后风速

#### TurbinePowerHead (已更新)
```python
class TurbinePowerHead(nn.Module):
    """从校正后风速预测功率"""
    
    # 新增参数
    use_corrected_speed: bool = False  # 是否使用校正后风速
    
    # 新增输入
    corrected_speed: Optional[B, T, N]  # 校正后风速
```

### 2.3 模型初始化参数

```python
model = MFWPN_Model(
    # ... 原有参数 ...
    turbine_coords=[(h1, w1), (h2, w2), ...],  # 精确点在特征图中的坐标
    enable_wind_correction=True,                # 启用风速校正
    wind_correction_hidden=64,                  # 校正头隐藏层维度
    use_corrected_speed_for_power=True,         # 功率预测使用校正后风速
)
```

---

## 3. 数据格式

### 3.1 精确点数据格式规范

#### 文件结构
```
data/
├── grid/                          # 网格数据 (已有)
│   ├── train/
│   │   └── *.npy
│   └── val/
│       └── *.npy
│
└── turbine_points/                # 精确点数据 (新增)
    ├── coords.json                # 坐标配置
    ├── train/
    │   └── turbine_YYYYMMDD.npz
    └── val/
        └── turbine_YYYYMMDD.npz
```

#### coords.json 格式
```json
{
    "feature_hw": [64, 80],
    "turbines": [
        {
            "id": "turbine_001",
            "name": "风机1号",
            "lat": 42.5678,
            "lon": 123.4567,
            "grid_h": 32,
            "grid_w": 45
        },
        {
            "id": "turbine_002",
            "name": "风机2号",
            "lat": 42.5700,
            "lon": 123.4600,
            "grid_h": 33,
            "grid_w": 46
        }
    ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 风机唯一标识 |
| `name` | str | 风机名称 |
| `lat`, `lon` | float | 经纬度坐标 |
| `grid_h`, `grid_w` | int | 在 64×80 特征图中的索引 (0-indexed) |

#### turbine_YYYYMMDD.npz 格式
```python
np.savez(
    'turbine_20240801.npz',
    timestamps=timestamps,      # [T] datetime64 时间戳
    wind_speed=wind_speed,      # [T, N_turbines] 实测风速 (m/s)
    power=power,                # [T, N_turbines] 实测功率 (kW)
    turbine_ids=turbine_ids     # [N_turbines] 风机ID列表
)
```

| 数组 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `timestamps` | [T] | datetime64 | 时间戳，与网格数据对齐 |
| `wind_speed` | [T, N] | float32 | 各风机实测风速 (m/s) |
| `power` | [T, N] | float32 | 各风机实测功率 (kW) |
| `turbine_ids` | [N] | str | 风机ID，与 coords.json 对应 |

### 3.2 坐标转换

将经纬度转换为网格索引：

```python
def latlon_to_grid(lat, lon, lat_range, lon_range, grid_h, grid_w):
    """
    Args:
        lat, lon: 目标点经纬度
        lat_range: (lat_min, lat_max) 网格覆盖纬度范围
        lon_range: (lon_min, lon_max) 网格覆盖经度范围
        grid_h, grid_w: 网格尺寸 (64, 80)
    
    Returns:
        (h_idx, w_idx): 网格索引
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    h_idx = int((lat_max - lat) / (lat_max - lat_min) * (grid_h - 1))
    w_idx = int((lon - lon_min) / (lon_max - lon_min) * (grid_w - 1))
    
    h_idx = max(0, min(grid_h - 1, h_idx))
    w_idx = max(0, min(grid_w - 1, w_idx))
    
    return h_idx, w_idx
```

---

## 4. 训练流程

### 4.1 阶段1：预训练主干 (已完成)

使用 2020-2025 网格数据训练，目标为网格风场预测。

```bash
python main.py
```

### 4.2 阶段2：训练校正头

```python
import torch
from openstl.models.mfwpn import MFWPN_Model

# 1. 加载预训练模型
checkpoint = torch.load('chkfile/mfwpn_pretrained.pth')

# 2. 初始化带校正头的模型
turbine_coords = [(32, 45), (33, 46), ...]  # 从 coords.json 读取

model = MFWPN_Model(
    turbine_coords=turbine_coords,
    enable_wind_correction=True,
    use_corrected_speed_for_power=True,
)

# 3. 加载预训练权重 (忽略新增模块)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# 4. 冻结主干参数
for name, param in model.named_parameters():
    if 'wind_correction_head' not in name and 'power_head' not in name:
        param.requires_grad = False

# 5. 只训练校正头和功率头
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

# 6. 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        x_raw, ele, gt_wind_speed, gt_power = batch
        
        outputs = model(x_raw, ele)
        
        # 风速校正损失
        loss_wind = F.mse_loss(outputs['corrected_speed'], gt_wind_speed)
        
        # 功率预测损失
        loss_power = F.mse_loss(outputs['power'].squeeze(-1), gt_power)
        
        # 总损失
        loss = loss_wind + 0.5 * loss_power
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4.3 损失函数

```python
# 风速校正损失
loss_wind = F.mse_loss(outputs['corrected_speed'], gt_wind_speed)

# 功率预测损失
loss_power = F.mse_loss(outputs['power'].squeeze(-1), gt_power)

# 可选：插值监督 (辅助)
loss_interp = F.mse_loss(outputs['interp_speed'], gt_wind_speed)

# 总损失
total_loss = loss_wind + λ_power * loss_power + λ_interp * loss_interp
```

---

## 5. 预测流程

### 5.1 推理代码

```python
import torch
import json
from openstl.models.mfwpn import MFWPN_Model

# 1. 加载坐标配置
with open('data/turbine_points/coords.json') as f:
    config = json.load(f)
turbine_coords = [(t['grid_h'], t['grid_w']) for t in config['turbines']]

# 2. 初始化模型
model = MFWPN_Model(
    turbine_coords=turbine_coords,
    enable_wind_correction=True,
    use_corrected_speed_for_power=True,
)

# 3. 加载训练好的权重
model.load_state_dict(torch.load('chkfile/mfwpn_with_correction.pth'))
model.eval()

# 4. 推理
with torch.no_grad():
    outputs = model(x_raw, ele)

# 5. 获取结果
grid_wind = outputs['wind']              # [B, T, 2, H, W] 网格风场
interp_speed = outputs['interp_speed']   # [B, T, N] 插值风速
corrected_speed = outputs['corrected_speed']  # [B, T, N] 校正后风速
power = outputs['power']                 # [B, T, N, 1] 功率预测
```

### 5.2 输出说明

| 输出键 | 形状 | 说明 |
|--------|------|------|
| `wind` | [B, T, 2, H, W] | 网格风场 (u, v 分量) |
| `interp_speed` | [B, T, N] | 双线性插值风速 (未校正) |
| `corrected_speed` | [B, T, N] | 校正后精确点风速 |
| `power` | [B, T, N, 1] | 各风机功率预测 |

---

## 6. 数据处理脚本

### 6.1 处理精确点原始数据

```python
#!/usr/bin/env python3
"""process_turbine_data.py: 处理精确点风机数据"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def process_turbine_excel(excel_path: str, output_dir: str, 
                          lat_range: tuple, lon_range: tuple,
                          grid_shape: tuple = (64, 80)):
    """
    处理风机 Excel 数据，转换为训练格式。
    
    Args:
        excel_path: 原始 Excel 文件路径
        output_dir: 输出目录
        lat_range: (lat_min, lat_max)
        lon_range: (lon_min, lon_max)
        grid_shape: 网格尺寸
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取数据
    df = pd.read_excel(excel_path)
    
    # 假设列名: timestamp, turbine_id, lat, lon, wind_speed, power
    turbines = df.groupby('turbine_id').first()[['lat', 'lon']].reset_index()
    
    # 生成坐标配置
    coords_config = {
        "feature_hw": list(grid_shape),
        "turbines": []
    }
    
    for _, row in turbines.iterrows():
        h_idx, w_idx = latlon_to_grid(
            row['lat'], row['lon'], lat_range, lon_range, *grid_shape
        )
        coords_config['turbines'].append({
            "id": row['turbine_id'],
            "lat": float(row['lat']),
            "lon": float(row['lon']),
            "grid_h": h_idx,
            "grid_w": w_idx
        })
    
    with open(output_dir / 'coords.json', 'w') as f:
        json.dump(coords_config, f, indent=2, ensure_ascii=False)
    
    # 按日期分组保存
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    for date, group in df.groupby('date'):
        date_str = date.strftime('%Y%m%d')
        
        # 透视表：时间 × 风机
        pivot_speed = group.pivot(index='timestamp', columns='turbine_id', values='wind_speed')
        pivot_power = group.pivot(index='timestamp', columns='turbine_id', values='power')
        
        np.savez(
            output_dir / f'turbine_{date_str}.npz',
            timestamps=pivot_speed.index.values,
            wind_speed=pivot_speed.values.astype(np.float32),
            power=pivot_power.values.astype(np.float32),
            turbine_ids=pivot_speed.columns.values
        )
    
    print(f"处理完成，保存到 {output_dir}")


def latlon_to_grid(lat, lon, lat_range, lon_range, grid_h, grid_w):
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    h_idx = int((lat_max - lat) / (lat_max - lat_min) * (grid_h - 1))
    w_idx = int((lon - lon_min) / (lon_max - lon_min) * (grid_w - 1))
    
    return max(0, min(grid_h - 1, h_idx)), max(0, min(grid_w - 1, w_idx))


if __name__ == '__main__':
    # 示例用法
    process_turbine_excel(
        excel_path='raw_data/turbine_data.xlsx',
        output_dir='data/turbine_points',
        lat_range=(38.0, 54.0),   # 东北地区纬度范围
        lon_range=(115.0, 135.0), # 东北地区经度范围
    )
```

---

## 7. 评估指标

```python
def evaluate_predictions(pred_speed, gt_speed, pred_power, gt_power):
    """评估预测结果"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    
    # 风速评估
    mae_speed = mean_absolute_error(gt_speed.flatten(), pred_speed.flatten())
    rmse_speed = np.sqrt(mean_squared_error(gt_speed.flatten(), pred_speed.flatten()))
    
    # 功率评估
    mae_power = mean_absolute_error(gt_power.flatten(), pred_power.flatten())
    rmse_power = np.sqrt(mean_squared_error(gt_power.flatten(), pred_power.flatten()))
    
    # 相对误差
    mape_speed = np.mean(np.abs(pred_speed - gt_speed) / (gt_speed + 1e-6)) * 100
    mape_power = np.mean(np.abs(pred_power - gt_power) / (gt_power + 1e-6)) * 100
    
    return {
        'wind_speed': {'MAE': mae_speed, 'RMSE': rmse_speed, 'MAPE': mape_speed},
        'power': {'MAE': mae_power, 'RMSE': rmse_power, 'MAPE': mape_power}
    }
```

---

## 8. 常见问题

### Q1: 校正头效果不好怎么办？
- 尝试增大 `wind_correction_hidden` (如 128)
- 增大 `roi_size` (如 7) 获取更多上下文
- 阶段3：解冻主干，用小学习率联合微调

### Q2: 功率预测误差大？
- 检查功率数据是否需要归一化
- 考虑添加额外输入特征（如温度、气压）
- 尝试更复杂的功率曲线建模

### Q3: 坐标转换不准确？
- 确认网格数据的经纬度范围与转换函数一致
- 考虑使用更精确的投影坐标系
