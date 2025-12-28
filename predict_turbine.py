#!/usr/bin/env python3
"""
预测脚本：使用训练好的模型进行精确点风速和功率预测
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from openstl.models.mfwpn import MFWPN_Model
from train_stage2 import Normalizer
from config import configs


class TurbinePredictor:
    """精确点风速和功率预测器"""
    
    def __init__(self, model_path: str, normalizer_path: str, turbine_coord: tuple):
        self.device = configs.device
        
        # 初始化模型
        self.model = MFWPN_Model(
            turbine_coords=[turbine_coord],
            enable_wind_correction=True,
            wind_correction_hidden=64,
            use_corrected_speed_for_power=True,
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {model_path}")
        
        # 加载归一化器
        self.normalizer = Normalizer().load(normalizer_path)
    
    @torch.no_grad()
    def predict(self, grid_input: np.ndarray, ele: np.ndarray) -> dict:
        """
        预测精确点风速和功率
        
        Args:
            grid_input: 网格输入数据 [T=24, C=4, H=64, W=80] 或 [B, T, C, H, W]
            ele: 高程数据 [H=64, W=80]
        
        Returns:
            dict: 包含预测结果
        """
        grid_tensor = torch.tensor(grid_input, dtype=torch.float32, device=self.device)
        ele_tensor = torch.tensor(ele, dtype=torch.float32, device=self.device)
        
        # 确保 batch 维度
        if grid_tensor.dim() == 4:
            grid_tensor = grid_tensor.unsqueeze(0)
        
        outputs = self.model(grid_tensor, ele_tensor)
        
        # 提取预测结果
        corrected_speed = outputs['corrected_speed'].cpu().numpy()  # [B, T, 1]
        power = outputs['power'].cpu().numpy()  # [B, T, 1, 1]
        interp_speed = outputs['interp_speed'].cpu().numpy()  # [B, T, 1]
        wind_field = outputs['wind'].cpu().numpy()  # [B, T, 2, H, W]
        
        # 反归一化
        corrected_speed_orig = self.normalizer.inverse_transform(
            corrected_speed.squeeze(-1), 'wind_speed'
        )
        power_orig = self.normalizer.inverse_transform(
            power.squeeze(-1).squeeze(-1), 'power'
        )
        
        return {
            'wind_field': wind_field,
            'interp_speed': interp_speed.squeeze(-1),
            'corrected_speed_norm': corrected_speed.squeeze(-1),
            'corrected_speed': corrected_speed_orig,  # m/s
            'power_norm': power.squeeze(-1).squeeze(-1),
            'power': power_orig,  # MW
        }


def main():
    parser = argparse.ArgumentParser(description='Turbine Wind Speed and Power Prediction')
    parser.add_argument('--model', type=str, default='chkfile/checkpoint_stage2.chk',
                        help='Path to trained model checkpoint')
    parser.add_argument('--normalizer', type=str, default='data/turbine_points/normalizer.npy',
                        help='Path to normalizer')
    parser.add_argument('--output', type=str, default='data/turbine_points/predictions.npz',
                        help='Output path for predictions')
    args = parser.parse_args()
    
    # 精确点坐标
    TURBINE_COORD = (32, 34)  # 辰阳风电场: 124.6138°E, 45.8470°N
    
    print("Loading predictor...")
    predictor = TurbinePredictor(
        model_path=args.model,
        normalizer_path=args.normalizer,
        turbine_coord=TURBINE_COORD
    )
    
    # 加载测试数据示例
    print("\nLoading test data...")
    
    # 加载网格数据
    uv_test = np.load("data/Northeast/uv100_test.npy").astype(np.float32)
    zt_test = np.load("data/Northeast/1000zt_test.npy").astype(np.float32)
    grid_test = np.concatenate((uv_test, zt_test), axis=1)
    print(f"Grid test data: {grid_test.shape}")
    
    # 高程
    ele = np.load('data/Northeast/DEM_northeast.npy').astype(np.float32)
    ele[ele < 0] = 0
    ele = (ele - ele.mean()) / ele.std()
    
    # 预测示例（取前 24 小时作为输入）
    print("\nPredicting...")
    sample_input = grid_test[:24]  # [24, 4, 64, 80]
    
    result = predictor.predict(sample_input, ele)
    
    print("\n=== Prediction Results ===")
    print(f"Wind field shape: {result['wind_field'].shape}")
    print(f"Corrected speed (m/s): {result['corrected_speed']}")
    print(f"Power (MW): {result['power']}")
    
    # 保存预测结果
    np.savez(args.output, **result)
    print(f"\nPredictions saved to {args.output}")


if __name__ == '__main__':
    main()
