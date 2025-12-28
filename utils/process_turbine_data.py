#!/usr/bin/env python3
"""
精确点数据处理脚本
- 数据来源：辰阳风电场测风塔数据 + 出力数据
- 清洗方案：A1(负功率置0) + C1(4条取平均) + D4(全缺失用前后平均)
- 输出：风速.npy, 功率.npy (1小时分辨率)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time


def load_and_filter_data(wind_file: str, power_file: str):
    """加载并筛选需要的数据行"""
    print("正在加载数据...")
    
    df_wind = pd.read_excel(wind_file)
    df_power = pd.read_excel(power_file)
    
    # 筛选轮毂高度风速
    wind_speed = df_wind[df_wind['类型'] == '轮毂高度风速'].copy()
    # 筛选场站实发功率
    power = df_power[df_power['类型'] == '场站实发功率'].copy()
    
    # 统一日期格式
    wind_speed['日期'] = pd.to_datetime(wind_speed['日期']).dt.date
    power['日期'] = pd.to_datetime(power['日期']).dt.date
    
    print(f"  轮毂风速: {len(wind_speed)} 天")
    print(f"  实发功率: {len(power)} 天")
    
    return wind_speed, power


def get_common_dates(wind_speed: pd.DataFrame, power: pd.DataFrame):
    """获取两个数据集的共同日期"""
    ws_dates = set(wind_speed['日期'])
    power_dates = set(power['日期'])
    common = sorted(ws_dates & power_dates)
    print(f"  共同日期: {len(common)} 天 ({common[0]} ~ {common[-1]})")
    return common


def extract_daily_values(df: pd.DataFrame, date, time_cols):
    """提取某天的所有 15 分钟数据"""
    row = df[df['日期'] == date]
    if len(row) == 0:
        return None
    return row[time_cols].values.flatten().astype(float)


def aggregate_to_hourly(values_15min: np.ndarray, method: str = 'mean'):
    """
    将 96 个 15 分钟数据聚合为 24 个小时数据
    
    Args:
        values_15min: shape (96,) 的 15 分钟数据
        method: 聚合方法
    
    Returns:
        hourly: shape (24,) 的小时数据
    """
    if values_15min is None:
        return np.full(24, np.nan)
    
    # reshape 为 (24, 4)，每行是一个小时的 4 个 15 分钟数据
    reshaped = values_15min.reshape(24, 4)
    
    # C1: 取平均（忽略 NaN）
    with np.errstate(all='ignore'):
        hourly = np.nanmean(reshaped, axis=1)
    
    return hourly


def clean_power(values: np.ndarray) -> np.ndarray:
    """A1: 负功率值置为 0"""
    cleaned = values.copy()
    cleaned[cleaned < 0] = 0
    return cleaned


def fill_missing_hours(hourly_data: np.ndarray) -> np.ndarray:
    """
    D4: 整小时全缺失用前后平均填充
    
    Args:
        hourly_data: shape (N_days, 24) 的小时数据
    
    Returns:
        filled: 填充后的数据
    """
    filled = hourly_data.copy()
    n_days, n_hours = filled.shape
    
    # 展平处理
    flat = filled.flatten()
    n_total = len(flat)
    
    nan_count_before = np.isnan(flat).sum()
    
    # 找到所有 NaN 位置
    nan_indices = np.where(np.isnan(flat))[0]
    
    for idx in nan_indices:
        # 找前一个非 NaN
        prev_val = np.nan
        for i in range(idx - 1, -1, -1):
            if not np.isnan(flat[i]):
                prev_val = flat[i]
                break
        
        # 找后一个非 NaN
        next_val = np.nan
        for i in range(idx + 1, n_total):
            if not np.isnan(flat[i]):
                next_val = flat[i]
                break
        
        # D4: 前后平均
        if not np.isnan(prev_val) and not np.isnan(next_val):
            flat[idx] = (prev_val + next_val) / 2
        elif not np.isnan(prev_val):
            flat[idx] = prev_val
        elif not np.isnan(next_val):
            flat[idx] = next_val
        # 如果前后都是 NaN，保持 NaN
    
    nan_count_after = np.isnan(flat).sum()
    print(f"  NaN 填充: {nan_count_before} -> {nan_count_after}")
    
    return flat.reshape(n_days, n_hours)


def process_data(wind_file: str, power_file: str, output_dir: str):
    """主处理流程"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载数据
    wind_speed_df, power_df = load_and_filter_data(wind_file, power_file)
    
    # 2. 获取共同日期
    common_dates = get_common_dates(wind_speed_df, power_df)
    
    # 3. 获取时间列（15 分钟间隔）
    time_cols_wind = [c for c in wind_speed_df.columns if c not in ['日期', '类型']]
    time_cols_power = [c for c in power_df.columns if c not in ['日期', '类型']]
    
    print(f"\n时间列数量: 风速={len(time_cols_wind)}, 功率={len(time_cols_power)}")
    
    # 4. 逐天处理
    n_days = len(common_dates)
    wind_speed_hourly = np.zeros((n_days, 24), dtype=np.float32)
    power_hourly = np.zeros((n_days, 24), dtype=np.float32)
    
    print("\n正在处理每日数据...")
    for i, date in enumerate(common_dates):
        # 提取 15 分钟数据
        ws_15min = extract_daily_values(wind_speed_df, date, time_cols_wind)
        power_15min = extract_daily_values(power_df, date, time_cols_power)
        
        # A1: 功率负值置 0
        if power_15min is not None:
            power_15min = clean_power(power_15min)
        
        # C1: 聚合为小时数据
        wind_speed_hourly[i] = aggregate_to_hourly(ws_15min)
        power_hourly[i] = aggregate_to_hourly(power_15min)
    
    print(f"  处理完成: {n_days} 天 x 24 小时")
    
    # 5. D4: 填充缺失的整小时数据
    print("\n正在填充缺失数据...")
    print("风速:")
    wind_speed_hourly = fill_missing_hours(wind_speed_hourly)
    print("功率:")
    power_hourly = fill_missing_hours(power_hourly)
    
    # 6. 统计信息
    print("\n=== 最终数据统计 ===")
    print(f"形状: ({n_days}, 24)")
    print(f"风速: min={wind_speed_hourly.min():.2f}, max={wind_speed_hourly.max():.2f}, "
          f"mean={wind_speed_hourly.mean():.2f} m/s")
    print(f"功率: min={power_hourly.min():.2f}, max={power_hourly.max():.2f}, "
          f"mean={power_hourly.mean():.2f} MW")
    print(f"风速 NaN: {np.isnan(wind_speed_hourly).sum()}")
    print(f"功率 NaN: {np.isnan(power_hourly).sum()}")
    
    # 7. 保存
    dates_array = np.array([str(d) for d in common_dates])
    
    np.save(output_dir / 'turbine_wind_speed.npy', wind_speed_hourly)
    np.save(output_dir / 'turbine_power.npy', power_hourly)
    np.save(output_dir / 'turbine_dates.npy', dates_array)
    
    print(f"\n=== 保存完成 ===")
    print(f"  {output_dir / 'turbine_wind_speed.npy'}")
    print(f"  {output_dir / 'turbine_power.npy'}")
    print(f"  {output_dir / 'turbine_dates.npy'}")
    
    return wind_speed_hourly, power_hourly, dates_array


if __name__ == '__main__':
    # 输入文件
    wind_file = '/home/xiao/Desktop/WFMPN/东北电网数据/黑龙江/辰阳风电场_测风塔数据.xlsx'
    power_file = '/home/xiao/Desktop/WFMPN/东北电网数据/黑龙江/辰阳风电场_出力数据.xlsx'
    
    # 输出目录
    output_dir = '/home/xiao/Desktop/WFMPN/WFMPN/data/turbine_points'
    
    # 处理
    wind_speed, power, dates = process_data(wind_file, power_file, output_dir)
