#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Convert hourly wind CSV to stage2 turbine_points format')
    parser.add_argument('--wind-csv', type=str, required=True, help='Path to hourly wind CSV')
    parser.add_argument('--coord-csv', type=str, required=True, help='Path to farm longitude/latitude CSV')
    parser.add_argument('--plant-name', type=str, required=True, help='Plant name used in both CSVs')
    parser.add_argument('--output-dir', type=str, required=True, help='Output turbine_points subdirectory')
    parser.add_argument('--lon-min', type=float, default=116.0)
    parser.add_argument('--lon-max', type=float, default=135.75)
    parser.add_argument('--lat-min', type=float, default=38.25)
    parser.add_argument('--lat-max', type=float, default=54.0)
    parser.add_argument('--grid-h', type=int, default=64)
    parser.add_argument('--grid-w', type=int, default=80)
    parser.add_argument('--fill-power', choices=['nan', 'zero'], default='nan',
                        help='How to create placeholder power labels when power file is unavailable')
    return parser.parse_args()


def dms_to_decimal(text: str) -> float:
    s = str(text).strip().replace(' ', '')
    if '°' not in s:
        return float(s)

    deg_part, rest = s.split('°', 1)
    degree = float(deg_part)

    minutes = 0.0
    seconds = 0.0
    if '′' in rest:
        min_part, rest = rest.split('′', 1)
        minutes = float(min_part) if min_part else 0.0
    if '″' in rest:
        sec_part, _ = rest.split('″', 1)
        seconds = float(sec_part) if sec_part else 0.0

    sign = -1.0 if degree < 0 else 1.0
    degree = abs(degree)
    return sign * (degree + minutes / 60.0 + seconds / 3600.0)


def latlon_to_grid(lat: float, lon: float, lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                   grid_h: int, grid_w: int) -> Tuple[int, int]:
    h = int(round((lat_max - lat) / (lat_max - lat_min) * (grid_h - 1)))
    w = int(round((lon - lon_min) / (lon_max - lon_min) * (grid_w - 1)))
    h = max(0, min(grid_h - 1, h))
    w = max(0, min(grid_w - 1, w))
    return h, w


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wind_df = pd.read_csv(args.wind_csv, encoding='utf-8-sig')
    wind_df = wind_df[wind_df['电厂名称'] == args.plant_name].copy()
    if wind_df.empty:
        raise ValueError(f'No rows found for plant: {args.plant_name}')

    hour_cols = [f'{i}:00' for i in range(24)]
    missing_cols = [c for c in hour_cols if c not in wind_df.columns]
    if missing_cols:
        raise ValueError(f'Missing hour columns: {missing_cols}')

    parsed_dates = pd.to_datetime(wind_df['日期'], errors='coerce')
    wind_df = wind_df.assign(日期=parsed_dates).dropna(subset=['日期'])
    wind_df['日期'] = wind_df['日期'].map(lambda x: x.strftime('%Y-%m-%d'))
    wind_df = wind_df.sort_values(by='日期').drop_duplicates(subset=['日期'], keep='first')

    wind_speed = np.asarray(wind_df[hour_cols].to_numpy(dtype=np.float32), dtype=np.float32)
    dates = np.array(wind_df['日期'].astype(str).tolist(), dtype='<U10')

    if args.fill_power == 'zero':
        power = np.asarray(np.zeros_like(wind_speed, dtype=np.float32), dtype=np.float32)
    else:
        power = np.asarray(np.full(wind_speed.shape, np.nan, dtype=np.float32), dtype=np.float32)

    coord_df = pd.read_csv(args.coord_csv, encoding='utf-8-sig')
    row = coord_df[coord_df['电厂名称'] == args.plant_name]
    if row.empty:
        raise ValueError(f'No coordinate row found for plant: {args.plant_name}')
    row = row.iloc[0]

    lon = dms_to_decimal(row['经度'])
    lat = dms_to_decimal(row['纬度'])
    grid_h_idx, grid_w_idx = latlon_to_grid(
        lat=lat,
        lon=lon,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
    )

    np.save(out_dir / 'turbine_wind_speed.npy', wind_speed)
    np.save(out_dir / 'turbine_power.npy', power)
    np.save(out_dir / 'turbine_dates.npy', dates)

    metadata = {
        'plant_name': args.plant_name,
        'longitude': lon,
        'latitude': lat,
        'grid_coord': [int(grid_h_idx), int(grid_w_idx)],
        'grid_shape': [args.grid_h, args.grid_w],
        'grid_extent': {
            'lat_min': args.lat_min,
            'lat_max': args.lat_max,
            'lon_min': args.lon_min,
            'lon_max': args.lon_max,
        },
        'power_fill': args.fill_power,
        'n_days': int(len(dates)),
    }

    with open(out_dir / 'turbine_meta.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f'Saved to: {out_dir}')
    print(f'  turbine_wind_speed.npy: {wind_speed.shape} {wind_speed.dtype}')
    print(f'  turbine_power.npy: {power.shape} {power.dtype} (fill={args.fill_power})')
    print(f'  turbine_dates.npy: {dates.shape}')
    print(f'  turbine_meta.json: coord=({grid_h_idx}, {grid_w_idx}), lat/lon=({lat:.6f}, {lon:.6f})')


if __name__ == '__main__':
    main()
