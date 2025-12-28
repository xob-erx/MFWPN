"""
Data processing script: Convert grib files to npy format
- uv100.grib → uv100_train.npy, uv100_test.npy
- geo.grib + temp.grib → 1000zt_train.npy, 1000zt_test.npy

Target shape:
- Train: (43824, 2, 64, 80)
- Test: (8760, 2, 64, 80)
"""

import numpy as np
import xarray as xr
import os

# Configuration
DATA_DIR = "data"
OUTPUT_DIR = "data/processed"
TRAIN_SIZE = 43824  # ~5 years
TEST_SIZE = 7324    # remaining data (~305 days)
TARGET_LAT = 64
TARGET_LON = 80

def load_grib_data():
    """Load all grib files"""
    print("Loading grib files...")
    
    uv100 = xr.open_dataset(os.path.join(DATA_DIR, "uv100.grib"), engine="cfgrib")
    geo = xr.open_dataset(os.path.join(DATA_DIR, "geo.grib"), engine="cfgrib")
    temp = xr.open_dataset(os.path.join(DATA_DIR, "temp.grib"), engine="cfgrib")
    
    print(f"uv100 time range: {uv100.time.values[0]} ~ {uv100.time.values[-1]}, count: {len(uv100.time)}")
    print(f"geo   time range: {geo.time.values[0]} ~ {geo.time.values[-1]}, count: {len(geo.time)}")
    print(f"temp  time range: {temp.time.values[0]} ~ {temp.time.values[-1]}, count: {len(temp.time)}")
    
    return uv100, geo, temp

def align_time(uv100, geo, temp):
    """Align datasets on common time points"""
    print("\nAligning time dimensions...")
    
    # Find common time intersection
    time_uv100 = set(uv100.time.values)
    time_geo = set(geo.time.values)
    time_temp = set(temp.time.values)
    
    common_times = sorted(time_uv100 & time_geo & time_temp)
    print(f"Common time points: {len(common_times)}")
    print(f"Common time range: {common_times[0]} ~ {common_times[-1]}")
    
    # Select common times
    uv100_aligned = uv100.sel(time=common_times)
    geo_aligned = geo.sel(time=common_times)
    temp_aligned = temp.sel(time=common_times)
    
    return uv100_aligned, geo_aligned, temp_aligned, common_times

def crop_spatial(data, target_lat=TARGET_LAT, target_lon=TARGET_LON):
    """Crop spatial dimensions to target size (keep north and west)"""
    # Original: 65 lat, 81 lon -> Target: 64 lat, 80 lon
    # Crop last row and last column
    return data.isel(latitude=slice(0, target_lat), longitude=slice(0, target_lon))

def process_uv100(uv100_aligned):
    """Process uv100 data: combine u100 and v100 into 2 channels"""
    print("\nProcessing uv100 data...")
    
    # Extract and crop
    u100 = crop_spatial(uv100_aligned['u100']).values  # (time, lat, lon)
    v100 = crop_spatial(uv100_aligned['v100']).values
    
    print(f"u100 shape after crop: {u100.shape}")
    print(f"v100 shape after crop: {v100.shape}")
    
    # Stack to (time, 2, lat, lon)
    uv100_data = np.stack([u100, v100], axis=1).astype(np.float64)
    print(f"uv100 combined shape: {uv100_data.shape}")
    
    return uv100_data

def process_1000zt(geo_aligned, temp_aligned):
    """Process geo and temp data: combine z and t into 2 channels"""
    print("\nProcessing 1000zt data...")
    
    # Extract and crop
    z = crop_spatial(geo_aligned['z']).values   # (time, lat, lon)
    t = crop_spatial(temp_aligned['t']).values
    
    print(f"z shape after crop: {z.shape}")
    print(f"t shape after crop: {t.shape}")
    
    # Stack to (time, 2, lat, lon) - z first, then t
    zt_data = np.stack([z, t], axis=1).astype(np.float64)
    print(f"1000zt combined shape: {zt_data.shape}")
    
    return zt_data

def normalize_data(train_data, test_data, name):
    """Z-score normalization using train statistics"""
    print(f"\nNormalizing {name} data...")
    
    # Calculate mean and std from training data (per channel)
    # Shape: (time, 2, lat, lon)
    mean = train_data.mean(axis=(0, 2, 3), keepdims=True)  # (1, 2, 1, 1)
    std = train_data.std(axis=(0, 2, 3), keepdims=True)
    
    print(f"  Channel 0 - mean: {mean[0,0,0,0]:.4f}, std: {std[0,0,0,0]:.4f}")
    print(f"  Channel 1 - mean: {mean[0,1,0,0]:.4f}, std: {std[0,1,0,0]:.4f}")
    
    # Normalize
    train_normalized = (train_data - mean) / std
    test_normalized = (test_data - mean) / std
    
    # Verify normalization
    print(f"  Train normalized - mean: {train_normalized.mean():.6f}, std: {train_normalized.std():.6f}")
    print(f"  Test normalized  - mean: {test_normalized.mean():.6f}, std: {test_normalized.std():.6f}")
    
    return train_normalized, test_normalized, mean, std

def split_train_test(data, train_size=TRAIN_SIZE, test_size=TEST_SIZE):
    """Split data into train and test sets"""
    total_needed = train_size + test_size
    
    if len(data) < total_needed:
        raise ValueError(f"Not enough data: have {len(data)}, need {total_needed}")
    
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    
    return train_data, test_data

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    uv100, geo, temp = load_grib_data()
    
    # Align time
    uv100_aligned, geo_aligned, temp_aligned, common_times = align_time(uv100, geo, temp)
    
    # Check if we have enough data
    total_needed = TRAIN_SIZE + TEST_SIZE
    if len(common_times) < total_needed:
        print(f"\nWARNING: Not enough common time points!")
        print(f"  Have: {len(common_times)}, Need: {total_needed}")
        return
    
    print(f"\nTotal common time points: {len(common_times)}")
    print(f"Will use: {total_needed} (train: {TRAIN_SIZE}, test: {TEST_SIZE})")
    
    # Process uv100
    uv100_data = process_uv100(uv100_aligned)
    uv100_train, uv100_test = split_train_test(uv100_data)
    uv100_train_norm, uv100_test_norm, uv_mean, uv_std = normalize_data(
        uv100_train, uv100_test, "uv100"
    )
    
    # Process 1000zt
    zt_data = process_1000zt(geo_aligned, temp_aligned)
    zt_train, zt_test = split_train_test(zt_data)
    zt_train_norm, zt_test_norm, zt_mean, zt_std = normalize_data(
        zt_train, zt_test, "1000zt"
    )
    
    # Save files
    print("\nSaving files...")
    
    np.save(os.path.join(OUTPUT_DIR, "uv100_train.npy"), uv100_train_norm)
    np.save(os.path.join(OUTPUT_DIR, "uv100_test.npy"), uv100_test_norm)
    np.save(os.path.join(OUTPUT_DIR, "1000zt_train.npy"), zt_train_norm)
    np.save(os.path.join(OUTPUT_DIR, "1000zt_test.npy"), zt_test_norm)
    
    print(f"\nFiles saved to {OUTPUT_DIR}/")
    print(f"  uv100_train.npy: {uv100_train_norm.shape}, dtype: {uv100_train_norm.dtype}")
    print(f"  uv100_test.npy:  {uv100_test_norm.shape}, dtype: {uv100_test_norm.dtype}")
    print(f"  1000zt_train.npy: {zt_train_norm.shape}, dtype: {zt_train_norm.dtype}")
    print(f"  1000zt_test.npy:  {zt_test_norm.shape}, dtype: {zt_test_norm.dtype}")
    
    # Save normalization parameters for future use
    norm_params = {
        'uv100_mean': uv_mean.squeeze(),
        'uv100_std': uv_std.squeeze(),
        'zt_mean': zt_mean.squeeze(),
        'zt_std': zt_std.squeeze()
    }
    np.savez(os.path.join(OUTPUT_DIR, "normalization_params.npz"), **norm_params)
    print(f"  normalization_params.npz: saved")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
