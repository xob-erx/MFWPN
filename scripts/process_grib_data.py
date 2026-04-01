import argparse
import os

import numpy as np
import xarray as xr


DEFAULT_DATA_DIR = "../data/2020-2025"
DEFAULT_OUTPUT_DIR = "../data/Northeast"
DEFAULT_TRAIN_END = "2024-12-31T23:00:00"
DEFAULT_DROP_LAT = 38.0
DEFAULT_DROP_LON = 136.0


def open_grib(path: str) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    return lat_name, lon_name


def drop_lat_lon(ds: xr.Dataset, drop_lat: float, drop_lon: float) -> xr.Dataset:
    lat_name, lon_name = coord_names(ds)
    lat_values = np.asarray(ds[lat_name].values, dtype=np.float64)
    lon_values = np.asarray(ds[lon_name].values, dtype=np.float64)

    lat_idx = int(np.argmin(np.abs(lat_values - drop_lat)))
    lon_idx = int(np.argmin(np.abs(lon_values - drop_lon)))

    if not np.isclose(lat_values[lat_idx], drop_lat, atol=1e-6):
        raise ValueError(f"Latitude {drop_lat} not found in coordinate list")
    if not np.isclose(lon_values[lon_idx], drop_lon, atol=1e-6):
        raise ValueError(f"Longitude {drop_lon} not found in coordinate list")

    keep_lat = np.ones(lat_values.shape[0], dtype=bool)
    keep_lon = np.ones(lon_values.shape[0], dtype=bool)
    keep_lat[lat_idx] = False
    keep_lon[lon_idx] = False

    cropped = ds.isel({lat_name: keep_lat, lon_name: keep_lon})

    out_lat = np.asarray(cropped[lat_name].values)
    out_lon = np.asarray(cropped[lon_name].values)
    if out_lat.shape[0] != 64 or out_lon.shape[0] != 80:
        raise ValueError(
            f"Unexpected cropped shape lat/lon=({out_lat.shape[0]}, {out_lon.shape[0]}), expected (64, 80)"
        )

    if np.any(np.isclose(out_lat, drop_lat)):
        raise ValueError(f"Latitude {drop_lat} still present after crop")
    if np.any(np.isclose(out_lon, drop_lon)):
        raise ValueError(f"Longitude {drop_lon} still present after crop")

    return cropped


def load_grib_data(data_dir: str) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    uv100 = open_grib(os.path.join(data_dir, "uv100.grib"))
    geo = open_grib(os.path.join(data_dir, "geo.grib"))
    temp = open_grib(os.path.join(data_dir, "temp.grib"))

    print(f"uv100 time: {uv100.time.values[0]} ~ {uv100.time.values[-1]}, count={len(uv100.time)}")
    print(f"geo   time: {geo.time.values[0]} ~ {geo.time.values[-1]}, count={len(geo.time)}")
    print(f"temp  time: {temp.time.values[0]} ~ {temp.time.values[-1]}, count={len(temp.time)}")

    lat_name, lon_name = coord_names(uv100)
    lat_vals = uv100[lat_name].values
    lon_vals = uv100[lon_name].values
    print(f"raw grid lat/lon: ({len(lat_vals)}, {len(lon_vals)})")
    print(f"raw latitude range: {float(lat_vals[0])} -> {float(lat_vals[-1])}")
    print(f"raw longitude range: {float(lon_vals[0])} -> {float(lon_vals[-1])}")

    return uv100, geo, temp


def align_time(uv100: xr.Dataset, geo: xr.Dataset, temp: xr.Dataset):
    common_times = sorted(set(uv100.time.values) & set(geo.time.values) & set(temp.time.values))
    if not common_times:
        raise ValueError("No common timestamps among uv100/geo/temp")

    uv100_aligned = uv100.sel(time=common_times)
    geo_aligned = geo.sel(time=common_times)
    temp_aligned = temp.sel(time=common_times)

    print(f"common time count: {len(common_times)}")
    print(f"common time range: {common_times[0]} ~ {common_times[-1]}")

    return uv100_aligned, geo_aligned, temp_aligned, common_times


def normalize_data(train_data, test_data, name: str):
    mean = train_data.mean(axis=(0, 2, 3), keepdims=True)
    std = train_data.std(axis=(0, 2, 3), keepdims=True)

    train_normalized = (train_data - mean) / (std + 1e-8)
    test_normalized = (test_data - mean) / (std + 1e-8)

    print(
        f"{name} channel0 mean/std={mean[0,0,0,0]:.6f}/{std[0,0,0,0]:.6f}, "
        f"channel1 mean/std={mean[0,1,0,0]:.6f}/{std[0,1,0,0]:.6f}"
    )
    return train_normalized, test_normalized, mean, std


def split_train_test_by_time(data, times, train_end: str):
    if len(data) != len(times):
        raise ValueError(f"Length mismatch: data={len(data)} times={len(times)}")

    train_end_ts = np.datetime64(train_end)
    times_np = np.asarray(times, dtype="datetime64[ns]")

    train_mask = times_np <= train_end_ts
    test_mask = times_np > train_end_ts

    train_data = data[train_mask]
    test_data = data[test_mask]

    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError(
            f"Invalid split by train_end={train_end}: train={len(train_data)}, test={len(test_data)}"
        )

    print(f"split by time train_end={train_end}: train={len(train_data)}, test={len(test_data)}")
    return train_data, test_data


def process_uv100(uv100_aligned: xr.Dataset, drop_lat: float, drop_lon: float):
    cropped = drop_lat_lon(uv100_aligned, drop_lat, drop_lon)
    u100 = cropped["u100"].values
    v100 = cropped["v100"].values
    data = np.stack([u100, v100], axis=1).astype(np.float32)
    print(f"uv100 processed shape: {data.shape}")
    return data


def process_1000zt(geo_aligned: xr.Dataset, temp_aligned: xr.Dataset, drop_lat: float, drop_lon: float):
    geo_cropped = drop_lat_lon(geo_aligned, drop_lat, drop_lon)
    temp_cropped = drop_lat_lon(temp_aligned, drop_lat, drop_lon)
    z = geo_cropped["z"].values
    t = temp_cropped["t"].values
    data = np.stack([z, t], axis=1).astype(np.float32)
    print(f"1000zt processed shape: {data.shape}")
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-end", type=str, default=DEFAULT_TRAIN_END)
    parser.add_argument("--drop-lat", type=float, default=DEFAULT_DROP_LAT)
    parser.add_argument("--drop-lon", type=float, default=DEFAULT_DROP_LON)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    uv100, geo, temp = load_grib_data(args.data_dir)
    uv100_aligned, geo_aligned, temp_aligned, common_times = align_time(uv100, geo, temp)

    uv100_data = process_uv100(uv100_aligned, args.drop_lat, args.drop_lon)
    zt_data = process_1000zt(geo_aligned, temp_aligned, args.drop_lat, args.drop_lon)

    if len(uv100_data) != len(common_times) or len(zt_data) != len(common_times):
        raise ValueError("Processed array time length mismatch")

    uv100_train, uv100_test = split_train_test_by_time(uv100_data, common_times, args.train_end)
    zt_train, zt_test = split_train_test_by_time(zt_data, common_times, args.train_end)

    uv100_train_norm, uv100_test_norm, uv_mean, uv_std = normalize_data(uv100_train, uv100_test, "uv100")
    zt_train_norm, zt_test_norm, zt_mean, zt_std = normalize_data(zt_train, zt_test, "1000zt")

    np.save(os.path.join(args.output_dir, "uv100_train.npy"), uv100_train_norm)
    np.save(os.path.join(args.output_dir, "uv100_test.npy"), uv100_test_norm)
    np.save(os.path.join(args.output_dir, "1000zt_train.npy"), zt_train_norm)
    np.save(os.path.join(args.output_dir, "1000zt_test.npy"), zt_test_norm)

    np.savez(
        os.path.join(args.output_dir, "normalization_params.npz"),
        uv100_mean=uv_mean.squeeze(),
        uv100_std=uv_std.squeeze(),
        zt_mean=zt_mean.squeeze(),
        zt_std=zt_std.squeeze(),
    )

    print("Saved files:")
    print(f"  {os.path.join(args.output_dir, 'uv100_train.npy')} {uv100_train_norm.shape} {uv100_train_norm.dtype}")
    print(f"  {os.path.join(args.output_dir, 'uv100_test.npy')} {uv100_test_norm.shape} {uv100_test_norm.dtype}")
    print(f"  {os.path.join(args.output_dir, '1000zt_train.npy')} {zt_train_norm.shape} {zt_train_norm.dtype}")
    print(f"  {os.path.join(args.output_dir, '1000zt_test.npy')} {zt_test_norm.shape} {zt_test_norm.dtype}")
    print(f"  {os.path.join(args.output_dir, 'normalization_params.npz')}")


if __name__ == "__main__":
    main()
