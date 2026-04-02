# A machine learning model for hub-height short-term wind speed prediction

*2025.01*

## Abstract

Accurate short-term wind speed prediction is crucial for maintaining the safe, stable, and efficient operation of wind power systems. We propose a multivariate meteorological data fusion wind prediction network (MFWPN) to study fine-grid vector wind speed prediction, taking Northeast China as an example.

![Graphabstract](data/graphabstract.jpg)


## Installation

```
conda create -n wpn python=3.8
conda activate wpn
git clone https://github.com/Zhang-zongwei/MF-WPN.git
cd MF-WPN
pip install -r requirements.txt
```

## Overview

- `data/:` contains a test set for the northeast region of the manuscript, which can be downloaded via the link .
- `openstl/models/mfwpn.py:` contains the network architecture of this MF-WPN.
- `openstl/modules/:` contains partial modules of the MF-WPN.
- `utils/:` contains data processing files and loss calculations..
- `chkfile/:` contains weights for predicting 24-hour wind speeds in the Northeast region, which can be downloaded via the link.
- `result/:` contains predicted wind speed results and evaluation methods.
- `config.py：`  training configs for the MF-WPN.
- `main.py:` Train the MF-WPN.
- `test.py` Test the MF-WPN.
- `docs/mfwpn_structure.md`: high-level structure diagram of the joint wind and turbine power prediction model.

## Data preparation
The data used in this study and its processing have been described in detail in the manuscript. To facilitate the testing, we have prepared the [MFWPN weights](https://drive.google.com/file/d/1YrJP1sCWUcsHcYdNL_sWFbkuS4WfaeJf/view?usp=sharing) , [test dataset](https://drive.google.com/drive/folders/1qQMV8xBRDI5Vg9pxigLAJNEOtNC4O87x?usp=sharing) and [train_val_dataset](https://drive.google.com/drive/folders/1ppxlPq2PABTpUfXTWfQZ3ZDCXvmXfuqk?usp=sharing).

If you have raw GRIB files (e.g. in `data/2020-2025`), generate stage1/stage2 grid npy files with:
```
python scripts/process_grib_data.py --data-dir data/2020-2025 --output-dir data/Northeast --train-end "2024-12-31T23:00:00" --drop-lat 38.0 --drop-lon 136.0
```

## Train
After the data is ready, use the following commands to start training.

### Stage 1 (grid wind field pretraining)
```
python main.py
```

### Stage 2 (turbine-point wind correction + power prediction)
```
python train_stage2.py
```

#### Stage 2 detailed guide

Stage 2 expects a turbine data directory containing:

- `turbine_wind_speed.npy` with shape `[N_days, 24]`
- `turbine_dates.npy` with shape `[N_days]`, date format `YYYY-MM-DD`
- Optional `turbine_power.npy` with shape `[N_days, 24]`
- Optional `turbine_meta.json` containing `latitude`, `longitude`, and/or `grid_coord`

You can build this format from hourly CSV + farm coordinates CSV using:

```bash
python scripts/prepare_stage2_turbine_data.py \
  --wind-csv "/path/to/风速_小时均值.csv" \
  --coord-csv "/path/to/风电场经纬度.csv" \
  --plant-name "龙源八虎山" \
  --output-dir "data/turbine_points/longyuan_bahushan" \
  --fill-power nan
```

Run Stage 2 on GPU (recommended environment: `wpn310`):

```bash
conda run -n wpn310 python train_stage2.py \
  --turbine-dir data/turbine_points/longyuan_bahushan \
  --turbine-lat 42.420278 \
  --turbine-lon 123.126111 \
  --coord-rounding round \
  --allow-missing-power \
  --align-hour-offset -8 \
  --window-start-hour 0 \
  --test-start-date 2025-11-01 \
  --result-dir result/exp \
  --device cuda
```

Use fixed ratio split (recommended for controlled experiments):

```bash
conda run -n wpn310 python train_stage2.py \
  --turbine-dir data/turbine_points/longyuan_bahushan \
  --turbine-lat 42.420278 \
  --turbine-lon 123.126111 \
  --coord-rounding round \
  --allow-missing-power \
  --split-mode ratio \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --test-ratio 0.2 \
  --ratio-split-strategy chronological \
  --align-hour-offset -8 \
  --window-start-hour 0 \
  --result-dir result/exp \
  --device cuda
```

Recommended quick comparison (to find best setup for your site):

```bash
# A) Baseline (no timezone shift)
conda run -n wpn310 python train_stage2.py --turbine-dir data/turbine_points/longyuan_bahushan --split-mode ratio --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 --ratio-split-strategy chronological --allow-missing-power --align-hour-offset 0 --window-start-hour 0 --result-dir result/exp --device cuda

# B) UTC->北京时间日对齐（推荐）
conda run -n wpn310 python train_stage2.py --turbine-dir data/turbine_points/longyuan_bahushan --split-mode ratio --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 --ratio-split-strategy chronological --allow-missing-power --align-hour-offset -8 --window-start-hour 0 --result-dir result/exp --device cuda

# C) 对齐后窗口从08:00开始
conda run -n wpn310 python train_stage2.py --turbine-dir data/turbine_points/longyuan_bahushan --split-mode ratio --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 --ratio-split-strategy chronological --allow-missing-power --align-hour-offset -8 --window-start-hour 8 --result-dir result/exp --device cuda
```

Notes:

- If power labels are all missing (`NaN`) and `--allow-missing-power` is enabled, Stage 2 automatically falls back to wind-only training for that run.
- Split options:
  - `--split-mode date`: split by `--test-start-date` (default behavior)
  - `--split-mode ratio`: split by `--train-ratio/--val-ratio/--test-ratio`
  - For time-series tasks, `--ratio-split-strategy chronological` is recommended.
- Output is organized by baseline directory name:
  - If `--turbine-dir data/turbine_points/longyuan_bahushan`
  - Then outputs are saved to `result/exp/longyuan_bahushan/`
- `--turbine-lat/--turbine-lon` are converted to fractional grid coordinates for bilinear interpolation, while integer grid indices are still used for local ROI extraction.
- If you want pure wind-only mode regardless of power files, add `--wind-only`.

#### Stage 2 参数说明（中文）

- `--align-hour-offset`：**对齐层参数**，用于在 `train_stage2.py` 中将 turbine 日期与网格时间轴进行小时级平移后再取样。
  - 默认 `0`：不平移（按 GRIB 原始时间轴对齐，通常是 UTC）。
  - 若 GRIB 是 UTC、turbine 日期按北京时间（UTC+8）统计，通常应设为 `-8`（北京时间 00:00 对应 UTC 前一天 16:00）。

- `--window-start-hour`：**切窗层参数**，用于控制“在已完成对齐的时间轴上”，每个 24 小时训练/预测窗口从几点开始。
  - 默认 `0`：窗口从 `00:00` 开始。
  - 设为 `8`：窗口从 `08:00` 开始（窗口长度仍为 24 小时）。

- 两者区别：
  - `align-hour-offset` 解决的是 **GRIB 与 turbine 的时间轴对齐问题**；
  - `window-start-hour` 解决的是 **样本切窗起点问题**。

## Test
We provide the test model weights and test dataset, which can be tested using the following commands after downloading:
```
python test.py
```

Note that the predictions are obtained as npy files containing u, v variables, which need to be converted to wind speed results using: 
```
cd result
python uv_to_wind.py
```
After that, we can obtain the wind speed prediction evaluation result by:
```
python evaluate.py
```
## Acknowledgments

Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL),[Restormer](https://github.com/swz30/Restormer),[CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse). We sincerely appreciate for their contributions.

If you have any questions or suggestions, please do not hesitate to contact us at [zhangzongwei@stu.hit.edu.cn].
