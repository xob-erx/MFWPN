# Joint Wind Field and Turbine Power Forecasting for Offshore Wind Farms

## Abstract
We propose a joint forecasting framework that couples high-resolution wind field prediction with turbine-level power regression. This draft summarizes the core design components that will be elaborated and validated in the full paper.

## 1. Precise Location Prediction
To recover wind speed at arbitrary turbine sites beyond the discrete grid, the decoder output \(\hat{\mathbf{Y}}\in\mathbb{R}^{B\times T_{out}\times 2\times H\times W}\) is post-processed with differentiable bilinear interpolation. Turbine coordinates expressed in the feature grid are normalized to \([-1,1]\) and fed to a sampling kernel so that both wind components are interpolated without aliasing. The interpolated vectors are converted to speed magnitudes that serve both as standalone evaluation targets and auxiliary regressors for downstream modules. This mechanism allows the model to support out-of-grid turbines while maintaining gradients for end-to-end training.

## 2. Temporal Alignment
Meteorological drivers (e.g., `uv100`, `1000zt`) follow an hourly cadence and may span multi-year archives, whereas supervisory turbine power measurements can present shorter coverage and finer (15 min) resolution. We align these streams by: (1) aggregating power readings to the hourly cadence via mean pooling, (2) synchronizing start and end timestamps with the atmospheric records, and (3) constructing shared sliding-window indices. Windows with missing power entries are flagged by a binary mask so that the loss function can ignore invalid steps. This alignment ensures that every supervision pair reflects the same physical horizon and avoids bias from partial coverage.

## 3. Turbine Power Forecasting
The turbine power head receives two complementary signals: ROI-aggregated spatiotemporal features extracted from the shared decoder representation and the interpolated wind speed at each turbine location. For every turbine, a \(K\times K\) region centered on its grid coordinate is cropped, processed by lightweight convolutions, and collapsed via adaptive pooling. The pooled descriptor is concatenated with the scalar wind speed feature and passed through an MLP to produce the forecasted power sequence \(\hat{\mathbf{P}}\in\mathbb{R}^{B\times T_{out}\times N_p\times 1}\). This design encourages the head to leverage both local flow context and precise wind magnitude cues.

## 4. Dual-Loss Optimization
Training optimizes the sum of a wind field loss \(\mathcal{L}_{wind}\) (combining RMSE, directional penalties, and structure-aware terms) and a turbine power loss \(\mathcal{L}_{power}\) defined as masked MSE over valid timestamps. We explore two weighting strategies: (a) fixed coefficients tuned on validation data, and (b) homoscedastic uncertainty weighting that learns per-task log-variances to balance gradients adaptively. The overall objective \(\mathcal{L}=w_{wind}\mathcal{L}_{wind}+w_{power}\mathcal{L}_{power}\) (or its uncertainty-weighted variant) encourages the network to jointly refine flow reconstruction and energy yield prediction.

## 5. Next Steps
Future iterations will elaborate on dataset statistics, architectural ablations, training schedules, and evaluation metrics. Empirical results, sensitivity analyses, and case studies will be added once experiments are completed.
