# MFWPN 模型结构概览

下图以模块级别展示了当前代码库中的多变量风场预测网络（MFWPN）以及新增的风机功率预测支路。图中的节点名称与 `openstl/models/mfwpn.py` 内的类或关键逻辑一致，可帮助快速定位源码实现。

```mermaid
graph TD
    A1[输入: 风场/温度等序列<br/>x_raw ∈ ℝ^{B×T×C×H×W}] -->|切分| A2[编码器 Encoder]
    A3[地形 DEM ele ∈ ℝ^{1×H×W}] -->|卷积嵌入| A2
    A2 -->|卷积/自注意力分支融合| B1[wcs 风场特征]
    A2 -->|卷积/自注意力分支融合| B2[tzcs 温度特征]
    B1 -->|reshape| C1[w ∈ ℝ^{B×T×C_w×H'×W'}]
    B2 -->|reshape| C2[tz ∈ ℝ^{B×T×C_tz×H'×W'}]
    C1 --> D1[MidMetaNet (风场分支)]
    C2 --> D2[MidMetaNet (温度分支)]
    D1 -->|通道注意力 + 门控融合| E1[融合特征 hid_w]
    D2 -->|提供调制信号| E1
    E1 -->|reshape| F1[解码输入 ℝ^{B·T×C_w×H'×W'}]
    F1 --> G1[Decoder]
    G1 --> H1[风场输出 Y ∈ ℝ^{B×T×2×H×W}]
    E1 -->|可选| P1[TurbinePowerHead]
    P1 --> H2[功率输出 P ∈ ℝ^{B×T×N_turbine×1}]
```

## 关键模块说明

- **Encoder**：由卷积 INN 分支和 Transformer 分支组成，分别处理风场 (`w`) 与温度 (`tz`) 输入，并利用地形卷积生成的门控权重进行融合。
- **MidMetaNet**：对时间维度展开后的特征执行一系列 `MetaBlock`（基于 `GASubBlock`）以建模时空依赖，风场与温度分支共享相同结构、独立参数。
- **通道注意力融合**：`CAM` 与 `SAM` 组合实现跨模态的注意力调制，将温度特征注入风场通道。
- **Decoder**：与编码器结构对称的卷积 + 自注意力模块，负责将融合后的高维特征还原为未来的风场。
- **TurbinePowerHead**（可选）：在开启风机坐标配置时，从 `hid_w` 上按 ROI 采样局部特征，经轻量卷积、池化与 MLP 输出各风机的功率序列。

## 使用提示

- 结构图中的 `H'×W'` 为编码器输出/解码器输入的空间尺寸（默认 64×80），应与 `configs.feature_hw` 保持一致。
- 若只预测风场，可将 `configs.turbine_coords` 设为 `None`，功率分支不会实例化，模型拓扑退化为原始 MFWPN。
- 功率预测分支对 ROI 大小、卷积通道和 MLP 隐层宽度的超参数可通过 `configs.power_roi_size`、`configs.power_conv_channels`、`configs.power_mlp_hidden` 调整。
