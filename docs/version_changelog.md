# MFWPN 版本迭代文档

## 版本概览

| 版本 | 日期 | 核心变更 |
|------|------|----------|
| v1.0 | 2025.01 | 基础网格风场预测 (u, v 分量) |
| v2.0 | 2025.12 | 新增精确点风速校正 + 功率预测 |

---

## 总体架构流程图

```mermaid
flowchart TB
    subgraph 输入层
        A[气象网格数据<br/>B, T, 4, H, W] --> B[Encoder]
        E[高程数据<br/>H, W] --> B
    end

    subgraph 特征提取["特征提取层 (阶段1训练, 阶段2冻结)"]
        B --> C[风场分支 wcs]
        B --> D[温压分支 tzcs]
        C --> F[MidMetaNet hid_w]
        D --> G[MidMetaNet hid_tz]
        F --> H[时空融合]
        G --> H
    end

    subgraph 解码层
        H --> I[Decoder]
        I --> J[网格风场 Y<br/>B, T, 2, H, W]
    end

    subgraph 精确点预测["精确点预测层 (阶段2训练)"]
        J --> K[grid_sample<br/>双线性插值]
        H --> L[ROI 局部特征提取]
        K --> M[插值风速<br/>interp_speed]
        M --> N[WindCorrectionHead]
        L --> N
        N --> O[校正后风速<br/>corrected_speed]
        O --> P[TurbinePowerHead]
        L --> P
        P --> Q[功率预测<br/>power]
    end

    subgraph 输出层
        J --> R[outputs]
        M --> R
        O --> R
        Q --> R
    end

    style 精确点预测 fill:#e1f5fe
    style 特征提取 fill:#fff3e0
```

---

## 数据流程图

```mermaid
flowchart LR
    subgraph 阶段1数据["阶段1: 网格数据 (2020-2025)"]
        A1[ERA5 再分析数据] --> A2[网格风场 u, v]
        A1 --> A3[温度/气压场]
        A2 --> A4[train_grid.npy]
        A3 --> A4
    end

    subgraph 阶段2数据["阶段2: 精确点数据 (2024.8-2025.9)"]
        B1[风机 SCADA 数据] --> B2[实测风速]
        B1 --> B3[实测功率]
        B2 --> B4[turbine_*.npz]
        B3 --> B4
        B5[风机坐标] --> B6[coords.json]
    end

    subgraph 训练流程
        A4 --> C1[阶段1: 预训练主干]
        C1 --> C2[保存 checkpoint]
        C2 --> C3[阶段2: 加载并冻结主干]
        B4 --> C3
        B6 --> C3
        C3 --> C4[训练校正头 + 功率头]
        C4 --> C5[保存最终模型]
    end

    style 阶段1数据 fill:#c8e6c9
    style 阶段2数据 fill:#ffecb3
```

---

## 两阶段训练流程

```mermaid
sequenceDiagram
    participant D as 数据
    participant M as 模型
    participant O as 优化器

    rect rgb(200, 230, 200)
        Note over D,O: 阶段1: 预训练主干 (5年网格数据)
        D->>M: 网格风场数据 (2020-2025)
        M->>M: Encoder + MidMetaNet + Decoder
        M->>O: loss = MSE(Y_pred, Y_true)
        O->>M: 更新所有参数
        M->>M: 保存 checkpoint
    end

    rect rgb(255, 236, 179)
        Note over D,O: 阶段2: 训练校正头 (13个月精确点数据)
        M->>M: 加载 checkpoint
        M->>M: 冻结主干参数
        D->>M: 精确点数据 (2024.8-2025.9)
        M->>M: WindCorrectionHead + TurbinePowerHead
        M->>O: loss = MSE(speed) + λ·MSE(power)
        O->>M: 只更新校正头参数
    end
```

---

## 方案选择分析

### 需求回顾

**问题**：双线性插值得到的风速与精确位置实际风速差别较大，需要预测精确点的真实风速。

**候选方案**：
1. **方案一**：插值后加 MLP 校正头
2. **方案二**：预测时训练 (Test-Time Training, TTT)

---

### 方案对比

```mermaid
flowchart TB
    subgraph 方案一["方案一: 校正头 (MLP)"]
        direction TB
        A1[双线性插值] --> A2[插值风速]
        A3[ROI 局部特征] --> A4[MLP]
        A2 --> A4
        A4 --> A5[校正后风速]
        A6[监督信号: 实测风速] --> A7[MSE Loss]
        A5 --> A7
    end

    subgraph 方案二["方案二: 预测时训练 (TTT)"]
        direction TB
        B1[测试样本] --> B2[自监督任务]
        B2 --> B3[临时更新模型]
        B3 --> B4[预测结果]
        B5[无监督信号] --> B2
    end

    style 方案一 fill:#c8e6c9
    style 方案二 fill:#ffcdd2
```

---

### 为什么选择方案一而非方案二？

#### 1. 监督信号可用性

| 方面 | 方案一 (校正头) | 方案二 (TTT) |
|------|-----------------|--------------|
| **监督信号** | ✅ 有精确点实测风速 | ❌ 需要自监督信号 |
| **标签质量** | 高质量直接监督 | 间接/代理任务 |
| **学习目标** | 直接学习插值→真值映射 | 学习辅助任务，间接改善 |

**结论**：你的场景**有精确点标签**（虽然时间短），直接监督学习比自监督更高效、更准确。

---

#### 2. 适用场景分析

```mermaid
graph LR
    subgraph TTT适用场景["TTT 适用场景"]
        T1[测试时无标签]
        T2[存在域偏移]
        T3[分布变化频繁]
    end

    subgraph 你的场景["你的实际场景"]
        Y1[有精确点标签 ✅]
        Y2[训练/测试分布相似]
        Y3[可离线训练]
    end

    TTT适用场景 -->|不匹配| X[❌ TTT 不适用]
    你的场景 -->|匹配| V[✅ 监督学习更优]
```

**TTT 典型用例**：
- 图像分类中测试图像风格变化（如医疗影像不同设备）
- 自动驾驶中遇到训练时未见过的天气

**你的场景**：
- 测试时精确点数据与训练时来自**同一批风机**
- 不存在显著的域偏移，只是插值误差问题

---

#### 3. 数据量与效率

| 方面 | 方案一 (校正头) | 方案二 (TTT) |
|------|-----------------|--------------|
| **训练数据需求** | 13 个月精确点数据足够 | 无需训练数据，但每次推理都要微调 |
| **推理速度** | 快 (单次前向传播) | 慢 (每样本需多次梯度更新) |
| **部署复杂度** | 简单 | 复杂 (推理时需要梯度计算) |

```mermaid
graph TB
    subgraph 推理效率对比
        A[输入样本] --> B1[方案一: 单次前向]
        A --> B2[方案二: N次梯度更新]
        B1 --> C1[~10ms]
        B2 --> C2[~500ms+]
    end
```

---

#### 4. 技术实现复杂度

| 方面 | 方案一 (校正头) | 方案二 (TTT) |
|------|-----------------|--------------|
| **实现难度** | 低 (标准监督学习) | 高 (需设计自监督任务) |
| **调参复杂度** | 标准超参 | 需调自监督任务权重、更新步数等 |
| **可解释性** | 高 (直接校正) | 低 (间接影响) |
| **稳定性** | 高 | 可能不稳定 |

---

#### 5. 决策树

```mermaid
flowchart TD
    Q1{测试时有标签吗?}
    Q1 -->|是| A1[✅ 监督学习]
    Q1 -->|否| Q2{分布偏移大吗?}
    Q2 -->|是| A2[考虑 TTT/TTA]
    Q2 -->|否| A3[直接用预训练模型]

    A1 --> R1[方案一: 校正头]
    A2 --> R2[方案二: TTT]

    style Q1 fill:#fff9c4
    style R1 fill:#c8e6c9
    style R2 fill:#ffcdd2
```

**你的路径**：有标签 → 监督学习 → 方案一

---

### 方案一的优势总结

| 优势 | 说明 |
|------|------|
| **直接监督** | 用实测风速直接训练，学习效率高 |
| **两阶段训练** | 充分利用 5 年网格数据 + 13 个月精确点数据 |
| **推理高效** | 无需推理时梯度计算 |
| **可扩展** | 易于加入更多精确点 |
| **可解释** | 校正量可直接分析 |

---

### 方案二的潜在价值（未来考虑）

如果未来遇到以下情况，可考虑 TTT：
1. 新增风机**无历史数据**
2. 气候模式**显著变化**（如极端天气）
3. 需要**在线自适应**

---

## 版本 v2.0 变更清单

### 新增模块

| 模块 | 文件 | 功能 |
|------|------|------|
| `WindCorrectionHead` | `mfwpn.py:25-115` | 校正插值风速 |
| `TurbinePowerHead` 更新 | `mfwpn.py:118-221` | 支持校正风速输入 |

### 新增参数

```python
MFWPN_Model(
    turbine_coords=[(h, w), ...],        # 精确点网格坐标
    enable_wind_correction=True,          # 启用风速校正
    wind_correction_hidden=64,            # 校正头隐藏维度
    use_corrected_speed_for_power=True,   # 功率头使用校正风速
)
```

### 新增输出

```python
outputs = {
    'wind': [B, T, 2, H, W],              # 网格风场 (原有)
    'interp_speed': [B, T, N],            # 插值风速 (新增)
    'corrected_speed': [B, T, N],         # 校正风速 (新增)
    'power': [B, T, N, 1],                # 功率预测 (新增)
}
```

### 新增文档

| 文档 | 路径 |
|------|------|
| 技术文档 | `docs/turbine_point_prediction.md` |
| 版本迭代 | `docs/version_changelog.md` |

---

## 后续迭代计划

| 版本 | 预计内容 |
|------|----------|
| v2.1 | 多风机联合建模、功率曲线物理约束 |
| v2.2 | 不确定性量化、概率预测 |
| v3.0 | 支持 TTT 自适应（针对新风机场景） |
