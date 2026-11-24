# 生产检测改为 y 规则（2026-09-12）

## 新生产定义（明确大抬头）

**鼻尖高于头顶**：`head_y − nose_y ≥ 5 px`（图像 y 向下为正）  
且持续 **≥ 0.35 s**，中值滤波 9 帧，置信度 ≥0.6。

不再把「俯仰角相对中位数下降」作为生产主判据（噪声大、全长误报多）。

## 全长人工 v2 金标准对比

| 方法 | P | R | F1 |
|---|---:|---:|---:|
| 俯仰角 k=1.25 + 30° | 0.19 | 1.00 | 0.32 |
| **生产 y 规则（5px / 0.35s / sm9）** | **0.72** | **0.50** | **0.59** |
| 扫描最优邻域（同参数） | ~0.91 | ~0.48 | ~0.63 |

精确率明显上升，召回下降——符合「只算明确大抬头」。

结果：`生产yrule_全长对比.csv`

## 已改代码

| 位置 | 改动 |
|---|---|
| `headlift.detect_lifts_yrule` | 增加 margin_px / 置信度 / 中值滤波 / min_dur_s |
| `headlift.detect_lifts_yrule_df` | 从 DLC DataFrame 一键检出 |
| `event_annotate_gui.draft_events` | 草稿改用生产 y 规则 |
| `headlift_core.detect_lifts_yrule` | 软件核心同步 |
| `main_gui` | 默认模式「鼻高于头顶（生产推荐）」，k=5，min_dur=0.35，sm=9 |

俯仰角 / 双通道等模式仍可切换，作研究对照。

## 使用

```python
import headlift as hl
df = hl.load_dlc_csv(csv_path)
ev, flags = hl.detect_lifts_yrule_df(df, fps=20)
```

GUI：默认已是「鼻高于头顶（生产推荐）」。

## 注意

- 1631/2171 在当前 y 规则下草稿很少，仍需人工扫全片补漏  
- 极短（&lt;0.35s）或斜向抬头可能漏——可把最短时长降到 0.2s 换召回  
- 论文仍可报告俯仰角法作方法学对照；落地以 y 规则为准
