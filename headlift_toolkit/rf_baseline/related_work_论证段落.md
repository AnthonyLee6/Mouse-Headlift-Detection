# Related Work 论证段落：为何选择规则法而非监督分类（中/英两版）

> 用途：毕设第 2 章（国内外研究现状）收尾 + EI 论文 Related Work / Method 开头。
> 方括号【】为文献占位符，成稿时替换为你参考文献列表的真实编号。
> 英文版比中文版更收敛，符合 EI 会议篇幅习惯。

---

## 中文版（毕设用）

在姿态估计层之上进行行为识别，现有工作大体形成三条路线。监督分类路线以 MARS
【MARS】与 SimBA【SimBA】为代表，将关键点转化为大量时空特征后训练随机森林等分类器，
并借助 SHAP 等工具提供可解释性；DeepEthogram【DeepEthogram】则直接从原始像素学习，
结合光流与时序模型输出行为序列。无监督发现路线以 B-SOiD【B-SOiD】与 MoSeq【MoSeq】
为代表，通过降维聚类自动发现行为单元，但聚类结果与实验者关注的行为类别之间仍需人工
对齐。主动学习路线如 A-SOiD【A-SOiD】，以最小标注代价迭代优化分类边界。

上述方法均面向多行为类别的通用场景设计，其代价是需要相当规模的人工标注：监督分类
要求对每种目标行为提供逐帧或逐段标签，无监督发现则要求人工解释并命名行为簇。而本文
关注的抬头行为具有两点特殊性：其一，它是单一视角 2D 视频下的头部微行为，主要由头部
俯仰角的变化刻画，特征空间天然低维且物理含义明确；其二，实验中每种行为条件下的可用
动物数量有限（本文为 4 只），逐帧标注的规模不足以支撑一个泛化稳健的监督分类器。

因此，本文选择将行为判定显式建模为头部俯仰角轨迹上的规则：以视频自适应的动态阈值
（中位数 − k·σ）划分抬头/中性状态，辅以幅度过滤、状态机与最短持续时间约束。该设计
零训练、逐帧可解释、每个参数都有明确物理含义，并且其跨动物稳健性可通过消融实验直接
检验（第 X 节）。作为对照，本文还训练了基于相同姿态特征的随机森林帧分类器
（留一鼠交叉验证），在人工标注帧集上与规则法进行定量比较（第 X 节）。

---

## 英文版（EI 会议用）

Existing pipelines for behavior recognition on top of pose estimation fall into
three families. Supervised approaches such as MARS [MARS] and SimBA [SimBA]
convert keypoints into rich spatio-temporal features and train classifiers
(e.g., random forests) with post-hoc explanations; DeepEthogram [DEG] learns
directly from pixels with optical flow and temporal networks. Unsupervised
discovery methods (B-SOiD [BSOID], MoSeq [MoSeq]) cluster posture dynamics into
behavioral syllables that must subsequently be matched to categories of
interest, and active-learning platforms such as A-SOiD [ASOID] reduce but do
not eliminate the labeling cost.

All of these target general-purpose, multi-behavior settings and presuppose a
labeling budget that scales with the number of behaviors. Vertical head-lift in
mice is different in two respects: it is a subtle head-posture behavior fully
characterized by the head pitch angle in monocular 2D video, and the number of
available animals per condition is small (four in this study), which makes a
per-frame annotated training set for a generalizing classifier impractical.

We therefore formulate lift detection explicitly as a rule on the head-pitch
trajectory: a video-adaptive dynamic threshold (median − k·σ) separates lifted
from neutral posture, combined with an amplitude filter, a state machine
requiring return to neutral between events, and a minimum-duration constraint.
The rule requires no training, is interpretable per frame, and its robustness
across animals can be evaluated directly by ablation (Sec. X). As a quantitative
baseline, we additionally train a random-forest frame classifier on identical
pose features and compare both methods against human frame-level labels under
leave-one-mouse-out cross-validation (Sec. X).

---

## 备选短版（英文 3 句，插在 Method 开头或 Rebuttal 用）

Supervised behavior classifiers (MARS, SimBA) and unsupervised behavior
discovery (B-SOiD, MoSeq) require per-frame labels or manual cluster
interpretation that do not scale to our setting (four animals, a single
low-dimensional head-pitch behavior). We instead detect lifts with an adaptive
threshold rule on the head-pitch trajectory, whose parameters have direct
physical meaning. A random-forest classifier on identical features serves as a
learned baseline under leave-one-mouse-out cross-validation.
