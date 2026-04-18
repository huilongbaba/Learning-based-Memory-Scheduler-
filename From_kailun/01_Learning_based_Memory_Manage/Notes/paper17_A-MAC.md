# Adaptive Memory Admission Control for LLM Agents (A-MAC)

**Source:** [arXiv:2603.04549](https://arxiv.org/abs/2603.04549) (Published at ICLR 2026 Workshop MemAgent, 2026-03-04)

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$H = \lbrace t_1, t_2, \ldots, t_k \rbrace$|$C$ / trajectory of turns|多轮对话历史（上下文）|
|$M$|$m_l$|长期记忆存储（已入池的记忆集合）|
|$\lbrace m_1, m_2, \ldots, m_n \rbrace$|候选 $m_l$ 写入项|从对话中抽取的原子候选记忆|
|$\mathcal{U}(m)$|— (论文新引入)|Utility：由 LLM 打分的"未来可用性"|
|$\mathcal{C}(m)$|— (论文新引入)|Confidence：基于 ROUGE-L 的事实支持度|
|$\mathcal{N}(m)$|— (论文新引入)|Novelty：基于 SBERT 嵌入的新颖度|
|$\mathcal{R}(m)$|— (论文新引入)|Recency：指数衰减的时间相关性|
|$\mathcal{T}(m)$|— (论文新引入)|Type Prior：基于规则的内容类别先验|
|$\mathcal{S}(m)$|— (论文新引入)|候选记忆的综合入池分数|
|$\boldsymbol{\omega} = [w_1, \ldots, w_5]$|$\theta_{\text{admission}}$ (待学习参数)|线性记分器的权重向量|
|$\theta$ (论文)|$\theta_{\text{thr}}$|入池决策阈值（与框架参数 $\theta$ 同字母，需注意区分）|
|Admission policy $\pi_{\text{adm}}$|$g_l$（写入函数）中的"是否写入"子策略|从候选到 {admit, update, reject} 的映射|

> 本文定义的五个特征函数 $\mathcal{U}, \mathcal{C}, \mathcal{N}, \mathcal{R}, \mathcal{T}$ 为对候选记忆 $m$ 的标量评估函数。它们共同参数化了长期记忆写入策略 $g_l$ 的"门控"部分。

---

## 概览

用带有 ground-truth admission 标签的对话数据，每个候选被标注为"应入池 / 应拒绝"。候选通过对话分段 + 共指消解得到，然后用五个特征来刻画每条候选的价值。  
最后得到一个可插拔的记忆准入控制器：具体形式是一个线性加权打分器 $\mathcal{S}(m) = \sum_i w_i \cdot f_i(m)$ 加上阈值 $\theta$。这个控制器作为一个写入门决定哪些对话内容进入长期记忆 $m_l$。  
优化记忆写入决策的 F1 分数，记住值得的东西，不值得记的丢掉。同时附带优化延迟。它是监督学习 + 交叉验证网格搜索，优化目标是写入决策本身的分类指标。

---

## 1. Problem Setting

- **记忆类型**：处理的是 **cross-chat、cross-session 的长期记忆 $m_l$**。短期工作记忆 $m_s$ 未被显式讨论。
- **决策过程形式化**：**不是 MDP 也不是 bandit**，而是一个**有监督的二元/三元分类问题**——对每个候选 $m$ 独立判定 admit/update/reject。候选之间的顺序依赖仅通过 Novelty (与已存 $m_l$ 的相似度) 捕捉，没有完整的时序 credit assignment。
- **状态/动作/观测**：
    - 状态（候选层面）：$(m, H, M)$，即候选记忆、对话历史、当前记忆库
    - 动作空间 $\mathcal{A}$：$\lbrace \text{admit}, \text{update}, \text{reject} \rbrace$
    - 观测：五个特征值 $[\mathcal{U}, \mathcal{C}, \mathcal{N}, \mathcal{R}, \mathcal{T}] \in [0,1]^5$
- **记忆的数据结构**：**纯文本、原子化的自然语言条目集合**，形式上是 $$m_l = \lbrace m_i : m_i \text{ 是一个自包含的事实/偏好陈述} \rbrace$$ 每条目可被 ROUGE-L 和 SBERT embedding 处理。支持 CRUD，冲突时走合并 (Merge) 逻辑。

**核心组件与符号映射**：

|组件|论文表示|框架符号|实现|
|---|---|---|---|
|对话上下文|$H$|$C$|多轮对话的原始文本|
|长期记忆|$M$|$m_l$|原子事实条目的集合|
|候选抽取器|`Candidate Extraction`|extraction 子模块|规则 + LLM 分段|
|特征抽取器|$\mathcal{U},\mathcal{C},\mathcal{N},\mathcal{R},\mathcal{T}$|新引入|LLM(仅 U) + ROUGE-L + SBERT + 规则|
|入池策略|$\mathcal{S}(m) \geq \theta$|$g_l$ 的 gating 部分|线性打分 + 阈值|
|冲突处理|`FindConflict` + `Merge`|$g_l$ 的 update 部分|相似度 > 0.85 时合并|

---

## 2. Training Procedure

- **优化对象**：**独立于主 LLM 的外部打分器参数**，具体是权重向量 $\boldsymbol{\omega}^* = [w_1, \ldots, w_5]$ 和阈值 $\theta^*$。主 LLM 参数 $\theta_{\text{LLM}}$ **全程冻结**（甚至连 prompt 都固定）。
- **优化算法**：**不是 RL，也不是梯度下降**，而是**网格搜索 (grid search) + 5-fold 交叉验证**。搜索空间：
    - 权重 $\boldsymbol{\omega}$：非负、和为 1 的离散网格
    - 阈值 $\theta \in [0.3, 0.6]$ 的离散网格
- **训练数据来源**：**离线人工标注**——LoCoMo 基准中每条候选记忆都有 ground-truth admission 标签。训练集 70%，验证集 15%，测试集 15%。
- **冻结情况**：主 LLM（Qwen 2.5 local model）完全冻结；Sentence-BERT 编码器也是预训练现成的；唯一"学习"的参数是 5+1 = 6 个标量。

**核心训练目标函数**：

论文原文形式——按 F1 最大化选权重：

$$(\boldsymbol{\omega}^*, \theta^*) = \arg\max_{\boldsymbol{\omega}, \theta} \text{F1}_{\text{CV}}(\boldsymbol{\omega}, \theta)$$

其中综合分数为

$$\mathcal{S}(m) = w_1 \cdot \mathcal{U}(m) + w_2 \cdot \mathcal{C}(m) + w_3 \cdot \mathcal{N}(m) + w_4 \cdot \mathcal{R}(m) + w_5 \cdot \mathcal{T}(m)$$

约束 $w_i \geq 0$，$\sum_{i=1}^{5} w_i = 1$，入池条件为 $\mathcal{S}(m) \geq \theta$。

框架符号双重标注——将此视为对长期记忆写入策略 $g_l$ 的 gating 函数的监督学习：

$$g_l^{\text{admit}}(m \mid C, m_l; \boldsymbol{\omega}) = \mathbb{1}\lbrace \mathcal{S}(m) \geq \theta \rbrace$$

训练目标是

$$\boldsymbol{\omega}^* = \arg\max_{\boldsymbol{\omega}} \mathbb{E}_{(m, y^*) \sim \mathcal{D}} \big[ \text{F1}(g_l^{\text{admit}}(m; \boldsymbol{\omega}), y^*) \big]$$

其中 $y^*$ 是专家标注的 ground-truth admission 决定。

> 特征工程 + 线性分类器。

---

## 3. Reward Signal

- **奖励类型**：**这是一个监督学习设置，没有 RL 意义上的 reward**。训练信号是 per-candidate 的 ground-truth 二元标签 $y^* \in \lbrace 0, 1 \rbrace$（应入池 / 不应入池）。若强行映射到 RL 语境，它等价于一个 bandit 的 immediate binary reward。
- **奖励来源**：LoCoMo 基准提供的**人工标注的 memory-dependent task annotations**——即"如果后续任务依赖此条记忆，则标签为正"。
- **奖励如何分配**：**每个候选独立评估**，没有 credit assignment 问题。F1 是在整批候选上计算的聚合指标。
- **辅助奖励或正则项**：
    - 权重约束（$w_i \geq 0$, $\sum w_i = 1$）起到正则化作用
    - 阈值搜索范围限制在 $[0.3, 0.6]$ 防止极端解
    - 冲突合并机制（相似度 > 0.85）是手工规则，不进入学习目标


---

## 4. Inference Procedure

- **记忆初始化**：推理时 $m_l$ 从空集或已有知识库开始。候选抽取由规则 + LLM 完成，不涉及推理时学习。
- **每步决策流水线**（Algorithm 1）：
    1. 从对话 $H$ 抽取候选 $m$（规则 + LLM 分段、共指消解、过滤）
    2. **并行**计算五个特征：$\mathcal{U}$（单次 LLM 调用，~2580ms）、$\mathcal{C}$（ROUGE-L，~18ms）、$\mathcal{N}$（SBERT cos 相似度，~32ms）、$\mathcal{R}$（指数衰减，<1ms）、$\mathcal{T}$（规则匹配，~14ms）
    3. 用学到的权重求和得到 $\mathcal{S}(m)$
    4. 若 $\mathcal{S}(m) \geq \theta^*$：检测冲突
        - 若与现有 $m_{\text{conflict}}$ 相似度 > 0.85 且新分数更高：合并
        - 否则：直接 admit
    5. 否则：reject
- **额外推理策略**：
    - 五个特征并行计算（latency 优化）
    - Utility 得分缓存以避免重复 LLM 调用
    - Recency 使用 $\lambda = 0.01/\text{hr}$（半衰期 ~69 小时）
    - 阈值敏感性分析显示 $\theta \in [0.50, 0.60]$ 为鲁棒平台
- **策略构成**：**混合式**——学到的是 6 个标量（5 权重 + 1 阈值），而特征抽取器本身（ROUGE-L、SBERT、规则、LLM prompt）全是**手工设计**的。检索 $v$ 未在本文讨论，推测沿用标准向量检索。

---

## 5. RQ 分析

### RQ1 (What is memory?)

A-MAC 将记忆定义为原子化的自然语言条目集合（即框架中的 $m_l$ 纯文本非参数记忆）。每条是一个自包含的事实/偏好/身份陈述。属于 T1。

### RQ2 (Which component is optimized? Which signal is used?)

优化对象是一个独立的、与主 LLM 解耦的准入控制器（线性打分器的权重 + 阈值，共 6 个标量）。主 LLM 完全冻结。信号是人工标注的 ground-truth admission 标签（不是 RL reward，而是监督学习的 F1 目标）。对应 O1a（独立可插拔的记忆管理模型）。

### RQ3 (Target of Optimization)

A-MAC 的优化目标是记忆写入决策本身的质量，即 admission F1（precision 与 recall 的平衡），而非下游 QA 正确率。同时附带优化延迟/效率。对应 G2（效率），优化每次记忆写入/检索的计算开销；但不完全对应 G1（答案准确度），因为论文的主指标是 admission F1 而非 QA accuracy。Admission F1 可视为记忆是否用得好的代理指标。

### RQ4 (How memory evolves, operates?)

运行时，每个新对话轮次触发候选抽取 - 五特征并行计算 - 加权打分 - 阈值决策 - admit / merge / reject。演化机制是单次前向的门控写入，配合"相似度 > 0.85 触发合并"的规则实现 update。没有周期性反思、没有遗忘衰减（Recency 只影响写入打分，不触发删除）、没有多步 replan。写入由学到的线性策略驱动，但冲突处理仍是手工规则。

---

## Conclusion

A-MAC 专注于 LLM agent 长期记忆的一个环节，写入门控 (admission control)。它把"这条对话内容是否值得记住"建模为一个五维线性加权打分问题，特征分别覆盖未来有用性、事实可信度、新颖度、时效性和内容类型。其中只有 Utility 用 LLM 打分，其余四个用 ROUGE-L、SBERT、指数衰减和规则实现，因此比纯 LLM-native 的 A-mem 快 31%。权重通过 5-fold 交叉验证在 LoCoMo 标注数据上网格搜索得到，是一种监督学习而非 RL。A-MAC 展示了轻量学习 + 手工特征的组合可以在记忆写入任务上击败更昂贵的 LLM-native 方案，同时保留了调试性和可审计性。