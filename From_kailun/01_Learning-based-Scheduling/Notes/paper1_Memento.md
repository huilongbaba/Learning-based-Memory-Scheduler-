# Memento: Fine-tuning LLM Agents without Fine-tuning LLMs

> source: https://arxiv.org/abs/2508.16153 From_kailun/01_Learning-based-Scheduling/Papers/Memento.pdf

## Problem Setting

现有 LLM agent 存在两类局限：静态框架（固定工作流，部署后无法适应新情况）和参数微调（通过 SFT/RL 更新 θ，计算代价高且难以持续在线学习）。

本文提出不修改 θ 的持续学习范式，将 agent 规划建模为 Memory-Based MDP (M-MDP) ⟨𝒮, 𝒜, 𝒫, ℛ, γ, ℳ⟩，通过 Case-Based Reasoning (CBR) 实现在线适应。

|组件|符号|定义|
|---|---|---|
|状态空间|𝒮|所有有限长度 token 序列（任务描述/当前上下文）|
|动作空间|𝒜|所有有限长度 token 序列（计划/工具调用）|
|记忆空间|$\mathcal{M} = (\mathcal{S} \times \mathcal{A} \times \mathbb{R})^*$|过往经验的集合|
|Case Bank|$M_t = {c_i}_{i=1}^{N_t}$|时间步 t 的记忆，每个 case $c_i = (s_i, a_i, r_i)$|
|检索策略|$\mu(c \mid s, M)$|给定当前状态 s 和记忆 M，选择 case c 的概率分布|
|LLM 策略|$p_{\text{LLM}}(a \mid s, c)$|给定状态 s 和检索到的 case c，LLM 生成动作 a|

整体策略 π 定义为检索策略 µ 与 LLM 策略的混合：

$$ \pi(a \mid s, M) = \sum_{c \in M} \mu(c \mid s, M) \cdot p_{\text{LLM}}(a \mid s, c) $$

关键设计：θ 固定不变，学习仅发生在 µ（检索策略）上，通过外部记忆 $m_l$（Case Bank）实现策略改进。

- **$m_s$**：Subtask Memory（文本形式，记录当前子任务及结果）+ Tool Memory（文本形式，记录工具调用历史）
- **$m_l$**：Case Memory（向量化存储的 Case Bank），支持 Write 和 Read 操作

轨迹 $T = {M_0, s_0, c_0, a_0, r_0, M_1, s_1, c_1, a_1, r_1, \cdots}$，其概率分解为：

$$ p(T) = \prod_{t=0}^{T-1} f_1(t) \cdot f_2(t) \cdot f_3(t) \cdot f_4(t) \cdot f_5(t) $$

其中：

- $f_1(t) = \mu(c_t \mid s_t, M_t)$（**Retrieve**：检索）
- $f_2(t) = p_{\text{LLM}}(a_t \mid s_t, c_t)$（**Reuse & Revise**：复用与修正）
- $f_3(t) = \mathbf{1}[r_t = \mathcal{R}(s_t, a_t)]$（**Evaluation**：评估）
- $f_4(t) = \mathbf{1}[M_{t+1} = M_t \cup (s_t, a_t, r_t)]$（**Retain**：留存）
- $f_5(t) = \mathcal{P}(s_{t+1} \mid s_t, a_t)$（**Transition**：转移）

架构为 Planner-Executor 框架：Planner 为 CBR agent（GPT-4.1），Executor 为工具增强 LLM（o3/o4-mini），通过 MCP 协议调用外部工具。

## Training Procedure

采用 **Soft Q-Learning** 优化检索策略 µ，θ（LLM 参数）保持冻结。

**最大熵 RL 目标**：

$$ J(\pi) = \mathbb{E}_{\tau \sim p}\left[\sum_{t=0}^{T-1} \left[\mathcal{R}(s_t, a_t) + \alpha \mathcal{H}(\mu(\cdot \mid s_t, M_t))\right]\right] $$

**最优检索策略**（闭式解）为 Q 值的 softmax：

$$ \mu^{\ast}(c \mid s, M) = \frac{\exp\bigl(Q^{\ast}(s, M, c) / \alpha\bigr)}{\sum_{c' \in M} \exp\bigl(Q^{\ast}(s, M, c') / \alpha\bigr)} $$

提供两种记忆变体：

### 非参数记忆（Non-Parametric）

- **Write**：直接追加 case：$M_{t+1} = M_t \cup {(s_t, a_t, r_t)}$
- **Read**：基于余弦相似度检索 TopK：$\text{Read}_{\text{NP}}(s_t, M_t) = \text{TopK}_{(s_i, a_i, r_i) \in M_t} \text{sim}(\text{enc}(s_t), \text{enc}(s_i))$
- **无需训练**，使用冻结的 SimCSE 编码器

### 参数记忆（Parametric）

由于 Memento 中 CBR 仅用于单步规划（非多步 M-MDP），TD target 退化为即时奖励，学习目标简化为监督学习：

$$ \mathcal{L}(\theta_Q) = \mathbb{E}_{(s,c,r)} \left[(Q(s, c; \theta_Q) - r)^2\right] $$

因奖励为二值信号 $r \in {0, 1}$，使用 **交叉熵损失**替代 MSE 以获得更稳定的梯度：

$$ \mathcal{L}(\theta_Q) = \mathbb{E}_{(s,c,r)} \left[-r \log Q(s, c; \theta_Q) - (1-r) \log(1 - Q(s, c; \theta_Q))\right] $$

其中 Q 网络为两层 MLP，输入为 SimCSE 编码的 $(s, c)$，输出为 $p(r=1 \mid s,c)$。训练数据存储于 Replay Buffer $\mathcal{B}$，在线持续更新。

**注意**：此处 $\theta_Q$ 是检索 Q 网络的参数，**非 LLM 的参数 θ**。LLM 参数自始至终保持冻结。

### 扩展：基于 Kernel 的 Episodic Control（理论框架）

论文还提出基于 kernel 的 Q 值近似（用于一般多步 M-MDP 设定，Memento 实际实现中未使用）：

$$ Q_{\text{EC}}(s, M, c; \theta_k) = \frac{\sum_{(s', c', Q') \in \mathcal{D}_c} k_{\theta_k}(s, s') , Q'}{\sum_{(\hat{s}, \hat{c}, \hat{Q}) \in \mathcal{D}_c} k_{\theta_k}(s, \hat{s})} $$

通过 TD 学习优化 kernel 参数 $\theta_k$。

## Reward Signal

- **任务级终端奖励**：$r \in {0, 1}$（二值）
- 评估方式：Exact Match（GAIA）/ LLM-as-judge 的 F1 和 PM 分数（DeepResearcher、SimpleQA、HLE）
- **无中间步骤奖励**：仅在任务完成后写入 Case Bank 并更新 Q 网络
- 奖励信号直接用于 Q 网络的监督训练（交叉熵损失），而非传统 RL 的策略梯度

## Inference Process

1. **输入**：用户 query q 作为初始状态 $s_0$
2. **Case-Based Planning（Stage 1）**：
    - Planner（GPT-4.1）接收 q，通过 Read 操作从 $m_l$（Case Bank）检索 K 个最相关 case
        - 非参数：按 $\text{sim}(\text{enc}(s_0), \text{enc}(s_i))$ 排序取 TopK
        - 参数：按 $Q(s_0, c_i; \theta_Q)$ 排序取 TopK（K=4 为最优）
    - 将检索到的 case（含成功与失败经验）拼接到 prompt 中，LLM 生成分解计划
3. **Tool-Based Execution（Stage 2）**：
    - Executor（o3/o4-mini）按子任务逐个执行，通过 MCP 协议调用外部工具（搜索、爬虫、代码执行、图像/视频/文档处理等）
    - $m_s$ 中的 Subtask Memory 记录子任务及结果，Tool Memory 记录工具调用历史
4. **迭代循环**：Planner 根据已完成的子任务结果判断是否需要 replan；未完成则更新上下文继续规划
5. **完成后写入记忆**：任务完成时，将 $(s, a, r)$ 写入 $m_l$（Case Bank），参数版本同时更新 Q 网络
6. **持续学习**：Case Bank 跨任务累积，后续任务可检索先前经验，实现无需更新 θ 的在线适应

策略完全由**检索策略 µ + 冻结的 $p_{\text{LLM}}$** 驱动。参数记忆通过在线更新 Q 网络实现自适应检索；非参数记忆仅靠相似度匹配。

## Dataset & Evaluation

**基座模型**：Planner = GPT-4.1；Executor = o3（GAIA）/ o4-mini（其他）；Encoder = SimCSE

|类型|数据集|设置|
|---|---|---|
|Long-horizon tool use|GAIA|450题，3个难度级别，validation 150 / test 300|
|Real-time web research|DeepResearcher (7个QA数据集)|NQ, TQ, HotpotQA, 2Wiki, Musique, Bamboogle, PopQA|
|Factual precision|SimpleQA|4,330 单跳事实题|
|Long-tail academic|HLE|2,500 跨学科专家级问题|

**指标**：EM（GAIA）/ F1 + PM（其他，GPT-4o-mini 作评估器）
> 术语说明：Pass@K 表示对同一任务运行 K 次，只要其中任意一次正确即算通过。Pass@3 即 3 次尝试取最优结果。PM（Partial Match）为基于 LLM-as-judge 的语义部分匹配分数，衡量生成答案与标准答案之间的语义一致性。

**主要结果**：

|方法|GAIA Val|DeepResearcher Avg F1/PM|SimpleQA PM|HLE PM|
|---|---|---|---|---|
|Memento (Pass@3)|**87.88%**|**66.6 / 80.4**|**95.0**|**24.4**|
|Manus|73.30%|—|—|—|
|DeepResearcher (training-based)|—|51.8 / 60.5|—|—|
|CoT + RAG|—|37.7 / 43.2|—|—|

**消融**：

- 去掉 CBR → DeepResearcher 平均 -6.7 F1 / -8.2 PM
- 参数 CBR > 非参数 CBR > 无 CBR（持续学习曲线持续上升）
- K=4 为最优检索数量；更多 case 带来噪声而非收益
- OOD 泛化：CBR 在未见任务上带来 +4.7% ~ +9.6% 绝对提升

## RQ

这篇文章处理的是 cross-task memory，不修改 LLM 参数 θ。以 Deep Research（长程工具使用 + 多跳推理）为牵引任务。

### 1. 如何在不修改 LLM 参数的前提下，实现 agent 的持续在线学习？

通过将规划过程建模为 M-MDP，引入外部 Case Bank 作为 $m_l$，用 CBR 范式替代参数微调：检索过往成功/失败经验来指导当前决策。

### 2. 如何优化 Case 检索策略，使 agent 能从记忆中选择最有价值的经验？

将检索策略 µ 的学习形式化为 Soft Q-Learning 问题，学习 $Q(s, c; \theta_Q)$ 来评估每个 case 对当前状态的价值，最优策略为 Q 值的 softmax 分布。单步设定下退化为二分类监督学习。