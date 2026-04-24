# LightSearcher: Efficient DeepSearch via Experiential Memory

**Source:** [arXiv:2512.06653v3](https://arxiv.org/abs/2512.06653) (发布于 2025-12-10，正在投递至 ACM)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$q$|$q$|用户 query（多跳 QA 问题）|
|$r_t$|$t$|每步的 reasoning content（thought trace）|
|$d_t \in {\text{search, continue, answer}}$|$a$（的子空间）|每步的 decision/action|
|$s_t$|$o$|调用搜索工具后返回的 observation|
|$c_t$|$C$|当前上下文（previous reasoning + retrieved info）|
|$\tau = (q, (r_1,d_1,s_1),\ldots,(r_T,d_T,s_T), y)$|$T$（trajectory）|完整推理轨迹|
|$y$|$A_N$|最终答案|
|$\mathcal{K}$|$m_l$（的一种实现）|外部知识库（Wikipedia + E5 retriever）|
|Experience Memory Bank|$m_l^{\text{exp}}$（新引入）|文本形式的"经验记忆"，由对比轨迹 summarize 得到|
|Few-shot$(\tau)$|$m_l^{\text{fs}}$（新引入）|从历史高分轨迹中随机抽取的 few-shot 示例|
|Prompt$_{\text{aug}}$|$C_N$（增强后）|Instructions + Experience + Few-shot + $q$ 拼接成的增强 prompt|
|$\pi_\theta$|$\pi_\theta$|主 LLM 的 policy|
|$\theta$|$\theta$|主 LLM 参数（Qwen2.5-3B/7B-Instruct）|
|$R(\tau)$|reward|整条轨迹的多目标 reward|
|$\text{F1}(\tau), \text{Format}(\tau), \text{Tool}(\tau)$|子 reward 项|准确度 / 格式 / 工具使用 reward|

> 说明：论文引入了两类以自然语言形式存在的外部记忆，"Experience Memory"（经验总结条目）与"Few-shot 轨迹库"。本文将其分别记为 $m_l^{\text{exp}}$ 和 $m_l^{\text{fs}}$，两者都是 T1 类非参数化外部记忆的变体，但内容物理意义不同（经验规则 vs. 完整轨迹样例）。

---

## 概览

**这篇文章用了什么东西？** 用了训练过程中主 LLM 自己 rollout 出来的**历史轨迹**——把它们按最终 reward 分为"高分组"（完全正确且工具用得少）和"低分组"（错了或格式不对），再让一个外部 LLM 对这两组做**对比分析**，自动总结出"成功策略 / 要避免的陷阱 / 推理指南"三类自然语言规则。此外还随机抽取一条高分轨迹做 few-shot 示例。

**这篇文章最后得到了什么东西？** 一个经过 RL 微调的 **Qwen2.5-3B / 7B-Instruct DeepSearch 模型**，加上一个不断演化的**文本形式经验记忆库**（Experience Memory Bank）。训练完后 memory 不再需要参与推理——所有的"经验"已经被内化到模型权重里，推理时轻量、快速。

**这篇文章达到了什么样的目标 / 优化了什么？** 同时优化**回答准确度（F1 / EM / LLM-judge）与推理效率（tool call 次数、推理时间、token 消耗）**。核心创新是一个 **adaptive reward shaping** 机制：只在答案正确的轨迹里惩罚过多的工具调用，避免在答错时还逼迫模型节省工具调用而雪上加霜。最终在 4 个多跳 QA 数据集上与 SOTA（ReSearch）准确度相当，但工具调用 −39.6%、推理时间 −48.6%、token −21.2%。

---

## 1. Problem Setting

- **记忆类型**：主要是 **cross-episode long-term memory**（$m_l$），即跨 query 的"经验总结库"与"few-shot 轨迹库"；单条 episode 内部的上下文拼接属于 $m_s$ 但论文未显式区分。
- **决策过程建模**：MDP。每步 agent 在 state $s_t$（当前 reasoning history + retrieved info）上采样 action $d_t \sim \pi_\theta(\cdot \mid s_t, q)$，$d_t \in {\text{search}, \text{continue}, \text{answer}}$，直到 $d_T = \text{answer}$ 产出 $y$。
- **状态 / 动作 / 观测**：
    - 𝒮：$(q, r_{1:t}, d_{1:t-1}, s_{1:t-1}, \text{Prompt}_{\text{aug}})$
    - 𝒜：`<think>…</think>`、`<tool_call>…</tool_call>`、`<answer>…</answer>` 三类结构化 token 序列
    - Ω：搜索工具的返回文本 $s_t$（top-k Wikipedia 段落）
- **记忆数据结构**：
    - **知识库 $\mathcal{K}$**：E5-base-v2 embedding + KILT 2018 Wikipedia 段落（**向量库**）
    - **Experience Memory $m_l^{\text{exp}}$**：自然语言规则条目，分三类模板（Success Strategies / Pitfalls to Avoid / Reasoning Guidelines），**每 5 个 training step 全量重写**
    - **Few-shot 库 $m_l^{\text{fs}}$**：存储历史 reward ≥ $\theta_r$ 的完整轨迹，每次训练 sampling 时抽 1 条

**核心组件—符号对照**：

|组件|论文表达|框架符号|
|---|---|---|
|主模型|Qwen2.5-7B-Instruct with GRPO|$\pi_\theta$|
|外部知识库|KILT Wikipedia + E5 retriever|$m_l$（向量库）|
|经验记忆|Experience Memory Bank（文本）|$m_l^{\text{exp}}$|
|少样本示例|Few-shot trajectory pool|$m_l^{\text{fs}}$|
|记忆更新函数|每 5 步对比 Good/Bad 轨迹后由 LLM 重写|$g_l$|
|检索机制|训练时：全量 inject；推理时不注入|$v$（退化）|
|reward 函数|$R(\tau) = \mathcal{R}(\text{F1}, \text{Format}, \text{Tool})$|—|

---

## 2. Training Procedure

- **优化对象**：**主 LLM 参数 $\theta$** 本身（全参数更新），而非独立的记忆管理模块。Experience Memory 作为 **prompt 层的输入** 参与 rollout，但它自身不是被优化的对象（它由 LLM 文本生成得到，不走反向传播）。
- **优化算法**：**GRPO**（Group Relative Policy Optimization）。每个 query 采样 12 条 rollout 构成一组，按 reward 归一化出 advantage。
- **训练数据来源**：
    - Musique 3,000 条 + 2WikiMultihopQA 4,000 条（in-domain 训练）
    - 每轮训练过程中由当前 $\pi_\theta$ online rollout 产生轨迹，同时作为经验生成的素材
- **LLM 冻结吗？** 否。主 LLM 参数 $\theta$ 被全量 RL 更新。
- **训练目标函数**（论文原始公式 + 框架符号）：

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid \text{Prompt}_{\text{aug}})}\left[\sum_{t=0}^{T}\log \pi_\theta(a_t \mid s_t, \text{Prompt}_{\text{aug}}) \cdot A_t\right]$$

其中

$$\text{Prompt}_{\text{aug}} = \lbrace \text{Instructions},\ m_l^{\text{exp}},\ m_l^{\text{fs}},\ q \rbrace$$

$A_t$ 由 $R(\tau)$（见 §3）组内归一化而来。

---

## 3. Reward Signal

- **奖励类型**：**Sparse terminal reward**——只在 rollout 结束时基于整条 trajectory 计算单一标量 $R(\tau)$，再由 GRPO 做 token 级 advantage 分配。
- **奖励来源**：
    - F1 分数：答案与 ground truth 的 token overlap（规则计算）
    - Format：基于 `<think>` / `<tool_call>` / `<answer>` 标签的二值合规检查（规则）
    - Tool：基于每条 query 历史上"最少成功工具数" $n$ 的自适应 shaping（规则）
- **奖励分配**：GRPO 做 group-relative normalization，然后均匀分配到 trajectory 的每个 token。
- **辅助奖励与正则项**：**Adaptive Reward Shaping** 是本文核心创新。Tool reward 定义为：

$$\text{Tool}(\tau) = \begin{cases} e^{-\lambda \cdot \max(0,\ m-n)}, & \text{if } \text{F1}(\tau) \geq \theta_t \ 0, & \text{if } \text{F1}(\tau) < \theta_t \end{cases}$$

其中 $m$ 是当前轨迹的工具调用次数，$n$ 是历史上该 query 最少成功工具数，$\lambda=0.75$。**只有在答案正确时（F1 ≥ 0.8）才惩罚多余工具调用**；答错时 tool reward 置 0，不再雪上加霜。

整体 reward：

$$R(\tau) = \begin{cases} -1, & \text{Format}(\tau) = -1 \ W_\alpha \cdot \text{F1}(\tau) + W_\beta \cdot \text{Tool}(\tau), & \text{Format}(\tau) = 0 \end{cases}$$

其中 $W_\alpha = W_\beta = 0.5$。

> 把 accuracy-efficiency 的 see-saw 显式解耦成两阶段：先保证答对，再在答对的前提下追求少用工具。

---

## 4. Inference Procedure

- **记忆初始化**：推理时 **$m_l^{\text{exp}}$ 和 $m_l^{\text{fs}}$ 都不注入 prompt**。只保留 Instructions + $q$。知识库 $\mathcal{K}$（Wikipedia + E5）照常在线使用。
- **每步决策流程**：
    1. Model 基于当前 $c_t$ 和 $q$ 生成 `<think>` 段（reasoning）；
    2. 生成 `<tool_call>` 或 `<answer>` 标签决定下一动作；
    3. 若为 `<tool_call>`，则调用 E5 retriever 在 Wikipedia 上检索 top-k 段落作为 $s_t$；
    4. 拼接 $s_t$ 回上下文，返回步骤 1，直到模型自发输出 `<answer>`。
- **额外推理策略**：
    - 无 TopK 调整、无温度退火、无多轮 replan
    - Fig. 4 显示训练后期（step > 200）加不加 experience 对性能几乎无影响 → 作者据此在推理时去除 experience 以省算力
- **策略完全由学习驱动**：决策由 $\pi_\theta$ 驱动，无手工规则介入。唯一的"硬编码"是格式标签解析器。

> 推理时 $m_l^{\text{exp}}$ 和 $m_l^{\text{fs}}$ 都不参与。

---

## 5. RQ 分析

### RQ1 (What is memory?)

LightSearcher 中"记忆"有两层：（a）外部知识库 $\mathcal{K}$：Wikipedia + E5 向量检索（T1 非参数外部记忆的标准形式，训练 / 推理都用）；（b）Experience Memory $m_l^{\text{exp}}$：训练过程中从对比轨迹总结出的自然语言规则条目（也是 T1，但只在训练阶段存在）。此外还有 Few-shot 轨迹池 $m_l^{\text{fs}}$（T1）。把训练 rollout 产生的元经验沉淀成可读文本条目这一显式记忆形态。

### RQ2 (Which component is optimized? Which signal is used?)

优化对象是主 LLM 的参数 $\theta$（Qwen2.5-3B/7B-Instruct），属于 O2a 单 LLM 优化。Experience Memory 本身不是优化对象，它是训练信号的一部分（通过 prompt augmentation 影响 rollout 分布）。训练信号来自多目标 reward：F1（答案正确性）+ Format（格式合规）+ adaptive Tool shaping（效率惩罚）。

### RQ3 (Target of Optimization)

同时优化 G1（回答准确度）+ G2（效率）。G1 通过 F1 奖励体现；G2 通过 adaptive tool-call 指数衰减惩罚体现。论文大量报告 tool call count / inference time / token 消耗等效率指标，其卖点正是在维持 G1 持平 SOTA 的前提下显著优化 G2。

### RQ4 (Training Signal and RL Algorithm)

A2: GRPO。每个 prompt 采样 12 条 rollout 成组，用组内 reward 归一化得到 advantage，然后做 policy gradient 更新。论文直接沿用 DeepSeekMath / ReSearch 的 GRPO 实现，无算法层面的变体。

### RQ5 (How memory evolves, operates?)

Experience Memory 的演化：每 5 个 training step，从当前 batch 中收集高 reward (R=1) 与低 reward (R<0.3) 的轨迹，做对比 summarize 后全量重写 $m_l^{\text{exp}}$（不是增量追加）。训练早期（step 5）规则聚焦格式，中期（step 50）聚焦"识别关键主体再检索"，后期（step 300）聚焦"能直接回答就不搜索"。Few-shot 池则持续累积高分轨迹。推理时两者都不参与，因为经验已被内化到 $\theta$。知识库 $\mathcal{K}$ 则在训练与推理时都被标准 embedding 检索。

---

## Conclusion

LightSearcher 解决的是 RL-based DeepSearch 系统中"为了准确率而过度调用搜索工具"这一问题。作者的做法有两个：第一，让 LLM 自己看自己 rollout 出来的好轨迹和坏轨迹，对比总结出自然语言形式的经验（比如"能直接答就别搜"），把这些经验塞进训练时的 prompt 里指导后续 rollout；第二，设计了一个"只在答对时才罚多余工具调用"的奖励函数，避免 RL 训练时因追求效率而牺牲准确率。两招配合 GRPO 训练一个 Qwen2.5-7B 模型，结果在 4 个多跳 QA 数据集上准确率追平 SOTA（ReSearch），但工具调用减了 40%、推理时间减了近一半、token 省了 20%。一个值得注意的现象是：训练后期这些文本经验的作用趋于 0（因为被模型内化到参数里了），所以推理时根本不用加载 experience，这也使得它在部署时非常轻量。