# From Experience to Strategy: Empowering LLM Agents with Trainable Graph Memory
> source: https://arxiv.org/pdf/2511.07800  
> From_huilong\2511.07800v1.pdf

## Problem Setting

LLM agent 在开放环境中做多轮工具调用 QA。核心矛盾：**implicit memory**（RL 训练写入 θ）易遗忘且不可解释；**explicit memory**（prompt 注入）缺乏自适应性。

本文提出 **可训练的多层图记忆**（trainable hierarchical graph memory），将 mₗ 组织为异构有向图 𝒢 = (𝑉, 𝐸)，节点分三层：

| 层 | 符号 | 内容 |
|---|---|---|
| Query Layer | 𝒬 = {qᵢ} | 任务实例（含输入、T、成败标签） |
| Transition Path Layer | 𝒯 = {tⱼ} | 由 FSM 𝒮 = (S, A, T) 从原始 T 抽象出的规范决策路径 |
| Meta-Cognition Layer | ℳ = {mₖ} | 从成功/失败路径对比中蒸馏出的高层策略原则 |

边集 𝐸 ⊆ (𝒬×𝒯) ∪ (𝒯×ℳ)，配可学习权重 w_qt、w_tm。

Agent 交互：sₜ = (q, h₁:ₜ₋₁)，aₜ ~ πθ(aₜ|sₜ)，动作为 `<think>` / `<tool_call>` / `<answer>`，最多 K=6 轮工具调用。

## Training Procedure

三阶段：

### Stage 1: 图记忆构建
1. 对每个 qᵢ 采样 N 条 T，经 FSM 𝒮 映射为规范路径 tⱼ
2. **Meta-cognition 归纳**：若同一 q 存在成功 τₛ 和失败 τ_f，对比 FSM 路径生成高置信 mₖ；若仅有失败，检索 top-K 语义相似 q 的成功路径生成推测性 mₖ
3. 动态更新：新路径强化现有原则（通过evaluate its strategic value，更新confidence） / 创建新节点 / 丢弃冗余（低置信度的）

### Stage 2: 图权重优化（REINFORCE）
- 对新 q_new，选 top-k 近邻激活子图 𝒢(q_new)
- 候选 mₖ 的相关性得分：ρ(mₖ|q_new) = Σ Sim(q_new, qᵢ) · w_qt(i,j) · w_tm(j,k)
- 采样概率 p(mₖ|q_new) ∝ exp(ρ(mₖ|q_new))
- 对比实验得 ΔRₖ = R_with(mₖ) − R_w/o
- 损失：ℒ_RL = −𝔼[ΔRₖ · log p(mₖ|q_new)]，正 ΔR 强化路径，负 ΔR 削弱

### Stage 3: 记忆引导的策略优化
- 检索 top-k mₖ，拼接为增强 prompt：q̃_train = [m₁, …, mₖ; q_train]
- 用 GRPO 优化 πθ：

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_t\left[\min\left(\frac{\pi_\theta(a_t|\tilde{q})}{\pi_{\theta_{\text{old}}}(a_t|\tilde{q})}\hat{A}_t,\; \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

## Inference Process

1. 给定新 q，计算与图中历史 qᵢ 的语义相似度，激活子图
2. 沿学习后的权重 w_qt、w_tm 聚合，选出 top-k 高相关 mₖ
3. 将 mₖ 作为策略先验拼入 prompt → agent 执行多轮 think-tool-answer 循环

不修改 θ，纯 prompt 注入。

## Dataset & Evaluation

**数据**：仅用 HotpotQA 训练集（1000 条构建记忆 + 5000 条训练权重 + 剩余用于 RL）

**评估**：7 个 QA 数据集，指标 Exact Match (EM)

| 类型 | 数据集 |
|---|---|
| General QA (OOD) | NQ, TriviaQA, PopQA |
| Multi-hop QA | HotpotQA†, 2WikiMultiHopQA, Musique, Bamboogle |

**主要结果**（Inference）：

| 模型 | 方法 | Avg EM |
|---|---|---|
| Qwen3-8B | ITR baseline | 0.334 |
| Qwen3-8B | **Ours** | **0.365** (+9.3%) |
| Qwen3-4B | ITR baseline | 0.279 |
| Qwen3-4B | **Ours** | **0.351** (+25.8%) |

**主要结果**（RL Training）：

| 模型 | 方法 | Avg EM |
|---|---|---|
| Qwen3-8B | Search-R1 | 0.395 |
| Qwen3-8B | **Ours** | **0.408** (+3.3%) |
| Qwen3-4B | Search-R1 | 0.375 |
| Qwen3-4B | **Ours** | **0.426** (+13.6%) |

**消融**：去掉权重学习 → 性能显著下降；k=3 为最佳 meta-cognition 数量；换用 Gemini-2.5-pro 构建记忆仍有效，方法对 LLM backend 无关。

## RQ
这篇论文是解决agent在解决long-horizon task时tool calling时不稳定的问题。
### 1. Agent从query->planning->execution(tool calling)->Get final answer的trajectory构建的memory怎么能够被复用？论文提出了将这些trajectory（成功+失败）转换为可以被复用的策略先验，复用这些策略先验。
### 2. 这些策略先验怎么动态更新？策略先验的使用怎么能够generalized/adaptive？论文是通过RL去实现的。