# Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM Memory

**Source:** [https://arxiv.org/abs/2602.15313](https://arxiv.org/abs/2602.15313) (February 2026) Accepted to ACL2026

---

## 符号映射表

|论文原始概念 / 符号|框架符号|说明|
|---|---|---|
|Episode|$m_l$ 中的原始文本条目|历史对话的原始片段，作为长期记忆的基本单元|
|Entity|$m_l$ 中的结构化节点|从 Episode 中抽取的具体人、物、组织等|
|Edge|$m_l$ 中的关系|描述 Entity 之间关系的可验证陈述|
|Episodic Edge|$m_l$ 中的索引关系|连接 Entity 与其来源 Episode 的链接|
|Category (层级图节点)|$m_l^{\text{hier}}$（新增符号）|从 Entity 自底向上抽象出的多层级语义类别|
|Base Graph|$m_l^{\text{base}}$|包含 Entity、Edge、Episode、Episodic Edge 的底层图|
|Hierarchical Graph|$m_l^{\text{hier}}$|包含 Category 节点和 Category Edge 的层级图|
|System-1 Similarity Search|$v_{\text{sim}}$|基于 embedding 相似度的检索路由|
|System-2 Global Selection|$v_{\text{global}}$|基于层级图自顶向下遍历的检索路由|
|Re-ranker|$v_{\text{rerank}}$|对两路检索结果的合并与重排序|
|LLM (GPT-4.1-mini)|$M$|用于图构建、层级抽象、全局选择和最终回答的模型|
|Query|$q$|用户查询|
|Final Answer|$A_N$|最终回答|

> **新增符号说明**：$m_l^{\text{base}}$ 和 $m_l^{\text{hier}}$ 是对 $m_l$ 的子分区，分别表示底层知识图和层级抽象图。$v_{\text{sim}}$ 和 $v_{\text{global}}$ 是两种不同的检索算法，合并后通过 $v_{\text{rerank}}$ 输出最终检索结果。

---

## 概览

这篇文章用历史对话消息 Episodes，通过 LLM prompting 从中抽取 Entity 和 Edge 构建 base graph，再自底向上将 Entity 抽象为多层 Category 构建 hierarchical graph。  
最后得到一个记忆框架，包含双路检索机制：System-1（embedding 相似度搜索）和 System-2（LLM 驱动的层级图自顶向下全局选择）。是一个纯 prompting + 图工程的系统。   
优化目标是回答准确度 G1（LLM-as-a-Judge score）。这篇文章是纯架构设计。

---

## 1. Problem Setting

- **记忆类型**：cross-chat memory，属于 $m_l$（长期记忆）。Mnemis 处理的是跨多轮对话积累的历史消息。
- **决策过程**：本文未将记忆管理建模为 MDP / POMDP 等决策过程。记忆的写入（图构建）和读取（检索）都是通过固定的 prompting pipeline 完成，不涉及学习式决策。
- **状态空间 / 动作空间 / 观测空间**：不适用。本文不是基于 RL 或序列决策的框架。
- **记忆数据结构**：图结构（Graph）。具体包含两个子图：
    - **Base Graph**（$m_l^{\text{base}}$）：包含 Entity 节点、Edge（实体间关系）、Episode（原始对话片段）、Episodic Edge（实体-对话链接）
    - **Hierarchical Graph**（$m_l^{\text{hier}}$）：包含多层 Category 节点和 Category Edge，支持自顶向下的语义层级遍历

|核心组件|框架符号|描述|
|---|---|---|
|历史对话片段|Episode $\in m_l^{\text{base}}$|原始文本，记忆的最底层数据|
|实体节点|Entity $\in m_l^{\text{base}}$|从 Episode 中抽取的具名实体|
|关系边|Edge $\in m_l^{\text{base}}$|Entity 之间的可验证关系陈述|
|层级类别|Category $\in m_l^{\text{hier}}$|对 Entity 的多层抽象概念|
|相似度检索|$v_{\text{sim}}$|System-1：基于 embedding 的 top-k 检索|
|全局选择|$v_{\text{global}}$|System-2：LLM 驱动的层级图遍历|
|重排序|$v_{\text{rerank}}$|合并两路结果，RRF 排序|

> Mnemis 的记忆结构是 T2（多组件 / 多层级记忆）的典型代表，其创新点在于引入层级图和双路检索，而非简单的 KV 或纯文本存储。与 A-MEM 类似，但 Mnemis 的层级是自底向上由 LLM 构建的语义抽象层级，而非预定义的 core/semantic/episodic 分区。

---

## 2. Training Procedure

本文不涉及任何训练过程。

- Mnemis 是一个纯 prompting + 图工程的框架，所有组件（图构建、层级抽象、检索、重排序）均通过 LLM prompting 完成。
- LLM（GPT-4.1-mini）的参数完全冻结，不进行任何微调或优化。
- 没有可学习的检索模型、辅助模型或参数化组件。
- 图的构建遵循三个手工设计的原则：Minimum Concept Abstraction、Many-to-Many Mapping、Compression Efficiency Constraint。

> 本文的核心贡献是架构设计而非学习算法。

---

## 3. Reward Signal

本文不涉及 reward signal。

---

## 4. Inference Procedure

- **记忆初始化**：对话历史逐条作为 Episode 输入。LLM 从每个 Episode 中抽取 Entity 和 Edge，构建 base graph。然后自底向上将 Entity 聚合为多层 Category，构建 hierarchical graph。
- **每步决策流程**：
    1. **接收查询** $q$
    2. **System-1 路由**：将 $q$ 编码为 embedding，在 base graph 中检索 top-k 相似的 Entity 和 Edge，再扩展到关联的 Episode
    3. **System-2 路由**：将 $q$ 输入 LLM，在 hierarchical graph 的最高层选择相关 Category，逐层向下遍历，最终到达底层 Entity，再检索关联的 Edge 和 Episode
    4. **合并与重排序**：将两路结果取并集，使用 Reciprocal Rank Fusion (RRF) 重排序
    5. **生成回答**：将排序后的记忆 context 与 $q$ 一起输入 LLM，生成最终回答 $A_N$
- **额外策略**：System-2 的层级遍历是一种"deliberate"（慎思型）检索策略，类比 Kahneman 的 System-2 思维；System-1 是快速的相似度匹配。两者互补。
- **策略来源**：完全由手工规则驱动，无学习得到的 $\pi$。

> Mnemis 的推理流程设计精巧，System-1 + System-2 的双路互补是其核心创新。但所有决策逻辑都是硬编码的 prompting pipeline，没有可学习的组件。这意味着框架的性能高度依赖于 LLM 的 prompting 能力和图构建质量。

---

## 5. RQ 分析

### RQ1 (What is memory?)

Mnemis 将记忆表示为双层图结构：base graph 包含从对话中抽取的 Entity、Edge、Episode 和 Episodic Edge；hierarchical graph 通过自底向上抽象将 Entity 组织为多层 Category。属于 T2。

### RQ2 (Which component is optimized? Which signal is used?)

本文未涉及此问题。 是纯架构设计。

### RQ3 (Target of Optimization)

本文未涉及训练意义上的优化目标。 但其评估指标是回答准确度 G1（LLM-as-a-Judge score）。若将 Mnemis 视为一个系统级优化（架构设计层面），其追求的是回答准确度的提升。

### RQ4 (How memory evolves, operates?)

记忆在 ingestion 阶段通过 LLM prompting 逐步构建：每条新对话 → 抽取 Entity/Edge → 更新 base graph → 自底向上更新 hierarchical graph。检索阶段通过双路并行：System-1 做 embedding 相似度搜索，System-2 做 LLM 驱动的层级遍历，两路结果合并重排。记忆是只增长的（append-only），论文未讨论记忆的删除或遗忘机制。

---

## Conclusion

这篇文章提出了一个长期记忆框架，核心创新是将记忆组织为两层图结构（base graph + hierarchical graph），并设计双路检索机制：System-1 基于 embedding 相似度做快速检索，System-2 通过 LLM 在层级图上做自顶向下的慎思遍历。两路结果通过 RRF 合并重排后输入 LLM 生成回答。该框架不涉及任何模型训练或参数优化，完全依赖 LLM prompting 和图工程设计。