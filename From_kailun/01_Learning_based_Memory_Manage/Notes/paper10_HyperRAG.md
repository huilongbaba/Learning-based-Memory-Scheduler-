# HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation

**Source:** [https://arxiv.org/pdf/2602.14470v1](https://arxiv.org/pdf/2602.14470v1) (Feb 2026, WWW '26)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$q$|$q$|用户查询 / 问题|
|$G = (E, R, F)$|$m_l$|外部超图知识库，作为长期记忆|
|$f^n$|$m_l$ 中的条目|一条 n-ary 关系事实（超边）|
|$\varphi(\cdot)$|$E(\cdot)$（embedding）|文本 embedding 函数|
|$f_\theta$ (MLP classifier)|$v$（检索算法）|HyperRetriever 的可训练打分器，决定检索哪些超边|
|LLM scoring ($S_F, S_E$)|$v$（检索算法）|HyperMemory 中 LLM 参数记忆引导的检索评分|
|Context|$C_N$|组装后输入 LLM 的上下文|
|Answer|$A_n$|最终生成的答案|
|$\tau$ (plausibility threshold)|无直接对应|自适应搜索的阈值超参数|
|$w, d$ (beam width/depth)|无直接对应|HyperMemory beam search 超参数|

> 论文中的超图 $G$ 对应 $m_l$（长期外部记忆）；MLP 打分器 $f_\theta$ 和 LLM-guided scoring 均对应检索算法 $v$；

---

## 概览

这篇文章用 n-ary 超图（hypergraph）作为外部知识库，将多实体关系编码为超边（hyperedge），并基于查询从超图中提取关系链（relational chain）作为证据。  
最后得到了两个检索模块：(1) HyperRetriever，一个可训练的 MLP 打分器，能从超图中自适应地提取与查询相关的 n-ary 关系链；(2) HyperMemory，一个利用 LLM 参数记忆引导 beam search 的检索模块。  
优化了最终回答的准确度 G1（包括EM、F1、MRR、Hits@10），和检索效率 G2（通过 n-ary 超图结构降低了检索路径深度）。

---

## 1. Problem Setting

- **记忆类型**：纯粹的 cross-query 外部知识库（$m_l$），不涉及 in-chat memory 或 $m_s$。超图在推理前已预构建完毕，推理时只读不写。
- **决策过程建模**：未显式建模为 MDP/POMDP/bandit。HyperRetriever 将检索建模为逐跳（hop-by-hop）的图遍历与剪枝决策；HyperMemory 则通过 beam search 进行路径扩展，但两者均非 RL 框架下的序贯决策。
- **状态空间 $\mathcal{S}$**：当前已检索到的子图（实体集合 + 超边集合 + 前沿实体 frontier entities）。
- **动作空间 $\mathcal{A}$**：选择下一跳中保留哪些 pseudo-binary triples（HyperRetriever：阈值筛选）或选择 top-$w$ 路径（HyperMemory：beam scoring）。
- **观测空间 $\Omega$**：每一跳中候选超边和实体的 embedding 表示及其打分。
- **记忆数据结构**：n-ary 超图 $G = (E, R, F)$，每条超边 $f^n = \lbrace e_i \rbrace_{i=1}^{n}$ 绑定多个实体和关系。

|核心组件|框架符号|论文对应|
|---|---|---|
|长期记忆|$m_l$|超图 $G = (E, R, F)$|
|检索算法|$v$|MLP plausibility scorer $f_\theta$ / LLM-guided beam scorer|
|上下文|$C_N$|Budget-aware contextualized evidence|
|模型|$M$|gpt-4o-mini（冻结，仅用于生成）|
|最终答案|$A_n$|Answer|

> 本文的"记忆"本质上是一个静态的外部知识超图，不涉及记忆的写入或更新操作。

---

## 2. Training Procedure

- **优化的组件**：仅优化 HyperRetriever 中的 MLP 打分器 $f_\theta$（即检索算法 $v$）。LLM（gpt-4o-mini）参数完全冻结，HyperMemory 无需训练（zero-shot 使用 LLM 参数记忆）。
- **优化算法**：监督学习（Supervised Learning），使用 Binary Cross-Entropy Loss。
- **训练数据来源**：离线构建——基于超图中 topic entity 到正确答案的最短路径自动生成正负样本。正样本为最短路径上的 triples，负样本为同一 hop 上其他不在最短路径中的 triples。
- **是否冻结 LLM**：是，LLM 完全冻结。优化的参数是 MLP classifier $f_\theta$ 的权重。

**核心训练目标函数：**

论文原始公式：

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log f_\theta(\mathbf{x}_i) + (1 - y_i) \log(1 - f_\theta(\mathbf{x}_i)) \right]$$

统一符号标注：

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log v_\theta(\mathbf{x}_i) + (1 - y_i) \log(1 - v_\theta(\mathbf{x}_i)) \right]$$

其中 $v_\theta = f_\theta$ 是检索打分器，$\mathbf{x}_i = [\varphi(q) \Vert \varphi(e_h) \Vert \varphi(f^n) \Vert \varphi(e_t) \Vert \delta(e_h, f^n, e_t)]$ 是 semantic + structural 特征的拼接，$y_i \in \lbrace 0, 1 \rbrace$ 表示该 triple 是否在最短路径上。

> 训练目标是二分类监督学习。

---

## 3. Reward Signal

- **奖励类型**：本文不使用 RL reward signal。使用的是监督学习中的 ground-truth label（最短路径上的 triple 为正、其余为负）。
- **奖励来源**：离线标注——通过超图中 topic entity 到 ground-truth answer 的最短路径自动产生标签。
- **奖励分配**：不适用（非 RL 框架）。
- **辅助奖励或正则项**：无显式提及。自适应阈值策略（adaptive thresholding）和密度感知策略（density-aware policy）作为推理时的启发式规则存在，但不参与训练。

> 严格来说，本文没有 reward signal，因为它不是 RL 方法。训练信号来自基于图结构的自动标注，属于 supervised signal。

---

## 4. Inference Procedure

- **记忆初始化**：超图 $G$ 在推理前已预构建。推理开始时，通过 LLM 提取查询 $q$ 中的 topic entities $E_q$。
    
- **HyperRetriever 决策流程**：
    
    1. **Topic Entity Extraction**：LLM 从 $q$ 中提取实体 $E_q$
    2. **Hyperedge Retrieval**：获取 $E_q$ 的邻接超边 $F_{e_s}$，展开为 pseudo-binary triples $T_q$
    3. **Structural Encoding**：计算双向距离编码 $\delta(e_h, f^n, e_t)$
    4. **Plausibility Scoring**：MLP $f_\theta$ 对每个 triple 打分
    5. **Adaptive Search**：保留得分超过阈值 $\tau$ 的 triples，将其尾实体作为下一跳 frontier，迭代扩展直至无新 triple 满足阈值
    6. **Context Assembly**：按 token 预算（超边 50%、实体 30%、文本块 20%）组装 $C_N$
    7. **Generation**：$A_n = \text{LLM}(C_N, q)$
- **HyperMemory 决策流程**：
    
    1. 同上提取 topic entities
    2. LLM 对每个候选超边打分 $S_F$，保留 top-$w$
    3. LLM 对候选尾实体打分 $S_E$，计算复合分 $S = S_F \cdot S_E$
    4. 保留 top-$w$ 路径，检查证据充分性（LLM 判断 yes/no）
    5. 若不充分且深度 $< d$，继续扩展
    6. 组装 Context 并生成答案
- **推理策略**：HyperRetriever 的核心策略（MLP 打分 + 自适应阈值）由学习得到的 $f_\theta$ 驱动，但阈值调整规则（公式 11、12）为手工设计。HyperMemory 完全依赖 LLM 的零样本打分能力 + 手工设计的 beam search 框架。
    

> HyperRetriever 结合了学习（MLP）和手工规则（adaptive threshold），是一种典型的 hybrid 推理策略。HyperMemory 则完全是 prompt-based，无需训练但推理成本更高（每步需多次 LLM 调用）。

---

## 5. RQ 分析

### RQ1 (What is memory?)

本文的"记忆"是一个预构建的 n-ary 超图知识库 $G = (E, R, F)$，属于 $m_l$（长期外部记忆）。其数据结构不是传统的文本条目集合或 KV 对，而是超图中的超边（hyperedge），每条超边绑定多个实体和关系。记忆在推理过程中只读不写，不涉及记忆更新。

### RQ2 (Which component is optimized? Which signal is used?)

优化的是 HyperRetriever 中的 MLP 检索打分器 $f_\theta$（对应检索算法 $v$），使用 Binary Cross-Entropy 监督信号进行训练。LLM 参数完全冻结。训练信号来自基于超图最短路径的自动标注，而非 RL reward。

### RQ3 (Target of Optimization)

最终优化目标是回答准确度 G1：EM、F1（open-domain QA）和 MRR、Hits@10（closed-domain QA）。同时，论文关注检索效率 G2，通过 n-ary 超图的浅层推理链降低检索时间和 token 消耗。

### RQ4 (How memory evolves, operates?)

记忆（超图）在运行时不演化——推理前预构建，推理时只读。推理时的操作流程是：提取 topic entity → 逐跳检索邻接超边 → MLP 打分剪枝（或 LLM beam scoring）→ 组装 context → LLM 生成答案。记忆没有写入、更新或删除操作。

---

## Conclusion

HyperRAG 提出了一个基于 n-ary 超图的 RAG 框架，解决了传统二元知识图谱在多跳推理中面临的语义碎片化和路径爆炸问题。框架包含两个检索变体：HyperRetriever 通过可训练的 MLP 打分器在超图上进行自适应搜索，提取与查询相关的 n-ary 关系链；HyperMemory 则利用 LLM 的参数记忆引导 beam search。我认为该工作的核心贡献在于将 graph-based RAG 从二元关系扩展到 n-ary 超图，实现了更浅、更语义完整的推理链。从记忆系统的角度看，本文关注的是如何更好地从静态外部记忆中检索，而不是如何管理和更新记忆。