# MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents

**Source:** [arXiv:2601.03236v2](https://arxiv.org/abs/2601.03236) (最后修订于 2026-04-16，ACL 2026 Main)

---
## 符号映射表

|论文原始符号|框架符号|含义说明|
|---|---|---|
|$q_t$ / $q$|$q$|用户查询 / human input|
|$o_t$|$o$|LLM 输出（观测）|
|$\mathcal{M}_t$|$m_l$|时变长期记忆（multi-graph + vector DB）|
|$\mathcal{G}_t = (\mathcal{N}_t, \mathcal{E}_t)$|$m_l$ 的具体数据结构|时变有向多重图，节点为事件，边为四种关系|
|$n_i = \langle c_i, \tau_i, v_i, A_i \rangle$|$e_i \in m_l$ 的条目|单个事件节点（内容、时间戳、向量、属性）|
|$\mathcal{E}_{\text{temp}}, \mathcal{E}_{\text{causal}}, \mathcal{E}_{\text{sem}}, \mathcal{E}_{\text{ent}}$|见 $\text{struct}_i$（T1 扩展）|四类关系子图：时间 / 因果 / 语义 / 实体|
|$\text{Retrieve}(q_t, \mathcal{M}_t)$|$v(q, m_l)$|检索算法（RRF 融合 + Beam Search）|
|$\text{Update}(\mathcal{M}_t, q_t, o_t)$|$g_l$|长期记忆更新函数（dual-stream：fast + slow path）|
|$\text{LLM}(\cdot)$|$M$ / $\pi_\theta$|主 LLM（gpt-4o-mini，冻结）|
|$C_{\text{prompt}}$|$C_N$|线性化后的最终 prompt 上下文|
|$T_q \in \lbrace \text{WHY, WHEN, ENTITY} \rbrace$|intent 路由信号|查询意图分类|
|$S(n_j \mid n_i, q)$|边转移评分函数|Beam Search 中的转移 score（结构对齐 + 语义相似）|
|$\Phi_{\text{reason}}$ / $\Phi_{\text{LLM}}$|$g_l$ 中的推理子模块|Slow path 中用于推断因果/实体边的 LLM|

---

## 概览

这篇文章用了用户的历史对话（跨 session 的长对话记忆），被切分成事件节点后构造成四个正交的关系图（语义图、时间图、因果图、实体图）+ 一个向量数据库。  
最后得到了一套 training-free 的 agentic memory 系统（即一个完全由启发式规则 + LLM 推断构成的 memory 读写与检索 pipeline）。主 LLM（gpt-4o-mini）全程冻结，不训练任何参数。  
这篇文章没有显式的优化目标函数，作者目标是在 LoCoMo 和 LongMemEval 这类长对话基准上，用更结构化的记忆表征，提升最终 QA 的准确率（LLM-as-judge）、降低 token 消耗和查询延迟。本文是一个架构贡献，不是一个学习/训练贡献。

---

## 1. Problem Setting

- **记忆类型**：cross-chat memory / 长期记忆 $m_l$（对应 Memory-Augmented Generation, MAG 范式）。本文无 $m_s$ 的显式建模；每次 query 时从 $m_l$ 中检索相关子图并线性化为 $C_N$ 作为一次性 context。
- **决策过程**：**未建模为 MDP/POMDP/bandit**。检索过程被刻画为一次"policy-guided graph traversal"（启发式 beam search），但这里的 "policy" 是启发式的意图路由权重向量 $\mathbf{w}_{T_q}$，**不是强化学习意义上的可训练 π**。
- **状态空间 𝒮、动作空间 𝒜、观测空间 Ω**：本文未按 RL 术语建模，因此无严格定义。可以近似地理解：
    - "状态" = 当前前沿节点集合 $\text{CurrentFrontier}$ + 已访问集合 $\text{Visited}$
    - "动作" = 选择某条边 $e_{ij}$ 从 $n_i$ 扩展到邻居 $n_j$
    - "观测" = 当前节点的内容 $c_i$、时间戳 $\tau_i$、向量 $v_i$
- **记忆数据结构**：**时变有向多重图 + 向量数据库的混合结构**
    - 节点：事件 $n_i = \langle c_i, \tau_i, v_i, A_i \rangle$
    - 边：四类关系子图（$\mathcal{E}_{\text{temp}}, \mathcal{E}_{\text{causal}}, \mathcal{E}_{\text{sem}}, \mathcal{E}_{\text{ent}}$）
    - 配套向量库用于 ANN 检索

**核心组件与符号映射**：

|框架组件|MAGMA 具体实现|
|---|---|
|$m_l$|$\mathcal{G}_t = (\mathcal{N}_t, \mathcal{E}_t)$ + Vector DB|
|$e_i = (\text{key}, \text{content}, \text{meta}, \text{struct})$|$n_i = \langle c_i, \tau_i, v_i, A_i \rangle$；$\text{struct}_i$ = 所属的四类子图边|
|$v(q, m_l)$|RRF 融合（dense + sparse + temporal）+ Heuristic Beam Search|
|$g_l$|Dual-stream：Fast path（synaptic ingestion）+ Slow path（async consolidation via LLM $\Phi$）|
|$C_N$|$C_{\text{prompt}}$ = 拓扑排序后的子图线性化结果|

---

## 2. Training Procedure

**本文无训练过程。** MAGMA 是一个完全 **training-free** 的系统：

- **优化对象**：无。主 LLM 参数 $\theta$ 全程冻结（使用 gpt-4o-mini 作为固定 backbone），没有任何可学习参数（无 adapter、无 Q-network、无 policy head）。
- **优化算法**：不适用。
- **训练数据**：不适用。
- **LLM 参数冻结？** 是，完全冻结。
- **训练目标函数**：无。

系统中所有"智能行为"都来自：

1. **手工设计的启发式**：RRF 融合公式、beam search 评分函数、意图分类器的权重向量。
2. **LLM 的 in-context 能力**：Slow path 中用 LLM 推断因果边和实体边（公式 8：$\mathcal{E}_{\text{new}} = \Phi_{\text{reason}}(\mathcal{N}(n_t), H_{\text{history}})$），但 $\Phi$ 本身不被训练，只是被 prompt。
3. **超参数调优**：附录 Table 6 列出的超参数（λ₁, λ₂, RRF k, beam width 等）是在 LoCoMo 上"empirically optimized"，属于手工 grid search，而非基于梯度的训练。

---

## 3. Reward Signal

**本文无 reward signal。** 因为没有 RL 训练过程：

- **奖励类型**：不适用。
- **奖励来源**：不适用。
- **Credit assignment**：不适用。
- **辅助奖励 / 正则项**：不适用。

评估时使用 **LLM-as-Judge** 对最终 QA 答案进行 [0, 1] 的语义打分（附录 C.3），但这只是 **evaluation metric**，不是训练信号。MAGMA 从未用 judge score 去更新任何参数或策略。

---

## 4. Inference Procedure

### 4.1 记忆初始化

推理开始时，$m_l = \mathcal{G}_t$ 已经通过 offline 的 Write/Update 过程构建完毕（Fast path 增量添加事件节点 + 时间骨干边；Slow path 异步由 LLM 推断出因果边和实体边）。向量库 VDB 存储所有节点的 dense embedding。

### 4.2 每步决策流程（四阶段 pipeline）

```
user query q
   ↓
[Stage 1] Query Analysis & Decomposition
   → 意图分类 T_q ∈ {WHY, WHEN, ENTITY}
   → 时间解析 [τ_s, τ_e]
   → dense embedding q⃗ + sparse keywords q_key
   ↓
[Stage 2] Multi-Signal Anchor Identification
   → 三路检索：Vector + Keyword + Time filter
   → RRF 融合得 anchor nodes S_anchor
   ↓
[Stage 3] Adaptive Traversal Policy (Heuristic Beam Search)
   → 从 S_anchor 出发，每步根据转移评分 S(n_j|n_i, q) 扩展 top-k
   → S 融合结构对齐 φ(type(e_ij), T_q) 和语义相似 sim(n⃗_j, q⃗)
   → 带 decay γ 控制深度
   ↓
[Stage 4] Narrative Synthesis via Graph Linearization
   → 拓扑排序（WHY → 因果排序；WHEN → 时间排序）
   → 结构化 scaffolding（带 timestamp + ref id）
   → Salience-based token budgeting
   ↓
C_prompt → frozen LLM → final answer
```

### 4.3 关键公式

**转移评分（结构 + 语义双信号）：**

$$S(n_j \mid n_i, q) = \exp\Big(\lambda_1 \cdot \phi(\text{type}(e_{ij}), T_q) + \lambda_2 \cdot \text{sim}(\vec{n}_j, \vec{q})\Big)$$

**意图路由权重（one-hot 加权）：**

$$\phi(r, T_q) = \mathbf{w}_{T_q}^{\top} \cdot \mathbf{1}_r$$

**RRF 融合锚点选择：**

$$S_{\text{anchor}} = \text{TopK}\Bigg(\sum_{m \in \lbrace \text{vec, key, time} \rbrace} \frac{1}{k + r_m(n)}\Bigg)$$

**线性化 context：**

$$C_{\text{prompt}} = \bigoplus_{n_i \in \text{Sort}(\mathcal{G}_{\text{sub}})} [\text{<t:}\tau_i\text{>}\ n_i.\text{content}\ \text{[ref:}n_i.\text{id](https://claude.ai/chat/1896a9e4-2834-46b6-bf68-8d275f1091ee)}]$$

### 4.4 推理时额外策略

- **TopK 检索**：每步 beam 保留 BeamWidth 个最高分节点。
- **温度调节**：LLM 生成 temperature = 0.0（附录 Table 6）。
- **Budget 约束**：超出 token budget 时低分节点被 summarize 成 brevity code（"...3 intermediate events..."）。
- **Depth decay**：$\text{score}_v \leftarrow \text{score}_u \cdot \gamma + s_{uv}$ 控制随深度衰减。
- **多轮 replan**：无。单次检索即返回。

### 4.5 推理策略来源

**完全由手工规则驱动**，没有学到的 $\pi$：

- 意图分类器是 lightweight classifier（论文未详述，实际上是一个基于 prompt 的 LLM 调用 + 规则）
- 权重向量 $\mathbf{w}_{T_q}$、超参 $\lambda_1, \lambda_2$、阈值 $\delta, \theta_{\text{sim}}$、beam width 均为人工设定
- 拓扑排序规则、salience budgeting 规则均为硬编码

> 所有决策逻辑都必须由设计者提前想好并编码为规则。如果场景迁移到新领域，这些超参需要重新调优。

---

## 5. RQ 分析

### RQ1 (What is memory?)

MAGMA 的记忆是显式、非参数化的外部多重图：四个正交关系子图（semantic / temporal / causal / entity）+ 向量库。每个记忆条目是一个事件节点 $n_i = \langle c_i, \tau_i, v_i, A_i \rangle$，可以被独立寻址、CRUD、查看。明确对应 RQ1 分类中的 T1（非参数化外部记忆，多层级/图结构）。

### RQ2 (Which component is optimized? Which signal is used?)

**本文未涉及此问题。** MAGMA 是完全 training-free 的系统，主 LLM 冻结。

### RQ3 (Target of Optimization)

本文未涉及 learning-based 的优化目标，但从实验指标看，系统设计上追求 G1（回答准确度，LLM-as-judge + F1 + BLEU-1） 和 G2（效率：token/query、latency、build time） 的联合改善。

### RQ4 (How memory evolves, operates?)

本文未涉及此问题。 归入 N/A: Non-RL Methods。

### **RQ5 (How memory evolves, operates?)**： 

记忆通过 dual-stream 机制演化：  
Fast path：对话到达时立即做事件分段、向量编码、更新时间骨干边。
Slow path：后台 worker 从队列取节点，用 LLM 推断因果边和实体边，加密图结构。
读取时通过 4 阶段 pipeline 完成（意图分类 - 锚点识别 - beam search - 线性化）。

---

## Conclusion

MAGMA 是一个完全 training-free 的 agentic memory 系统，它的核心贡献在于把"记忆是什么"和"怎么检索记忆"两件事解耦：记忆被组织成四个正交的关系子图（语义、时间、因果、实体），而检索变成一个由查询意图引导的图遍历过程。文章用一套手工设计的启发式（RRF 融合 + 意图路由 + beam search + 拓扑线性化）在 LoCoMo 和 LongMemEval 上超过了 A-MEM、Nemori、MemoryOS 等更"学习驱动"的 baseline，同时大幅降低 token 消耗和延迟。这篇文章在方法论上属于架构方向，它证明了在当前 LLM 能力下，一个精心设计的图结构 + 意图感知检索足以解决大量长对话记忆问题，但也意味着所有策略都必须依赖人工经验和超参数调优，缺乏跨域自动适应能力。