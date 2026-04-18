# RQ1: What is Memory?

> **研究问题**：在 LLM Agent 系统中，"记忆"以什么形态存在？不同记忆表征对系统行为有什么影响？

## 分类总览

基于显式记忆（Explicit Memory）与隐式记忆（Implicit Memory）的总分法，将现有工作中的记忆表征分为六类：

|类别|记忆类型|表征名称|代表论文|
|---|---|---|---|
|T1|显式|纯文本非参数记忆|Memory-R1, Memento, Just in Time|
|T2|显式|多组件 / 多层级记忆|Mem-α, MemOS|
|T3|显式|固定长度压缩记忆|MemAgent, MemSearcher|
|T4|隐式|推理时记忆|MSA, MemGen|
|T5|隐式|参数化 / 模块化记忆|Knowledge Modules, Fine-tuning with RAG|
|T6|隐式|独立可插拔模型|Memory-R1 (memory manager), General Agentic Memory|

---

## 显式记忆（Explicit Memory）

显式记忆的核心特征：信息可被有意识地提取、查看和编辑。对应框架中的 $m_s$（短期记忆）和 $m_l$（长期记忆），其内容可明确指出"这条信息在这里"。

### T1：纯文本非参数记忆

> 代表：Memory-R1, Memento, Just in Time

**定义**：$m_l$ 是一个可 CRUD（Create / Read / Update / Delete）的文本条目集合，通常配合 embedding 检索 $v$ 使用，更新函数 $g_l$ 由 LLM 驱动。本质上是外挂式的 RAG 长期记忆。

**形式化**：

$$m_l = {(k_i, \text{text}_i, \text{meta}_i)}_{i=1}^{|m_l|}$$

$$v(q, m_l) = \text{top-}k(\text{sim}(E(q), E(k_i)))$$

**特点**：灵活、可解释、可编辑，但依赖 extraction quality。

---

### T2：多组件 / 多层级记忆

> 代表：Mem-α, MemOS

**定义**：$m_l$ 被拆分为多个子存储（如 core / semantic / episodic），每个子存储有不同的更新函数 $g_l$ 和检索机制 $v$。也包括图结构记忆（knowledge graph）、记忆胶囊（MemCube）等结构化表征。核心贡献在于 $g_l$ 中的路由策略，决定新信息应写入哪个子存储。

**形式化（以Mem-α为例）**：

$$m_l = (m_l^{\text{core}}, m_l^{\text{sem}}, m_l^{\text{epi}})$$

- $m_l^{\text{core}}$：始终在 context 中的核心摘要（always-in-context）
- $m_l^{\text{sem}}$：离散的事实性陈述
- $m_l^{\text{epi}}$：带时间戳的事件记录

$$g_l: (m_l^{(t)}, m_s^{(t)}, \text{trigger}) \rightarrow \text{route}(m_l^{\text{core}}, m_l^{\text{sem}}, m_l^{\text{epi}})$$

---

### T3：固定长度压缩记忆

> 代表：MemAgent, MemSearcher

**定义**：$m_s$ 被约束为固定长度的 token 序列，$g_s$ 在每一步对 $m_s$ 进行全量重写（而非增量追加）。$m_l$ 可能不存在或退化。本质上是对 context window 长度施加预算约束。

**形式化**：

$$|m_s| \leq L \quad (\text{e.g., } L = 1024 \text{ tokens})$$

$$g_s^{(t)}: m_s^{(t)} = \text{compress}(m_s^{(t-1)}, q^{(t)}, o^{(t)}) \quad \text{s.t. } |m_s^{(t)}| \leq L$$

---

## 隐式记忆（Implicit Memory）

隐式记忆的核心特征：知识内化在模型参数或模型行为中，不可直接查看或编辑，但隐式地影响所有输出行为。类比人类心理学中的内隐记忆，即无意识的、难以言说的、通过经验改变行为模式。

### T4：推理时记忆

> 代表：MSA, MemGen

**定义**：独立于模型权重之外的、在推理时被显式构造或检索出来的 latent space 对象，作为一个可分离的实体被注入到推理过程中。

**形式化**：

引入 $m_z$（latent memory）：

$$m_z^{(t)} = \pi_\theta(\text{encode} \mid m_s^{(t)}, C_N) \in \mathbb{R}^{d \times n}$$

$$C_N = f(q, m_s, v_{\text{diff}}(q, m_z), \text{other signals})$$

其中 $v_{\text{diff}}$ 是可微的注意力路由（如 top-k sparse attention），而非离散检索。

**特点**：不可解释、不可编辑；既不完全是外部数据库，也不完全是模型权重。

---

### T5：参数化 / 模块化记忆

> 代表：Knowledge Modules, Fine-tuning with RAG

**定义**：知识编码在模型权重中，可作为可插拔的参数模块。对应"将显式记忆内化为隐式记忆"的过程，即把 $C_N$ 中的知识固化到 $\theta$。

**形式化**：

$$\theta' = h(\theta, C_N) \quad \text{使得} \quad \pi_{\theta'}(\cdot \mid q) \approx \pi_\theta(\cdot \mid q, C_N)$$

**特点**：推理时无需检索（$v$ 退化），但更新成本高，可能遗忘。

---

### T6：独立可插拔模型

> 代表：Memory-R1 (memory manager), General Agentic Memory

**定义**：独立于主 LLM 之外的辅助模型，专门负责记忆的管理（读 / 写 / 更新 / 检索决策）。该模型拥有自己的参数，可以独立训练和替换，不修改主 LLM 的权重或 context 构造方式。T6 是一个独立的模型来管理记忆操作，主模型本身不变。

**特点**：模块化程度高，主模型与记忆管理解耦。

---

## 判定标准

为确保分类的可操作性，对每篇论文按以下规则判定：

1. **记忆是否可被直接查看/编辑？** - 是：显式；否：隐式
2. **显式记忆的数据结构？** - 纯文本集合(T1) / 多层级子存储(T2) / 固定长度 token 序列(T3)
3. **隐式记忆的载体？** - 推理时构造的 latent 向量(T4) / 主模型权重中的模块(T5) / 独立辅助模型(T6)

---

## 论文分类表

| Paper                          | T1 纯文本 | T2 多组件 | T3 固定长度压缩 | T4 推理时 | T5 参数化 | T6 可插拔模型 |
| ------------------------------ | ------ | ------ | --------- | ------ | ------ | -------- |
| Memory-R1                      | +      |        |           |        |        | +        |
| MemSearcher                    |        |        | +         |        | +      |          |
| General Agentic Memory         | +      |        |           |        |        | +        |
| Memento                        | +      |        |           |        |        |          |
| Just in Time                   | +      |        |           |        |        |          |
| Fine-tuning with RAG           | +      |        |           |        | +      |          |
| Knowledge Modules (DCD)        |        |        |           |        | +      |          |
| A-MEM                          |        | +      |           |        |        |          |
| MSA                            |        |        |           | +      |        |          |
| RLIF                           | -      | -      | -         | -      | -      | -        |
| RF-Mem                         | +      |        |           |        |        |          |
| GRU-Mem                        |        |        | +         |        |        |          |
| UMA                            | +      | +      |           |        |        |          |
| HyperRAG                       |        | +      |           |        |        |          |
| Mnemis                         |        | +      |           |        |        |          |
| MIRA                           |        | +      |           |        |        |          |
| Memory-Based Advantage Shaping |        | +      |           |        |        |          |
| EMPO                           | +      |        |           |        | +      |          |
| MemPO                          |        |        | +         |        |        |          |
| MemSifter                      | +      |        |           |        |        | +        |
| A-MAC                          | +      |        |           |        |        |          |
| CoMAM                          |        | +      |           |        |        |          |
| Memex(RL)                      | +      |        | +         |        |        |          |

> `+` = 适用  
> `—` = 此维度不适用（无记忆系统 / 无训练过程）  
> `!` = 无法归入现有类别，需扩展分类  