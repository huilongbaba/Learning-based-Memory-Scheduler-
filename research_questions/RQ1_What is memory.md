# RQ1: What is Memory?

> **研究问题**：在 LLM Agent 系统中，"记忆"以什么形态存在？不同记忆表征对系统行为有什么影响？

---

## 0. 记忆的定义（Definition）

**定义**：记忆（Memory）是**对过去信息的沉淀**，且**能够影响未来行为**的系统组成部分。

两个判定条件：

1. **源自过去（Origin）**：信息来自过去的外部数据（如 RAG 检索库）或过去的交互轨迹（对话历史、工具调用记录、推理中间态等）。
2. **影响未来（Effect）**：信息通过更新模型参数（$\theta \rightarrow \theta'$）、被注入上下文窗口（$C_N$）或以其他形式作用于后续推理（如 KV cache、latent memory 等），改变模型后续输出。

说明：

- 本定义为宏观筛选工具，帮助识别尚未被现有 T1–T5 分类覆盖的记忆相关技术。
- 当分类不清晰时，回归上述内容重新审视。

> 待持续完善

---

## 1. 记忆的性质维度（Properties Dimensions）

下列维度刻画记忆的性质，与 T1–T5 的分类正交：

### 1.1 显式 vs. 隐式（Explicitness）

- **显式**：信息可被有意识地提取、查看、编辑（对应 T1、T2）。
- **隐式**：信息内化在参数或 latent 空间中，不可直接查看或编辑（对应 T3–T5）。

### 1.2 生命周期（Lifecycle）

| 生命周期    | 含义                   |
| ------- | -------------------- |
| L-inf   | 仅在**推理阶段**存在，不改变模型参数 |
| L-train | 仅在**训练阶段**存在，不跨越到推理  |
| L-both  | 在训练与推理中**均**存在       |

---

## 2. 分类总览（T1–T5）

基于显式/隐式的总分法，将现有工作中的记忆表征分为五类：

| 类别  | 显/隐 | 表征名称                       | 代表论文                                               |
| --- | --- | -------------------------- | -------------------------------------------------- |
| T1  | 显式  | 非参数化外部记忆（flat / 多层级 / 图结构） | Memory-R1, Memento, Mem-α, MemOS, A-MEM, HyperRAG  |
| T2  | 显式  | 固定长度压缩记忆（context-window 级） | MemAgent, MemSearcher                              |
| T3  | 隐式  | 推理时 latent 记忆              | MSA, MemGen                                        |
| T4  | 隐式  | 参数化 / 模块化记忆                | Knowledge Modules, Fine-tuning with RAG, EMPO      |
| T5  | 隐式  | 独立可插拔辅助模块                  | Memory-R1 (memory manager), General Agentic Memory |

---

## 3. 显式记忆（Explicit Memory）

显式记忆的核心特征：信息可被有意识地提取、查看和编辑。对应框架中的 $m_s$（短期记忆）和 $m_l$（长期记忆），其内容可明确指出"这条信息在这里"。

### T1：非参数化外部记忆（Non-Parametric External Memory）

> 代表：Memory-R1, Memento, Just in Time, Mem-α, A-MEM, HyperRAG

**定义**：$m_l$ 是一个存放于主 LLM 之外的、以离散条目形式组织的可 CRUD（Create / Read / Update / Delete）记忆集合。条目不固化在模型权重中，通过检索机制 $v$ 被召回，通过更新函数 $g_l$ 被写入与维护。

**三个共性**：

1. **非参数化**：条目是离散可寻址对象（文本 / 结构化记录 / 图节点 / 边），而非模型权重的一部分。
2. **外部**：存储位置独立于主 LLM 之外，可被单独加载 / 替换 / 清空。
3. **可寻址**：存在一个检索机制 $v(q, m_l)$ 能以查询 $q$ 为输入召回相关条目。

**统一形式化**：

$$m_l = {e_i}_{i=1}^{|m_l|}, \quad e_i = (\text{key}_i, \text{content}_i, \text{meta}_i, \text{struct}_i)$$

$$v: (q, m_l) \rightarrow \text{top-}k(\text{score}(q, e_i))$$

$$g_l: (m_l^{(t)}, m_s^{(t)}, \text{trigger}) \rightarrow m_l^{(t+1)}$$

其中 $\text{struct}_i$ 是可选的结构信息（层级标签、图节点属性等），$\text{score}$ 可为 embedding 相似度、图距离、关键词匹配或其混合。

---

### T2：固定长度压缩记忆（Fixed-Length Compressed Memory）

> 代表：MemAgent, MemSearcher, MemPO

**定义**：$m_s$ 被约束为固定长度的 token 序列，$g_s$ 在每一步对 $m_s$ 进行全量重写（而非增量追加）。$m_l$ 可能不存在或退化。本质上是对 context window 长度施加预算约束。

**形式化**：

$$|m_s| \leq L \quad (\text{e.g., } L = 1024 \text{ tokens})$$

$$g_s^{(t)}: m_s^{(t)} = \text{compress}(m_s^{(t-1)}, q^{(t)}, o^{(t)}) \quad \text{s.t. } |m_s^{(t)}| \leq L$$

**与 T3 的关系**：T2 的压缩结果是显式 token 序列（可查看可理解）；若压缩结果落入 latent 空间（不可读向量），则归为 T3。两者共同描述"上下文窗口层面的记忆预算"，但分处显式 / 隐式两端。

---

## 4. 隐式记忆（Implicit Memory）

隐式记忆的核心特征：知识内化在模型参数或模型行为中，不可直接查看或编辑，但隐式地影响所有输出。类比人类心理学中的内隐记忆——无意识的、难以言说的、通过经验改变行为模式。

### T3：推理时 latent 记忆（Inference-Time Latent Memory）

> 代表：MSA, MemGen

**定义**：独立于模型权重之外的、在推理时被显式构造或检索出来的 **latent space** 对象。作为一个可分离的实体被注入到推理过程中，但本身不是 token，也不被重新训练到权重里。

**形式化**：

引入 $m_z$（latent memory）：

$$m_z^{(t)} = \pi_\theta(\text{encode} \mid m_s^{(t)}, C_N) \in \mathbb{R}^{d \times n}$$

$$C_N = f(q, m_s, v_{\text{diff}}(q, m_z), \text{other signals})$$

其中 $v_{\text{diff}}$ 是**可微的注意力路由**（如 top-k sparse attention），而非离散检索。

**特点**：不可解释、不可编辑；既不完全是外部数据库（不经 CRUD），也不完全是模型权重（不跨 episode 持久）。Titans 在此基础上进一步参数化 memory matrix $M$，与 T4 交界。

---

### T4：参数化 / 模块化记忆（Parametric / Modular Memory）

> 代表：Knowledge Modules, Fine-tuning with RAG

**定义**：知识编码在模型权重中。**既包含**针对主 LLM 的直接重训 / 全参数微调，**也包含**以 LoRA / adapter / knowledge module 等形式存在的可插拔参数模块。对应"将显式记忆内化为隐式记忆"的过程，即把 $C_N$ 中的知识固化到 $\theta$。

**形式化**：

$$\theta' = h(\theta, C_N) \quad \text{使得} \quad \pi_{\theta'}(\cdot \mid q) \approx \pi_\theta(\cdot \mid q, C_N)$$

**特点**：推理时无需检索（$v$ 退化），但更新成本高，可能遗忘；重训全参数与插入 adapter 是同一类别下的两种实现，区别在更新范围。

---

### T5：独立可插拔辅助模块（Pluggable Auxiliary Module）

> 代表：Memory-R1 (memory manager), General Agentic Memory

**定义**：独立于主 LLM 之外的辅助模块，专门负责记忆的管理（读 / 写 / 更新 / 检索决策）。该模块拥有自己的参数，可以独立训练和替换，不修改主 LLM 的权重或 context 构造方式。

**特点**：模块化程度高，主模型与记忆管理解耦；便于替换与多系统复用。


> "模块"（module）比"模型"（model）更准确，实现形式可以是小 LLM、MLP、gate 网络或规则引擎。T5 强调的是"独立可插拔"这一架构属性，而非实现细节。这个是20号会议中我想表达但没有很好地表达出来的内容。

---

## 5. 判定标准

为确保分类的可操作性，对每篇论文按以下规则判定：

1. **记忆是否可被直接查看 / 编辑？** - 是：显式（T1、T2）；否：隐式（T3–T5）。
2. **显式记忆的载体？** - 外部离散条目集合（T1） / 固定长度 token 序列（T2）。
3. **隐式记忆的载体？** - 推理时 latent 向量（T3） / 主模型权重或可插拔参数模块（T4） / 独立辅助模块（T5）。

---

## 6. 论文分类表

| Paper                          | T1 非参数外部 | T2 固定长度压缩 | T3 推理时 latent | T4 参数化 | T5 可插拔模块 |
| ------------------------------ | -------- | --------- | ------------- | ------ | -------- |
| Memory-R1                      | +        |           |               |        | +        |
| MemSearcher                    |          | +         |               | +      |          |
| General Agentic Memory         | +        |           |               |        | +        |
| Memento                        | +        |           |               |        |          |
| JitRL                          | +        |           |               |        |          |
| Fine-tuning with RAG           | +        |           |               | +      |          |
| Knowledge Modules (DCD)        |          |           |               | +      |          |
| A-MEM                          | +        |           |               |        |          |
| MSA                            | +        |           | +             |        |          |
| RLIF                           | —        | —         | —             | —      | —        |
| RF-Mem                         | +        |           |               |        |          |
| GRU-Mem                        |          | +         |               |        |          |
| UMA                            | +        |           |               |        |          |
| HyperRAG                       | +        |           |               |        |          |
| Mnemis                         | +        |           |               |        |          |
| MIRA                           | +        |           |               |        |          |
| Memory-Based Advantage Shaping | +        |           |               |        |          |
| EMPO                           | +        |           |               | +      |          |
| MemPO                          |          | +         |               |        |          |
| MemSifter                      | +        |           |               |        | +        |
| A-MAC                          | +        |           |               |        |          |
| CoMAM                          | +        |           |               |        |          |
| Memex (RL)                     | +        | +         |               |        |          |
| Titans                         |          |           | +             | +      |          |
| Mem-$\alpha$                   | +        | +         |               |        | +        |
| Memory as Action               | +        |           |               | +      |          |
| Scaling Context Folding        |          | +         |               | +      |          |
| MAGMA                          | +        |           |               |        |          |
| MACLA                          | +        |           |               |        |          |
| LightSearcher                  | +        |           |               | +      |          |
| MemVerse                       | +        | +         |               | +      | +        |

> `+` = 适用  
> `—` = 此维度不适用（无记忆系统 / 无训练过程）  
> `!` = 无法归入现有类别，需扩展分类  