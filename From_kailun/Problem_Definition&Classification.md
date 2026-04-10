# Notation & Definitions

## Core Interaction

- **q**: question / human input
- **t**: thought (reasoning trace)
- **a**: action
- **o**: observation
- **C**: context

---

## Memory

- **mₛ**: short-term memory (working memory)
- **mₗ**: long-term memory (external memory)
- **gₛ​​**: short-term memory update function (working memory refresh)
- **gₗ**: long-term memory update function (memory consolidation & forgetting)

---

## System Components

- **v**: retrieval algorithm
- **R**: RAG system
- **M**: model

---

## Learning & Policy

- **T**: trajectory
- **θ**: model parameters
- **π**: policy of the model

---

## Output

- **Aₙ**: final answer

---

## 核心框架

给定数据库 $R$ 和问题 $q$，模型在 context window $C$ 下，按照 policy $\pi_\theta$ 输出一个分布：

$$\hat{a} = \pi_\theta(\cdot \mid q, C)$$

其中 context window $C$ 可以是：

- 单轮场景：一轮对话产生的 context window
- 多轮 Agent 场景：经过多轮交互后，最终一轮的 context window $C_N$

在多轮场景下，表达为：

$$\hat{a}_N \sim \pi_\theta(\cdot \mid q, C_N)$$

其中：

$$C_N = f(q, m_s, v(q, m_l), \text{other signals})$$

另外：

$$m_s^{(t)} = g_s(m_s^{(t-1)}, q^{(t)}, o^{(t)})$$

$$m_l^{(t)} = g_l(m_l^{(t-1)}, m_s^{(t)}, \text{trigger})$$

### 核心优化目标

提高输出准确性 — 让 $\hat{a}$ 更接近正确答案

### 统一视角

显式记忆视角：Memory 决定了哪些信息有资格被放入 $C_N$，当前 Memory 的研究方向是这个函数 $f$ 和检索机制 $v$ 的设计：

1. 怎么存（Write）：写入策略，决定什么信息、以什么形式写入 $m_s$ 和 $m_l$
2. 怎么管（Manage）：记忆的更新、合并、遗忘、整理
3. 怎么取（Read）：检索机制 $v$，决定什么时候取、取什么、取多少
4. 怎么用（Assemble）：函数 $f$，决定取出来的信息怎么组织到 $C_N$ 里 Memory 研究的目标是让 $C_N$​ 包含最优的信息组合，或者可以说在优化信息从 $m_s$​ 和 $m_l$ 流向 $C_N$​ 的整个 pipeline。

隐式记忆视角：知识内化在模型参数 $\pi_\theta$​ 中，不可直接查看或编辑，但隐式地影响所有输出行为。 Memory 研究的目标是最大可能的提高 $\pi_\theta$ 的能力​，预训练、SFT、RLHF、推理时 scaling (CoT)、PEFT (LoRA)

心理学上将人类的记忆分为外显记忆和内隐记忆： 内隐记忆的核心特征是无意识的、难以言说的、通过经验改变行为模式。$\pi_\theta$ 不也是这样吗？模型通过大量训练数据学到的 pattern 都内化在参数里，你无法直接查看某条知识存在哪个参数中，但它确实影响了输出行为。

外显记忆的核心特征是可以有意识地提取和陈述的，$C_N$ 不也是这样吗？检索回来的文档、对话历史、用户画像，都是可以明确指出这条信息在这里的。

---

## 思路：按人类记忆模式划分

记忆以什么形态存在？用什么方法优化？

### 优化显式记忆 ($C_N$)：

**核心思想**：优化 $C_N$ 怎么存、怎么管、怎么取、怎么组装

对应坐标 **(a, *)**：横轴为优化显式记忆管理，纵轴可以是任意学习方法。

---

### 优化隐式记忆 ($\pi_\theta$):

**核心思想**：在预训练、SFT、RLHF 等阶段通过微调来内化经验

对应坐标 **(b, *)**：横轴为优化隐式记忆，纵轴可以是任意学习方法。

---

### 优化显式到隐式的过程（内化）

> 论文：[Fine-tuning with RAG](https://arxiv.org/abs/2510.01375)，[Knowledge Modules](https://arxiv.org/abs/2503.08727)

**核心思想**：把外显记忆内化为内隐记忆，即把 $C_N$ 中的知识固化到 $\theta$

**形式化**：

$$\theta' = h(\theta, C_N) \quad \text{使得} \quad \pi_{\theta'}(\cdot \mid q) \approx \pi_\theta(\cdot \mid q, C_N)$$

即通过某种更新函数 $h$ （未定义），将 $C_N$​ 中的信息压缩进参数 $\theta$，使得不再需要 $C_N$​ 也能得到类似的输出。

---

### 优化隐式到显式的过程（外化）

> 论文：[A-MEM](https://arxiv.org/abs/2502.12110)

**核心思想**：模型利用自身能力主动生成、整理、增强外部记忆

**形式化**：

$$m_l^{(t+1)} = g_l(m_l^{(t)}, \pi_\theta(\text{reflect} \mid m_s^{(t)}, C_N))$$

即模型用 $\pi_\theta​$ 对当前短期记忆 $m_s$ 和 context $C_N$ 进行反思，总结和提炼，然后将结果写入长期记忆 $m_l$。


---

### 记忆表征分类（RQ1：$m_l$ 和 $m_s$ 的数据结构是什么？）

在核心框架

$$C_N = f(q, m_s, v(q, m_l), \text{other signals})$$ 
中，不同论文对 $m_l$ 和 $m_s$ 的实例化方式截然不同：

---

#### 表征 T1：纯文本非参记忆

> 代表：Memory-R1

**定义**：$m_l$ 是一个可 CRUD 的文本条目集合，$v$ 是 embedding 检索，$g_l$ 是 LLM 驱动的 add/update/delete。

**框架作用点**：完全作用于 $f$ 和 $v$ 的设计——不碰 $\theta$。

**形式化**：

$$m_l = {(k_i, \text{text}_i, \text{meta}_i)}_{i=1}^{|m_l|}$$

$$v(q, m_l) = \text{top-}k(\text{sim}(E(q), E(k_i)))$$

**特点**：灵活、可解释、可编辑，但依赖 extraction quality。

---

#### 表征 T2：多组件多层级记忆

> 代表：Mem-α, MemOS

**定义**：$m_l$ 被拆成多个子存储，每个有不同的 $g_l$ 和 $v$。$m_s$ 也可能有显式层级。模型需要学会判断一条新信息应该写入哪个组件。图结构记忆，MemCube等

**框架作用点**：$g_l$ 中的路由策略——决定信息流向哪个子存储——是核心贡献。

**形式化**：

$$m_l = (m_l^{\text{core}}, m_l^{\text{sem}}, m_l^{\text{epi}})$$

- $m_l^{\text{core}}$：始终在 context 中的核心摘要（always-in-context）
- $m_l^{\text{sem}}$：离散的事实性陈述
- $m_l^{\text{epi}}$：带时间戳的事件记录

$$g_l: (m_l^{(t)}, m_s^{(t)}, \text{trigger}) \rightarrow \text{route}(m_l^{\text{core}}, m_l^{\text{sem}}, m_l^{\text{epi}})$$

**特点**：认知科学启发，路由策略是关键难点。

---

#### 表征 T3：固定长度压缩记忆

> 代表：MemAgent, MemSearcher

**定义**：$m_s$ 被约束为固定长度的 token 序列，$g_s$ 在每一步完全重写 $m_s$。$m_l$ 可能不存在或退化。即固定上下文窗口长度。

**框架作用点**：本质上是约束 $C_N$ 的大小，优化 $g_s$ 的压缩质量。

**形式化**：

$$|m_s| \leq L \quad (\text{e.g., } L = 1024 \text{ tokens})$$

$$g_s^{(t)}: m_s^{(t)} = \text{compress}(m_s^{(t-1)}, q^{(t)}, o^{(t)}) \quad \text{s.t. } |m_s^{(t)}| \leq L$$

**特点**：计算开销 $O(1)$，牺牲细节换严格计算边界。Memory budget 约束 $|m_s| \leq L$ 深刻影响 $g_s$ 的设计——从增量更新变成全量重写。

---

#### 表征 T4：推理时记忆

> 代表：MSA，MemGen（？）

**定义**：独立于模型权重之外的、在推理时被显式构造或检索出来的 latent space 对象，作为一个可分离的实体被注入到推理过程中

**框架作用点**：模糊了 $m_l$（显式）和 $\theta$（隐式）的边界，既不完全是外部数据库，也不完全是模型权重。

**形式化**：

引入 $m_z$（latent memory）：

$$m_z^{(t)} = \pi_\theta(\text{encode} \mid m_s^{(t)}, C_N) \in \mathbb{R}^{d \times n}$$

$$C_N = f(q, m_s, v_{\text{diff}}(q, m_z), \text{other signals})$$

其中 $v_{\text{diff}}$ 是可微的注意力路由（e.g., top-k sparse attention），而非离散检索。

**特点**：不可解释、不可编辑。

---

#### 表征 T5：参数化模块化记忆

> 代表：Knowledge Modules, Fine-tuning with RAG

**定义**：知识编码在模型权重中，可作为可插拔模块。

**特点**：推理时无需检索（$v$ 退化），但更新成本高，可能遗忘。

---

### 训练信号分类（Non-QA Training Paradigm 的出发点）

训练信号作为论文属性标签而非分类维度，因为同一种优化方法可以搭配不同的信号。

已发现训练信号类型：

| 训练信号类型                | 代表论文                             | 信号来源           | 局限性                               |
| --------------------- | -------------------------------- | -------------- | --------------------------------- |
| QA accuracy           | Memory-R1, MemAgent, MemSearcher | 下游答案正确性        | 无法衡量偏好、人格一致性、时序推理                 |
| QA + compression      | Mem-α                            | 答案正确 + 惩罚冗余存储  | 仍以 QA 为主信号                        |
| Self-certainty        | Intuitor (ICLR 2026)             | 模型自身对答案的置信度    | 可能放大错误的高置信                        |
| Hidden state matching | Knowledge Modules                | Teacher 模型隐藏状态 | 需要 teacher，不适用于在线场景               |
| Random / spurious     | Spurious Rewards                 | 完全随机           | 某些模型上效果接近真实信号（GRPO clipping bias） |

研究问题：Spurious Rewards 论文表明，Qwen2.5-Math-7B 使用完全随机奖励在 MATH-500 上提升 21.4%，接近真实奖励的 29.1%。这意味着我们甚至不确定目前的 QA 训练信号到底在教模型什么。它可能只是在激活预训练中已有的 pattern（GRPO 的 clipping bias），而非真正学习记忆管理策略。