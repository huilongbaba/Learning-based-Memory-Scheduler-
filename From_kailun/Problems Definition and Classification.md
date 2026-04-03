# Notation & Definitions

## Core Interaction

* **q**: question / human input
* **t**: thought (reasoning trace)
* **a**: action
* **o**: observation
* **C**: context

---

## Memory

* **mₛ**: short-term memory (working memory)
* **mₗ**: long-term memory (external memory)
* **gₛ​​**: short-term memory update function (working memory refresh)
* **gₗ**: long-term memory update function (memory consolidation & forgetting)

---

## System Components

* **v**: retrieval algorithm
* **R**: RAG system
* **M**: model

---

## Learning & Policy

* **T**: trajectory
* **θ**: model parameters
* **π**: policy of the model

---

## Output

* **Aₙ**: final answer


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

1. 提高输出准确性 — 让 $\hat{a}$ 更接近正确答案
2. 减少 Agent 思考轮数 — 降低 $N$（Agent loop 的迭代次数）

### 统一视角

Memory 决定了哪些信息有资格被放入 $C_N$，当前 Memory 的研究方向是这个函数 $f$ 和检索机制 $v$ 的设计：
1. 怎么存（Write）：写入策略，决定什么信息、以什么形式写入 $m_s$ 和 $m_l$
2. 怎么管（Manage）：记忆的更新、合并、遗忘、整理
3. 怎么取（Read）：检索机制 $v$，决定什么时候取、取什么、取多少
4. 怎么用（Assemble）：函数 $f$，决定取出来的信息怎么组织到 $C_N$ 里
Memory 研究的目标是让 $C_N$​ 包含最优的信息组合，或者可以说在优化信息从 $m_s$​ 和 $m_l$ 流向 $C_N$​ 的整个 pipeline。

心理学上将人类的记忆分为外显记忆和内隐记忆：
内隐记忆的核心特征是无意识的、难以言说的、通过经验改变行为模式。$\pi_\theta$ 不也是这样吗？模型通过大量训练数据学到的 pattern 都内化在参数里，你无法直接查看某条知识存在哪个参数中，但它确实影响了输出行为。

外显记忆的核心特征是可以有意识地提取和陈述的，$C_N$ 不也是这样吗？检索回来的文档、对话历史、用户画像，都是可以明确指出这条信息在这里的。

---

## Tschibo思路：按数据 Setting 划分

### Setting A：提供离线历史数据（Offline Training）

训练时提供历史数据：

- 对话场景：历史 dialog + 正确回答
- Agent 场景：历史 episode（标注成功/失败）

基于这些历史数据进行 offline training。

---

#### 方法 A1：以Memory R1为代表

> 论文：[Memory R1](https://arxiv.org/abs/2508.19828)

**核心思想**：将 context window 替换为基于检索的方式构建，对话轮数固定为 $N=1$。

**流程**：

1. 给定问题 $q$，通过 embedding 模型 $E$ 对历史数据 $R$ 进行编码
2. 通过 memory manager $\gamma$ 从 $E(R)$ 中检索相关信息，构建 context

**形式化**：

$$C = \gamma(q, E(R)), \quad N = 1$$

**强化学习目标**：

- 第一步：强化 $\pi_\theta$ 和 $E(R)$（联合优化 policy 和 embedding）
- 第二步：用收敛后的 $a_N$ 进一步优化

特点：单轮检索，不需要多轮交互，优化重点在 retrieval quality。

---

#### 方法 A2：以MemSearcher为代表

> 论文：[MemSearcher](https://arxiv.org/abs/2511.02805)

**核心思想**：允许多轮对话交互，每一轮的 context window 由 policy 动态生成。

**流程**：

1. 在第 $N$ 轮，模型根据上一轮 context $C_{N-1}$ 生成 action $a_N$
2. 结合 action、对 $R$ 的查询结果和观测，构建新的 $C_N$

**形式化**：

$$C_N \sim \pi_\theta(a_N, q(R), C_N \mid q, C_{N-1})$$

**强化学习目标**：

- 第一步：强化 $\pi_\theta$，使其更准确地生成每一轮的 $C_N$
- 第二步：用收敛后的 $q$ 和 $a_N$ 进一步优化

**特点**：多轮交互式检索，优化重点在 context 的迭代构建过程。

---

### Setting B：提供实时数据（Online / JIT）

不再提供历史数据，而是实时获取数据。每个 episode 实时收集，即时优化 policy。

---

#### 方法 B1：以JIT (Just-In-Time)为代表

> 论文：[JIT](https://arxiv.org/abs/2601.18510)

**核心思想**：因为优化速度足够快，可以在线实时强化 policy。

**强化学习目标**：

- **第一步**：强化 $\pi_\theta$（因优化速度快，可视为 online learning）
- **第二步**：用实时收集的 $q$，通过生成 $r$ 的方式进行优化

**特点**：无需预先积累历史数据，实时优化，适合动态变化的场景。

---
## Kailun思路：按人类记忆模式划分

### 优化显式记忆 ($C_N$)：

**核心思想**：优化怎么存、怎么管、怎么取、怎么组装

### 优化隐式记忆 ($\pi_\theta$):

**核心思想**：在预训练、SFT、RLHF 等阶段通过微调来内化经验

### 优化显式到隐式的过程（内化）

> 论文：[Fine-tuning with RAG](https://arxiv.org/abs/2510.01375)

**核心思想**：把外显记忆内化为内隐记忆，即把 $C_N$ 中的知识固化到 $\theta$

$$\theta' = h(\theta, C_N) \quad \text{使得} \quad \pi_{\theta'}(\cdot \mid q) \approx \pi_\theta(\cdot \mid q, C_N)$$

即通过某种更新函数 $h$ （未定义），将 $C_N$​ 中的信息压缩进参数 $\theta$，使得不再需要 $C_N$​ 也能得到类似的输出

### 优化隐式到显式的过程（外化）

> 论文：[A-MEM](https://arxiv.org/abs/2502.12110)

**核心思想**：模型利用自身能力主动生成、整理、增强外部记忆

$$m_l^{(t+1)} = g_l(m_l^{(t)}, \pi_\theta(\text{reflect} \mid m_s^{(t)}, C_N))$$

即模型用 $\pi_\theta​$ 对当前短期记忆 $m_s$ 和 context $C_N$ 进行反思，总结和提炼，然后将结果写入长期记忆  $m_l$。