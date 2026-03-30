# AtomMem: Learnable Dynamic Agentic Memory with Atomic Memory Operation
> source: https://arxiv.org/pdf/2601.08323
> From_huilong\2601.08323v3.pdf

## Problem Setting

现有 agent 记忆机制依赖**静态手工流程**（如固定遗忘调度、每步强制摘要），one-size-fits-all，无法适配不同任务。

本文将 mₗ 管理建模为 **POMDP**  (𝒮, 𝒜, P, Ω, 𝒪, ℛ, γ)，将高层记忆流程解构为**原子 CRUD 操作**，通过 RL 学习动态记忆策略。

| 组件 | 符号 | 定义 |
|---|---|---|
| 全局状态 | sₜ = (sₜᵉⁿᵛ, sₜᵐᵉᵐ) | 外部环境 + 内部记忆状态 |
| 动作 | aₜ = (aₜᵉⁿᵛ, aₜᵐᵉᵐ) | 任务动作（如 search）+ 记忆操作 |
| 记忆动作空间 | 𝒜ᵐᵉᵐ = {Create, Read, Update, Delete} | 原子 CRUD |
| 观测 | oₜ = {oₜᵉⁿᵛ, mₜˢᶜʳ, ℳ̂ₜ} | 环境观测 + scratchpad + 检索结果 |

记忆为动态集合 ℳₜ = {mᵢ}，初始 s₀ᵐᵉᵐ = ∅（每个独立任务重置，区别于跨任务经验积累型记忆）。

- **mₛ**：scratchpad（mₜˢᶜʳ），每步确定性检索，维护全局任务状态摘要
- **mₗ**：FAISS 向量数据库，通过 CRUD 操作管理

每步 agent 生成一组记忆动作序列 𝒜ₜ = {aₜ¹, …, aₜᴷᵗ}，非 Read 操作顺序执行：

$$\mathcal{M}_{t+1} = a_t^{K_t} \circ \cdots \circ a_t^1(\mathcal{M}_t)$$

**混合检索**：Read 不改变记忆状态，采用双通道检索——scratchpad（每步必取）+ 选择性语义检索：

$$\hat{\mathcal{M}}_t = \text{TopK}(\{m_i \in \mathcal{M}_t \mid \text{sim}(q_{t-1}, m_i)\})$$

CRUD 原子性的三个性质：**完备性**（任意记忆状态可达）、**最小性**（不可再分解）、**任务无关性**（泛化能力依赖 LLM 决策而非操作设计）。

## Training Procedure

用 **GRPO** 端到端优化，记忆操作作为结构化 token 纳入输出序列。

- T = (o₁, a₁, …, o_T, a_T)，aₜ 包含 aₜᵉⁿᵛ 和 aₜᵐᵉᵐ
- 仅使用**任务级终端奖励**（EM / LLM-as-judge），无中间奖励
- 优势函数（Dr.GRPO，不归一化）：

$$A_i = r_i - \frac{1}{|G|}\sum_{j \in G} r_j$$

- 优化目标：

$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \rho_\theta^i A_i - \beta \cdot \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]\right]$$

任务级优势**均匀分配**到所有 token（含记忆操作 token），使 agent 联合优化记忆使用与任务表现。

训练动态：RL 过程中 Read 频率先高后降，Create/Update/Delete 频率持续上升——从「过度检索」转向「主动维护紧凑记忆」。

## Inference Process

1. 长文本按 chunk（4K token）逐步喂入 agent
2. 每步 agent 观测 oₜ = {oₜᵉⁿᵛ, mₜˢᶜʳ, ℳ̂ₜ}，自主决定执行哪些 CRUD 操作
3. scratchpad 维护全局状态，向量库存储细粒度信息，Read 检索 top-6 条目
4. 处理完所有 chunk 后，基于记忆回答问题

策略完全由训练后的 πθ 驱动，无需手工规则。

## Dataset & Evaluation

**基座模型**：Qwen3-8B；Embedding：Qwen3-embedding-0.6B

| 类型 | 数据集 | 设置 |
|---|---|---|
| Long-context QA | HotpotQA, 2WikiMQA, Musique | 200doc (~28K) / 800doc (~112K)，multi-question (1~10题) |
| Web Search | GAIA, WebWalkerQA | 最多 40 次工具调用 |

**指标**：Exact Match (EM) / LLM-as-judge

**主要结果**（200doc，Qwen3-8B）：

| 方法 | HotpotQA | 2WikiMQA | Musique | GAIA | WebWalker | Avg |
|---|---|---|---|---|---|---|
| Full Context | 63.5 | 55.7 | 42.8 | 23.3 | 29.5 | 46.0 |
| A-Mem | 73.5 | 62.7 | 47.1 | 30.1 | 29.0 | 51.4 |
| MemAgent (RL) | 76.5 | 65.8 | 54.7 | 33.0 | 50.0 | 56.7 |
| AtomMem w/o RL | 65.9 | 52.8 | 47.8 | 35.2 | 45.6 | 50.3 |
| **AtomMem** | **77.8** | **67.5** | **55.1** | **37.4** | **48.7** | **58.8** |

**消融**：
- 去掉 Update → 平均 -6.2；去掉 Delete → 平均 -0.8（当前任务为信息积累型，Delete 不关键）
- 去掉 scratchpad → -8.8；去掉 storage → -9.3；**两者都去 → -45.2**（灾难性下降）
- 检索数 K=6 最优；chunk size 对性能影响不大（模型鲁棒）

## RQ
### 1. 怎么让 agent 的记忆管理从静态流程变为动态自适应？
将记忆管理解构为原子 CRUD 操作，建模为 POMDP，agent 每步自主决策执行哪些操作（而非遵循固定 pipeline）。操作的原子性（完备+最小+任务无关）保证了表达能力和泛化性。
### 2. 怎么学习这个记忆策略？
直接用 GRPO 端到端优化，任务级终端奖励均匀分配到所有 token（含 CRUD 操作 token）。训练过程中 agent 自发学到了 task-aligned 的记忆管理模式（如减少冗余 Read、增加 Update/Delete 维护紧凑记忆）。
