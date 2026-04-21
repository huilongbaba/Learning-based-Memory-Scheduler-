# RQ2: Which Component is Optimized?

> **研究问题**：在记忆增强的 LLM 系统中，优化（训练 / 微调 / 强化学习）作用于哪个组件？

## 分类总览

根据被优化的系统组件，将现有工作分为五类：

| 类别  | 优化对象                     | 含义                                      | 代表论文                             |
| --- | ------------------------ | --------------------------------------- | -------------------------------- |
| O1  | 独立可插拔记忆管理器（RAG 式）        | 独立于主 LLM 的被训练"记忆管理模块"，负责读/写/检索决策        | Memory-R1                        |
| O2  | 主 LLM 本身                 | 对主模型的权重或 attach 模块（LoRA 等）进行微调          | MemSearcher, Knowledge Modules   |
| O3  | 检索算法                     | 优化记忆 / 文档的检索机制 $v$                      | Memento, HyperRAG, RF-Mem        |
| O4  | 工具触发概率                   | 优化模型调用工具（含记忆操作）的策略与概率                   | Just in Time (JitRL), Memex (RL) |
| O5  | KV Cache / 参数化 Attention | 优化推理时 memory matrix 或 KV cache 的生成/更新机制 | Titans                           |

---

## 类别定义

### O1：独立可插拔记忆管理器（Pluggable Memory Manager, RAG-style）

**定义**：训练一个与主 LLM 分离的独立模块，主 LLM 的参数在优化过程中保持冻结。该模块的职责是在 RAG 风格的 pipeline 中管理记忆的读、写、更新、检索决策，其输入是当前 context 与记忆状态，输出是记忆操作决策；主 LLM 根据该模块提供的记忆内容生成最终回答。

**优化目标**：让辅助模块学会更好的记忆管理策略（写入决策、检索决策、记忆更新、遗忘等）。

**关键特征**：

- 主 LLM 参数冻结
- 被优化对象是独立模块（可以是小 LLM、gate、MLP 等）
- 主 LLM 与记忆管理解耦，模块可独立替换

**代表论文**：Memory-R1（memory manager）

---

### O2：主 LLM 参数优化（Host LLM Parameter Optimization）

**总类定义**：直接对承担记忆任务（写入、抽象、检索、回答等）的 LLM 的参数 $\theta$ 或其 attach 模块（如 LoRA adapter）进行训练，使 LLM 本身具备更好的记忆利用或管理能力。被优化对象是 **LLM 本身**，而非独立于 LLM 之外的辅助模块（后者属于 O1）。

根据被优化的 LLM 数量和优化方式的协同程度，进一步分为两个子类：

---

#### O2a：单 LLM 优化（Single-Agent LLM Optimization）

**定义**：系统中只有一个被优化的 LLM，它独自承担记忆增强任务中的核心角色（如检索 + 回答、或 in-context 记忆利用）。优化目标是让这个单一 LLM 在给定 context $C_N$ 下输出更优的回答，或使其在无外部记忆时也能利用内化的知识。

**关键特征**：

- 被优化的 LLM 数量 = 1
- 不涉及跨 agent 的 credit assignment 问题
- 训练目标直接以该 LLM 的 rollout 结果计算 reward

**代表论文**：MemSearcher, EMPO, MemPO, UMA。

---

#### O2b：多 LLM 联合优化（Joint Multi-Agent LLM Optimization）

**定义**：系统包含多个协同工作的 LLM agent，每个 agent 负责记忆 pipeline 中的一个子任务（如写入细粒度记忆、抽象粗粒度画像、检索、回答等），这些 LLM 的参数**被同时更新**，且它们的优化过程相互耦合——通过共享轨迹、联合 reward 设计、或 credit assignment 机制，把 agent 间的依赖关系显式编码进训练过程。

**关键特征**：

- 被优化的 LLM 数量 ≥ 2
- 存在跨 agent 的 credit assignment 问题（global reward 如何分配给各 agent）
- 各 agent 的优化不是相互独立的 single-agent 训练的简单叠加

**代表论文**：CoMAM。

>多 LLM 是否比单 LLM 更强？如何证明？

---

### O3：检索算法（Retrieval Algorithm）

**定义**：优化检索机制 $v$，使其能从 $m_l$ 中更精准地召回与当前查询 $q$ 相关的记忆条目。**不修改模型参数或辅助模块**，而是改进"取"的环节——包括 embedding 空间、排序函数、查询重写、混合检索链路等。

**优化目标**：提高检索的 precision 和 recall，使 $C_N$ 中包含更相关、更少噪声的信息。

**代表论文**：Memento, RF-Mem（双链路 / 深度回忆），HyperRAG（超图检索），MSA。

---

### O4：工具触发概率（Tool Probability / Action Policy）

**定义**：优化模型在 agent loop 中调用工具（包括记忆读写工具）的策略和触发概率。**不直接修改记忆内容或检索算法**，而是学习"什么时候该用记忆工具"。

**优化目标**：提升工具使用的精准度，在需要记忆时调用，不需要时跳过，减少无效 loop。

**代表论文**：Just in Time (JitRL)，Memex (RL)（compress / read experience 作为工具）。

---

### O5：KV Cache / 参数化 Attention

**定义**：优化的对象是推理时的 KV cache 或参数化的 memory matrix，即把上下文窗口级的记忆结构本身（而非其内容或检索过程）视为可训练的参数对象。比如把原本 static 的 KV 矩阵重写为可学习的、随交互动态更新的 memory 模块，并通过端到端训练塑造其行为。

**关键特征**：

- 优化对象位于 attention 计算路径内部
- 记忆更新规则是可学习的

**与 O2 的关系**：O2 优化的是整体权重或 attach 的 adapter；O5 只针对 memory matrix / KV cache 这一结构。

**代表论文**：Titans。

---

## 论文分类表

| Paper                          | O1 可插拔管理器 | O2 主 LLM | O3 检索算法 | O4 工具概率 | O5 KV cache |
| ------------------------------ | --------- | -------- | ------- | ------- | ----------- |
| Memory-R1                      | +         |          |         |         |             |
| MemSearcher                    |           | +        |         |         |             |
| General Agentic Memory         | +         |          |         |         |             |
| Memento                        |           |          | +       |         |             |
| JitRL                          |           |          |         | +       |             |
| Fine-tuning with RAG           |           | +        |         |         |             |
| Knowledge Modules (DCD)        |           | +        |         |         |             |
| A-MEM                          | —         | —        | —       | —       | —           |
| MSA                            |           | +        | +       |         |             |
| RLIF                           |           | +        |         |         |             |
| RF-Mem                         |           |          | +       |         |             |
| GRU-Mem                        |           | +        |         |         |             |
| UMA                            |           | +        |         |         |             |
| HyperRAG                       |           |          | +       |         |             |
| Mnemis                         | —         | —        | —       | —       | —           |
| MIRA                           | +         |          |         |         |             |
| Memory-Based Advantage Shaping | +         |          |         |         |             |
| EMPO                           |           | +        |         |         |             |
| MemPO                          |           | +        |         |         |             |
| MemSifter                      | +         |          | +       |         |             |
| A-MAC                          | +         |          |         |         |             |
| CoMAM                          | +         | +        |         |         |             |
| Memex (RL)                     |           | +        |         | +       |             |
| Titans                         |           | +        |         |         | +           |
| Mem-$\alpha$                   | +         |          |         |         |             |
| Memory as Action               |           | +        |         |         |             |
| Scaling Context Folding        |           | +        |         |         |             |

> `+` = 适用  
> `—` = 此维度不适用（无记忆系统 / 无训练过程）  
> `!` = 无法归入现有类别，需扩展分类