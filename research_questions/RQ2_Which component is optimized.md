# RQ2: Which Component is Optimized?

> **研究问题**：在记忆增强的 LLM 系统中，优化（训练 / 微调 / 强化学习）作用于哪个组件？

## 分类总览

根据被优化的系统组件，将现有工作分为四类：

| 类别  | 优化对象     | 含义                          | 代表论文                           |
| --- | -------- | --------------------------- | ------------------------------ |
| O1  | 独立可插拔模型  | 独立于主 LLM 的被训练模型（含记忆辅助的任务模型） | Memory-R1, MIRA                |
| O2  | 主 LLM 本身 | 对主模型的权重或 attach 模块进行微调      | MemSearcher, Knowledge Modules |
| O3  | 检索算法     | 优化记忆 / 文档的检索机制 $v$          | Memento                        |
| O4  | 工具触发概率   | 优化模型调用工具（含记忆操作）的策略与概率       | Just in Time (JitRL)           |

---

## 类别定义

### O1：独立可插拔模型（Pluggable Model）

**定义**：训练一个与主 LLM 分离的独立模型，主 LLM 的参数在优化过程中保持冻结。

**优化目标**：让辅助模型学会更好的记忆管理策略（写入决策、检索决策、记忆更新等）。

根据被训练模型的职责不同，可进一步分为两个子类：  

---

#### O1a：记忆管理模型（Memory Manager）

训练一个辅助模型，专门负责记忆的管理操作——何时读写、读写什么、如何组织和更新记忆。该模型的输入是当前 context 和记忆状态，输出是记忆操作决策（写入、检索、更新、删除等）。主 LLM 根据该模型提供的记忆内容生成最终回答，但其自身参数不被修改。

**代表论文**：Memory-R1

---

#### O1b：记忆辅助的任务模型（Memory-Augmented Task Model）

训练一个独立的任务执行模型（如 RL policy network），该模型直接在环境中执行任务。LLM 仅作为知识源提供 prior（如 subgoal 分解、trajectory 建议），这些 prior 被存储在记忆结构中，用于辅助任务模型的训练（如 shaping advantage estimation）。记忆结构本身的管理（节点添加、剪枝、更新）可以由规则驱动而非学习得到。

**核心区别**：O1a 中被训练的模型管理记忆，O1b 中被训练的模型使用记忆来执行任务。两者的共同点是主 LLM 参数冻结、被优化对象独立于 LLM 之外。

**代表论文**：MIRA

---

### **O2：主 LLM 参数优化（Host LLM Parameter Optimization）**

**总类定义**：直接对承担记忆任务（写入、抽象、检索、回答等）的 LLM 的参数 θ\theta θ 或其 attach 模块（如 LoRA adapter）进行训练，使 LLM 本身具备更好的记忆利用或管理能力。 被优化对象是 LLM 本身，而非独立于 LLM 之外的辅助模型（后者属于 O1）。

根据被优化的 LLM 数量和优化方式的协同程度，进一步分为两个子类：

---

#### **O2a：单 LLM 优化（Single-Agent LLM Optimization）**

**定义**：系统中只有一个被优化的 LLM，它独自承担记忆增强任务中的核心角色（如检索+回答、或 in-context 记忆利用）。优化目标是让这个单一 LLM 在给定 context CNC_N CN​ 下输出更优的回答，或使其在无外部记忆时也能利用内化的知识。

**关键特征**：

- 被优化的 LLM 数量 = 1
- 不涉及跨 agent 的 credit assignment 问题
- 训练目标直接以该 LLM 的 rollout 结果计算 reward

**代表论文**：MemSearcher

---

#### **O2b：多 LLM 联合优化（Joint Multi-Agent LLM Optimization）**

**定义**：系统包含多个协同工作的 LLM agent，每个 agent 负责记忆 pipeline 中的一个子任务（如写入细粒度记忆、抽象粗粒度画像、检索、回答等），这些 LLM 的参数**被同时更新**，且它们的优化过程**相互耦合**——通过共享轨迹、联合 reward 设计、或 credit assignment 机制，把 agent 间的依赖关系显式编码进训练过程。

**关键特征**：

- 被优化的 LLM 数量 ≥ 2
- 存在跨 agent 的 credit assignment 问题（global reward 如何分配给各 agent）
- 各 agent 的优化不是相互独立的 single-agent 训练的简单叠加

**代表论文**：CoMAM

---

### O3：检索算法（Retrieval Algorithm）

**定义**：优化检索机制 $v$，使其能从 $m_l$ 中更精准地召回与当前查询 $q$ 相关的记忆条目。不修改模型参数或辅助模型，而是改进"取"的环节。

**优化目标**：提高检索的 precision 和 recall，使 $C_N$ 中包含更相关、更少噪声的信息。

---

### O4：工具触发概率（Tool Probability / Action Policy）

**定义**：优化模型在 agent loop 中调用工具（包括记忆读写工具）的策略和触发概率。不直接修改记忆内容或检索算法，而是学习"什么时候该用记忆工具"。

**优化目标**：提升工具使用的精准度，在需要记忆时调用，不需要时跳过，减少无效 loop。

---

## 论文分类表

| Paper                          | O1 可插拔模型 | O2 主 LLM | O3 检索算法 | O4 工具概率 |O5 KV cache |
| ------------------------------ | -------- | -------- | ------- | ------- | ------- |
| Memory-R1                      | +        |          |         |         |         |
| MemSearcher                    |          | +        |         |         |         |
| General Agentic Memory         | +        |          |         |         |         |
| Memento                        |          |          | +       |         |         |
| Just in Time (JitRL)           |          |          |         | +       |         |
| Fine-tuning with RAG           |          | +        |         |         |         |
| Knowledge Modules (DCD)        |          | +        |         |         |         |
| A-MEM                          | -        | -        | -       | -       |         |
| MSA                            |          | +        | +       |         |         |
| RLIF                           |          | +        |         |         |         |
| RF-Mem                         |          |          | +       |         |         |
| GRU-Mem                        |          | +        |         |         |         |
| UMA                            |          | +        |         |         |         |
| HyperRAG                       |          |          | +       |         |         |
| Mnemis                         | -        | -        | -       | -       |         |
| MIRA                           | +        |          |         |         |         |
| Memory-Based Advantage Shaping | +        |          |         |         |         |
| EMPO                           |          | +        |         |         |         |
| MemPO                          |          | +        |         |         |         |
| MemSifter                      | +        |          | +       |         |         |
| A-MAC                          | +        |          |         |         |         |
| CoMAM                          | +        | +        |         |         |         |
| Memex(RL)                      |          | +        |         | +       |         |
| Titans                         |        |+        |           |       | +      |
| Mem $\alpha$                    |+       |        |          |       |        |
| Scalingcontextfolding          |        |+        |          |       |       |

> `+` = 适用  
> `—` = 此维度不适用（无记忆系统 / 无训练过程）  
> `!` = 无法归入现有类别，需扩展分类  
