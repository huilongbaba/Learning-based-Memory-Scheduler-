# RQ2: Which Component is Optimized?

> **研究问题**：在记忆增强的 LLM 系统中，优化（训练 / 微调 / 强化学习）作用于哪个组件？

## 分类总览

根据被优化的系统组件，将现有工作分为四类：

|类别|优化对象|含义|代表论文|
|---|---|---|---|
|O1|独立可插拔模型|训练一个独立于主 LLM 的辅助模型来管理记忆操作|Memory-R1, General Agentic Memory|
|O2|主 LLM 本身|对主模型的权重或 attach 模块进行微调|MemSearcher, Knowledge Modules|
|O3|检索算法|优化记忆 / 文档的检索机制 $v$|Memento|
|O4|工具触发概率|优化模型调用工具（含记忆操作）的策略与概率|Just in Time (JitRL)|

---

## 类别定义

### O1：独立可插拔模型（Pluggable Model）

**定义**：训练一个与主 LLM 分离的辅助模型，专门负责记忆管理（何时读写、读写什么、如何组织）。主 LLM 的参数在优化过程中保持冻结。

**优化目标**：让辅助模型学会更好的记忆管理策略（写入决策、检索决策、记忆更新等）。

---

### O2：主 LLM 本身（Original / Host Model）

**定义**：直接对主 LLM 的权重 $\theta$ 或其 attach 模块（如 LoRA adapter）进行训练，使模型本身具备更好的记忆利用能力。

**优化目标**：提升 $\pi_\theta$ 在给定 context $C_N$ 下的输出质量，或使模型在无外部记忆时也能利用内化的知识。

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

| Paper                   | O1 可插拔模型 | O2 主 LLM | O3 检索算法 | O4 工具概率 |
| ----------------------- | -------- | -------- | ------- | ------- |
| Memory-R1               | +        |          |         |         |
| MemSearcher             |          | +        |         |         |
| General Agentic Memory  | +        |          |         |         |
| Memento                 |          |          | +       |         |
| Just in Time (JitRL)    |          |          |         | +       |
| Fine-tuning with RAG    |          | +        |         |         |
| Knowledge Modules (DCD) |          | +        |         |         |
| A-MEM                   | -        | -        | -       | -       |
| MSA                     |          | +        | +       |         |
| RLIF                    |          | +        |         |         |

> `+` = 适用  
> `—` = 此维度不适用（无记忆系统 / 无训练过程）  
> `!` = 无法归入现有类别，需扩展分类  
