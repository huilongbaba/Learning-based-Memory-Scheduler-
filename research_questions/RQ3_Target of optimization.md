# RQ3: Target of Optimization

> **研究问题**：记忆增强系统的优化最终追求什么目标？用什么指标衡量"记忆用得好"？

## 分类总览

当前现有工作有以下三类优化目标：

|类别|优化目标|含义|代表论文|
|---|---|---|---|
|G1|回答准确度|最终回答的正确性（QA accuracy）|Memory-R1, MemSearcher, General Agentic Memory, Memento|
|G2|效率|Token 消耗 / Agent loop 轮数 / KV cache 命中率等|MemSearcher|
|G3|行为 / 行动奖励|对 agent 每一步操作的"帮助度"打分（Tool action reward）|Just in Time (JitRL)|

---

## 类别定义

### G1：回答准确度（Answer Accuracy）

**定义**：以下游任务的最终回答是否正确作为优化目标。这是目前最主流的 reward signal。

**度量方式**：EM (Exact Match)、F1，或基于 LLM-as-judge 的正确性评分。

**框架对应**：优化 $\hat{a}_N$ 与 ground truth $a^*$ 之间的距离。

---

### G2：效率（Efficiency）

**定义**：在保持回答质量的前提下，降低计算资源消耗。包括但不限于：

- **Token efficiency**：完成任务所需的总 token 数
- **Loop efficiency**：Agent loop 的轮数（越少越好）
- **KV cache 命中率**：缓存复用的比例
- **Memory budget 利用率**：在固定 memory budget $L$ 下的信息保留质量
- **Memory compression**: 保留质量的条件下缩小memory budget $L$

**框架对应**：约束 $|C_N|$ 或 agent loop 次数 $N$ 的同时最大化回答质量。

---

### G3：行为 / 行动奖励（Tool Action Reward）

**定义**：不以最终回答的正确性为目标，而是对 agent 在每一步选择的行动（action）进行奖励。典型做法是使用 LLM-as-judge 对每步操作的"帮助度"（helpfulness）打分。

**度量方式**：基于 LLM judge 的逐步评分、行动序列的 trajectory reward。

**框架对应**：优化 policy $\pi_\theta$ 在 trajectory $T = (a_1, o_1, a_2, o_2, \ldots)$ 上的累积 reward。

---

### **G4：写入决策质量（Write-Decision Quality）**

**定义**：以"每条候选记忆是否应写入 $m_l$"这一分类问题本身的指标作为优化目标，通常对应于人工标注或规则生成的 ground-truth admission labels。

**度量方式**：记忆写入决策本身的质量，不假设下游任务。或者使用LLM judge评估压缩是否丢失重要信息。

**框架对应**：优化 $g_l$ 中的 gating 子策略，使其决策与 $y^*$ 对齐。

---

## 论文分类表

| Paper                          | G1 准确度 | G2 效率 | G3 行动奖励 | G4 写入决策质量 | G5 task success (execution) | G6 supplementary targets|
| ------------------------------ | ------ | ----- | ------- | --------- |------- | --------- |
| Memory-R1                      | +      |       |         |           |         |           |
| MemSearcher                    | +      | +     |         |           |         |           |
| General Agentic Memory         | +      |       |         |           |         |           |
| Memento                        | +      |       |         |           |         |           |
| Just in Time (JitRL)           |        |       | +       |           |         |           |
| Fine-tuning with RAG           | +      |       |         |           |         |           |
| Knowledge Modules (DCD)        | +      |       |         |           |         |           |
| A-MEM                          | -      | -     | -       |           |         |           |
| MSA                            | +      |       |         |           |         |           |
| RLIF                           | +      |       |         |           |         |           |
| RF-Mem                         | +      | +     |         |           |         |           |
| GRU-Mem                        | +      | +     | +       | +         |         |           |
| UMA                            | +      |       | +       |           |         |           |
| HyperRAG                       | +      | +     |         |           |         |           |
| Mnemis                         | -      | -     | -       |           |         |           |
| MIRA                           | +      | +     |         |           |         |           |
| Memory-Based Advantage Shaping | +      | +     |         |           |         |           |
| EMPO                           | +      |       |         |           |         |           |
| MemPO                          | +      | +     |         |           |         |           |
| MemSifter                      | +      | +     |         |           |         |           |
| A-MAC                          |        | +     |         | +         |         |           |
| CoMAM                          | +      |       | +       |           |         |           |
| Memex(RL)                      | +      | +     | +       |           |         |           |
| Mem $\alpha$                    |+       | +       |+          | +      |         |           |
| Scalingcontextfolding          |        |        |           |       | +        |  +         |

> `+` = 适用  
> `—` = 此维度不适用（无记忆系统 / 无训练过程）  
> `!` = 无法归入现有类别，需扩展分类  
