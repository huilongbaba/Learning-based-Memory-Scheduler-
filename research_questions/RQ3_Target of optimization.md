# RQ3: Target of Optimization

> **研究问题**：记忆增强系统的优化最终追求什么目标？用什么指标衡量"记忆用得好"？

## 分类总览

当前现有工作的优化目标分为六类：

| 类别  | 层次   | 优化目标       | 含义                                       | 代表论文                                                    |
| --- | ---- | ---------- | ---------------------------------------- | ------------------------------------------------------- |
| G1  | 核心   | 回答准确度      | 最终回答的正确性（QA accuracy）                    | Memory-R1, MemSearcher, General Agentic Memory, Memento |
| G2  | 核心   | 效率         | Token 消耗 / Agent loop 轮数 / KV cache 命中率等 | MemSearcher, HyperRAG, MemPO                            |
| G3  | 局部   | 行为 / 行动奖励  | 对 agent 每一步操作的"帮助度"打分                    | Just in Time (JitRL)                                    |
| G4  | 局部   | 写入决策质量     | 候选记忆是否应写入 $m_l$ 的分类质量                    | A-MAC, GRU-Mem, Mem-α                                   |
| G5  | 核心   | 任务成功率（可执行） | 通过执行（如运行代码）评估任务是否完成                      | Memory as Action, Scaling Context Folding               |
| G6  | 补充兜底 | 补充优化目标     | 物理含义不明确或难以归入 G1–G5 的 reward 项            | Scaling Context Folding（format reward 等）                |

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
- **Memory compression**：保留质量的条件下缩小 memory budget $L$

**框架对应**：约束 $|C_N|$ 或 agent loop 次数 $N$ 的同时最大化回答质量。

---

### G3：行为 / 行动奖励（Tool Action Reward）

**定义**：不以最终回答的正确性为目标，而是对 agent 在每一步选择的行动（action）进行奖励。典型做法是使用 LLM-as-judge 对每步操作的"帮助度"（helpfulness）打分，或根据工具调用是否"正确"给出离散奖励。

**度量方式**：基于 LLM judge 的逐步评分、行动序列的 trajectory reward。

**框架对应**：优化 policy $\pi_\theta$ 在 trajectory $T = (a_1, o_1, a_2, o_2, \ldots)$ 上的累积 reward。

> 待定是否并入 G1
> **为什么并入 G1**：G3 本质上是 G1 的过程监督实现，最终都是为了让下游答案更好。
   **为什么保持独立**：G3 可以在不依赖下游 ground truth 的情况下提供训练信号（如纯 LLM-judge 评分）。

---

### G4：写入决策质量（Write-Decision Quality）

**定义**：以"每条候选记忆是否应写入 $m_l$"这一分类问题本身的指标作为优化目标，通常对应于人工标注或规则生成的 ground-truth admission labels。也可使用 LLM judge 评估某次压缩 / 写入是否丢失重要信息。

**典型形式**：

- **基于 label 的 BCE / classification loss**：对每条候选记忆判断 admit / reject；
- **基于信息保真度的评分**：写入或压缩后，能否从 $m_l$ 重构出关键信息（LLM judge 或 information retention 指标）；
- **基于对比基线的 margin utility**（MemSifter）：写入后对下游检索质量的边际贡献。

**度量方式**：记忆写入决策本身的质量，不假设下游任务，G4 的可贵之处在于不依赖最终答案即可提供密集训练信号。

**框架对应**：优化 $g_l$ 中的 gating 子策略，使其决策与 $y^*$ 对齐。

> 待定是否并入 G1
> **为什么并入 G1**：写入质量最终仍服务于下游回答，可以视为 G1 的代理。
> **为什么保持独立**：G4 能在任何下游任务尚未定义时就独立评估记忆系统的好坏。

---

### G5：任务成功率（Task Success via Execution）

**定义**：不通过语义比对（text match、LLM judge 等）判断任务是否完成，而是通过实际执行（running code、调用 API、环境交互）所产生的客观结果来判断。典型如：提交的代码 patch 能否通过单元测试、命令序列能否让环境进入目标状态、SQL 查询是否返回正确结果集等。

**度量方式**：

- **Pass@k / Test pass rate**：提交解能否通过测试套件（SWE-bench 家族）
- **Goal achievement rate**：环境状态是否达到 goal state（Web agent / robotics）
- **End-to-end success**：整个任务流水线的黑盒成功 / 失败

**框架对应**：

$$r^{\text{task}} = \mathbb{1}[\text{execute}(a_{1:N}, \text{env}) \vDash g^*]$$

其中 $g^*$ 是可验证的目标状态，$\vDash$ 表示满足。

**代表论文**：Memory as Action（SWE-bench），Scaling Context Folding（在可执行 agent 任务上验证）。

---

### G6：补充优化目标（Supplementary  Rewards）

**定义**：物理含义不明确、或难以归入 G1–G5 的辅助 reward 项。典型如：

- **Format reward**：要求输出符合特定结构（JSON、XML、特定标签等）；
- **Length / brevity penalty**：避免输出过长或过短（不直接等同于 G2 的效率目标，更偏 shaping）；
- **Self-consistency bonus**：多次采样结果一致性；
- **Spurious / process reward**：无明确物理语义、但实证有效的过程奖励。

**度量方式**：通常是辅助项，与主 reward 做加权组合：

$$r_{\text{total}} = r_{\text{main}} + \sum_k \lambda_k \cdot r^{\text{aux}}_k$$

**代表论文**：Scaling Context Folding（format reward 等辅助项）。

---

## 论文分类表

| Paper                          | G1 准确度 | G2 效率 | G3 行动奖励 | G4 写入决策 | G5 task success | G6 辅助 |
| ------------------------------ | ------ | ----- | ------- | ------- | --------------- | ----- |
| Memory-R1                      | +      |       |         |         |                 |       |
| MemSearcher                    | +      | +     |         |         |                 |       |
| General Agentic Memory         | +      |       |         |         |                 |       |
| Memento                        | +      |       |         |         |                 |       |
| JitRL                          |        |       | +       |         |                 |       |
| Fine-tuning with RAG           | +      |       |         |         |                 |       |
| Knowledge Modules (DCD)        | +      |       |         |         |                 |       |
| A-MEM                          | —      | —     | —       |         |                 |       |
| MSA                            | +      |       |         |         |                 |       |
| RLIF                           | +      |       |         |         |                 |       |
| RF-Mem                         | +      | +     |         |         |                 |       |
| GRU-Mem                        | +      | +     | +       | +       |                 |       |
| UMA                            | +      |       | +       |         |                 |       |
| HyperRAG                       | +      | +     |         |         |                 |       |
| Mnemis                         | —      | —     | —       |         |                 |       |
| MIRA                           | +      | +     |         |         |                 |       |
| Memory-Based Advantage Shaping | +      | +     |         |         |                 |       |
| EMPO                           | +      |       |         |         |                 |       |
| MemPO                          | +      | +     |         |         |                 |       |
| MemSifter                      | +      | +     |         | +       |                 |       |
| A-MAC                          |        | +     |         | +       |                 |       |
| CoMAM                          | +      |       | +       |         |                 |       |
| Memex (RL)                     | +      | +     | +       |         |                 |       |
| Titans                         |        |       |         |         |                 | +     |
| Mem-$\alpha$                   | +      | +     |         | +       |                 |       |
| Memory as Action               |        |       |         |         | +               |       |
| Scaling Context Folding        |        |       |         |         | +               | +     |
| MAGMA                          | -      | -     | -       | -       | -               | -     |
| MACLA                          | +      | +     |         | +       |                 |       |
| LightSearcher                  | +      | +     |         |         |                 | +     |
| MemVerse                       | +      | +     |         |         |                 |       |
| Inside Out                     | +      | +     |         | +       |                 |       |
| MemBuilder                     | +      | +     |         |         |                 | +     |
| MemoBrain                      | +      | +     |         |         |                 |       |
| Fine-Mem                       | +      | +     |         | +       |                 | +     |

> `+` = 适用  
> `—` = 此维度不适用（无记忆系统 / 无训练过程）  
> `!` = 无法归入现有类别，需扩展分类