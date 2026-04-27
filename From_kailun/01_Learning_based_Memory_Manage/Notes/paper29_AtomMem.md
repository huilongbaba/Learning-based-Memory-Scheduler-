# AtomMem: Learnable Dynamic Agentic Memory with Atomic Memory Operation

**Source:** [arXiv:2601.08323v3](https://arxiv.org/abs/2601.08323) (2026-03-27 修订)

---
## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$s_t = (s_t^{\text{env}}, s_t^{\text{mem}})$|$s_t$（POMDP 全局状态）|包含外部环境状态与内部记忆状态|
|$a_t = (a_t^{\text{env}}, a_t^{\text{mem}})$|$a$|联合动作：环境动作 + 记忆动作|
|$a_t^{\text{mem}} \in$ {Create, Read, Update, Delete}|作用于 $g_s$ / $g_l$ / $v$ 的具体操作|CRUD 原子操作集|
|$o_t = (o_t^{\text{env}}, m_t^{\text{scr}}, \hat{M}_t)$|$o$|观测 = 环境反馈 + scratchpad + 检索结果|
|$m_t^{\text{scr}}$（scratchpad）|$m_s$（短期/工作记忆）|每步全量重写的固定化短期记忆|
|$M_t = \lbrace m_i \rbrace_{i=1}^{N_t}$（vector DB）|$m_l$（长期记忆）|FAISS 向量数据库存储的离散条目集合|
|$\hat{M}_t = \text{TopK}(\text{sim}(q_{t-1}, m_i))$|$v(q, m_l)$|语义相似度检索算法|
|Update / Create / Delete 操作|$g_l$|长期记忆的写入/更新/遗忘函数|
|`<update_scratchpad>` 操作|$g_s$|短期记忆的全量重写|
|$\pi_\theta$（Qwen3-8B 策略）|$\pi$ / $\theta$|主模型策略与参数|
|$\tau = (o_1, a_1, \ldots, o_T, a_T)$|$T$（trajectory）|多步交互轨迹|
|EM / LLM-judge reward $r_i$|terminal reward|任务完成度奖励|

---
## 概览

这篇文章用了任务执行轨迹（rollout 出来的 trajectory，包含每一步的 CRUD 记忆操作 + 环境动作）以及任务级终端奖励（QA 用 EM 匹配，web 任务用 LLM-as-judge）。    
最后得到了一个会自己管理记忆的主 LLM（fine-tuned Qwen3-8B），不是一个独立的记忆管理器，而是内化到主模型自身里，让它在每一步自己决定 CRUD。  
最终优化的是最终答案的正确性（QA accuracy / web task success），通过 GRPO 把任务级奖励均匀分配到所有输出 token，从而让模型学会一种任务自适应的记忆策略。

---
## 1. Problem Setting

- **记忆类型**：in-chat memory（每个 task 开始时 $s_0^{\text{mem}} = \varnothing$，不跨任务积累），同时包含 $m_s$（scratchpad）和 $m_l$（vector DB）。
- **决策建模**：POMDP $(\mathcal{S}, \mathcal{A}, P, \Omega, \mathcal{O}, \mathcal{R}, \gamma)$，记忆被显式建模为环境的可控部分。
- **状态空间**：$\mathcal{S} \ni s_t = (s_t^{\text{env}}, s_t^{\text{mem}})$，环境状态 + 内部记忆状态。
- **动作空间**：$\mathcal{A} \ni a_t = (a_t^{\text{env}}, a_t^{\text{mem}})$，其中 $a_t^{\text{mem}} \in \lbrace \text{Create}, \text{Read}, \text{Update}, \text{Delete} \rbrace$。每步可以输出一个由若干原子操作组成的复合宏动作 $A_t = \lbrace a_t^1, \ldots, a_t^{K_t} \rbrace$，非读操作按序执行。
- **观测空间**：$\Omega \ni o_t = (o_t^{\text{env}}, m_t^{\text{scr}}, \hat{M}_t)$，记忆部分通过 hybrid retrieval 给出（scratchpad 必出 + Top-K 向量检索）。
- **数据结构**：双层结构 —— scratchpad 是单条结构化文本条目（$m_s$ 角色），vector DB 是离散条目集合 $M_t = \lbrace m_i \rbrace_{i=1}^{N_t}$（$m_l$ 角色），后者通过 FAISS 实现。

|框架组件|论文实现|
|---|---|
|$m_s$（短期）|Scratchpad $m_t^{\text{scr}}$，每步通过 `<update_scratchpad>` 全量重写|
|$m_l$（长期）|FAISS 向量数据库 $M_t$，离散条目|
|$g_s$|`<update_scratchpad>` 操作（全量重写）|
|$g_l$|Create / Update / Delete 三个原子操作的组合 $M_{t+1} = a_t^{K_t} \circ \cdots \circ a_t^1 (M_t)$|
|$v$|Hybrid retrieval：deterministic scratchpad + Top-K 语义检索|
|$\pi$|Qwen3-8B 直接输出 XML 标签格式的 CRUD tokens|

---


## 2. Training Procedure

- **优化对象**：主 LLM 参数 $\theta$（Qwen3-8B），fully on-policy 更新。记忆操作是 LLM 词表中的结构化 XML token，因此优化序列输出似然就隐式优化了记忆策略。
- **优化算法**：GRPO（Dr.GRPO 变体——不做 advantage normalization）。
- **训练数据**：在线交互 rollout，每个 task 对应一条多步 trajectory $\tau$。QA 任务在 HotpotQA / 2WikiMQA / Musique 的训练集上训练，web 任务在 Asearcher 数据集上训练。
- **是否冻结 LLM**：否。主 LLM 全参数训练（无 LoRA / adapter / 独立辅助模块）。
- **核心目标函数**（论文公式 (8)(9) + 框架符号）：

$$A_i = r_i - \frac{1}{\lvert G \rvert} \sum_{j \in G} r_j$$

其中 $G$ 是同 task 重复 rollout 的轨迹组，$r_i$ 是终端奖励。

$$\mathcal{J}(\theta) = \mathbb{E}\Big\lbrack \frac{1}{G}\sum_{i=1}^{G} \rho_\theta^i A_i - \beta D_{KL}\big(\pi_\theta \Vert \pi_{\text{ref}}\big) \Big\rbrack$$

$\rho_\theta^i$ 是第 $i$ 个样本的重要性采样比。**advantage 在所有输出 token（包括记忆操作 token）上均匀分配**。

---
## 3. Reward Signal

- **奖励类型**：sparse terminal reward（轨迹末尾给一次）。
- **奖励来源**：
    - QA 任务（HotpotQA / 2WikiMQA / Musique）：EM 与 ground truth 比对；
    - Web 任务（GAIA / WebWalkerQA）：LLM-as-judge。
- **分配机制**：terminal reward 通过 GRPO group baseline 转换为 advantage，然后**均匀分配到 trajectory 中所有 output token**。无 step-level 奖励，无 process reward。
- **辅助奖励 / 正则项**：仅有 KL 约束 $D_{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ 一项，无 format reward / length penalty 等。

> 用 group relative + KL 约束控制住了训练稳定性。

---

## 4. Inference Procedure

- **记忆初始化**：每个新任务开始时 $s_0^{\text{mem}} = \varnothing$，scratchpad 与 vector DB 同时清空（不跨任务持久）。
- **每步决策流程**：
    1. **观测**：agent 接收 $o_t = (o_t^{\text{env}}, m_t^{\text{scr}}, \hat{M}_t)$，包括当前文档 chunk（4K tokens）、scratchpad、上一步 read query 检索到的 Top-6 entries。
    2. **思考与决策**：LLM 在一次 forward 中输出包含 `<update_scratchpad>` 与若干 `<create_memory>` / `<read_memory>` / `<update_memory>` / `<delete_memory>` 的 XML 序列。
    3. **执行**：scratchpad 全量重写；非读操作按序作用于 $M_t$；read 操作的 query 被存下，下一步检索结果作为 $\hat{M}_{t+1}$。
- **额外策略**：
    - **Hybrid retrieval**：scratchpad 100% 必出（确定性检索），vector DB 上做 Top-K=6 语义相似度检索（选择性检索）。
    - **Chunk size**：长文档切成 4K 一块逐步喂入。
    - **Inference 温度**：0.7（Qwen3 推荐），top-p=1，top-k 关闭。
    - **Web 任务**：最多 40 次工具调用。
- **是否完全由学习驱动**：CRUD 操作的**触发**与**参数**完全由学到的 $\pi_\theta$ 决定；但**框架本身**（双层记忆、Top-K 检索、4K chunk 切分、scratchpad 必出）仍是手工规则。

---
## 5. RQ 分析

### RQ1 (What is memory?)

记忆是双层结构：scratchpad（$m_s$）+ vector DB（$m_l$）。前者是每步全量重写的固定化短期记忆（ T2：固定长度压缩），后者是非参数化的离散条目集合 + Top-K 检索（T1：非参数外部记忆）。两者都是显式记忆，可被有意识地查看与编辑。

### RQ2 (Which component is optimized? Which signal is used?)

优化的是主 LLM 自身的参数 $\theta$（Qwen3-8B 全参数训练），不冻结、无独立辅助模块。训练信号是任务级 outcome reward（QA 用 EM，web 用 LLM-judge），通过 GRPO 的 group-relative advantage 转换为策略梯度。

### RQ3 (Target of Optimization)

主要追求 G1 回答准确度（QA EM、web 任务的 judge 评分）；同时通过 case study 与 training dynamic 隐式追求记忆策略的任务对齐性（如 Read 频率下降、Create/Update/Delete 频率上升）。Web 任务部分接近 G5（task success via execution），但仍以 LLM-judge 作为代理。

### RQ4 (Training Signal and RL Algorithm)

采用 A2: GRPO。

### RQ5 (How memory evolves, operates?)

运行时通过 LLM 自主输出的 XML 标签触发 CRUD 操作：Create 写入新条目至 vector DB；Read 触发下一步的 Top-K 检索；Update 按 memory id 修改既有条目；Delete 按 memory id 永久移除。Scratchpad 每步全量重写。整体是一个 context-sensitive、动态触发的记忆演化模式，当观测信息不足时记 scratchpad 备忘，部分匹配时主动 Create + Read 凑齐，全部齐备时 Update 总结结论并覆写冗余条目。

---
## Conclusion

AtomMem 把 LLM agent 的记忆管理从"手工 workflow"改造成"模型自己学会的决策过程"。具体做法是把所有高层记忆操作分解成最原子的 CRUD 四件套，让主 LLM 在每一步自己决定要做哪些记忆操作，再用 GRPO + 任务级 outcome reward 进行强化学习训练。论文设计了 scratchpad（短期）+ vector DB（长期）的双层显式记忆，并通过 hybrid retrieval 把两者整合进每步的观测中。在 3 个长上下文 QA 与 2 个 web 任务上，相比 mem0 / A-Mem / MemAgent 等静态 workflow 基线，平均提升 3-8 个百分点。