# MemoBrain: Executive Memory as an Agentic Brain for Reasoning

**Source:** [arXiv:2601.08079v1](https://arxiv.org/abs/2601.08079) (2026-01-12 修订)

---

## 符号映射表

|论文原始符号|框架统一符号|含义|
|---|---|---|
|$X$|$q$|复杂推理任务 / 用户输入|
|$Y$|$A_N$|最终答案|
|$x_t = (\tau_t, \omega_t)$|episode（一轮完整推理 trace，含 $t$, $a$, $o$）|单步推理 episode：$\tau_t$ 是 transient execution 痕迹（tool call + raw output），$\omega_t$ 是该步语义产出|
|$\tau_t$|$(a_t, o_t)$ + 中间 $t$|执行级痕迹（工具调用与原始输出）|
|$\omega_t$|episode 的语义结论|该 episode 解决的子问题与产出|
|thought $v_t$|压缩后的 $m_l$ 条目（带依赖结构）|一个 episode 抽象出的紧凑记忆单元|
|memory graph $G_t = (V_t, E_t)$|$m_l$（结构化长期记忆）|全局依赖感知记忆图，节点为 thoughts，边为依赖关系|
|working context $C_t$|$C_N$（输入到主 LLM 的上下文）|主推理 agent 实际看到的 context|
|memory model $\phi(\cdot)$|$g_l$（长期记忆更新函数） + $v$（检索/投影） 的合并实现|独立的 memory model，同时承担记忆构造和管理|
|executive operations $O_t$|$g_l$ 输出的操作集合（FOLD / FLUSH）|记忆管理动作|
|projection $\psi(\cdot)$|$C \leftarrow \text{render}(m_l)$|把 $G_t$ 投影回 working context 的渲染函数|
|Dep$(v_t)$|thought 间的显式依赖边|当前 thought 依赖的前驱 thoughts 集合|
|reasoning agent|主 LLM $M$（参数 $\theta$ 在本文中 frozen）|执行推理与工具调用的主体|

---

## 概览

这篇文章用了主推理 agent 在长 horizon 工具调用过程中产生的 reasoning episodes（每一段含 think + tool call + tool result 的完整 trace），并把这些 episodes 抽象成带依赖关系的 memory graph。  
最后得到了一个独立训练的 executive memory model（叫 MemoBrain），它作为 copilot 与主 LLM 并行运行，负责（1）异步把每个 episode 抽象成一个 thought 节点并连边，（2）当 context 接近预算时对 memory graph 执行 FOLD（折叠已完成子轨迹）和 FLUSH（清除无用步骤）两类操作，从而维护一个紧凑且语义充分的 working context。  
最终目标是在固定 context budget 下让主 agent 的最终任务表现（QA accuracy / Pass@1）尽可能高。优化分两阶段：阶段一用 SFT（teacher 是 DeepSeek V3.2）训练 thought 抽象能力；阶段二用 DPO 训练 FOLD / FLUSH 决策，preference pair 来自不同操作集合下游推理表现的对比。

---

## 1. Problem Setting

- **记忆类型**：本文提出了一类新的 in-process、task-specific 记忆，称为 **executive memory**——既不是 cross-task 的 persistent memory，也不是被动累积的 long-term memory，而是为单个任务从零开始构造、在线演化、主动管理 working context 的记忆。它在框架中既扮演 $m_l$（持久化结构化条目）也部分承担 $g_s$ 的角色（决定 working context $C_N$ 中保留什么）。
- **决策建模**：记忆管理被建模为一个**条件决策过程**——给定当前 memory graph $G_t$，输出一组 executive operations $O_t = \phi(G_t)$。论文未明确写成 MDP，但 DPO 阶段实际上把它当作"在 $G_t$ 上选择操作集"的决策问题（无显式 reward function，只有 preference）。
- **状态空间**：$\mathcal{S}$ = 当前 memory graph $G_t = (V_t, E_t)$，其中节点是 thoughts（含 type ∈ {task, subtask, evidence, summary}、status ∈ {active, inactive}、内容），边是依赖关系。
- **动作空间**：$\mathcal{A} = \lbrace \text{FOLD}(T_{i:j}), \text{FLUSH}(v_k) \rbrace$，即对子图折叠或对单节点 flush。同一节点不能同时被两种操作选中。
- **观测空间**：$\Omega$ = 完成的 episode $x_t = (\tau_t, \omega_t)$（构造阶段）+ 当前 graph 状态（管理阶段）。
- **记忆数据结构**：**有向图**（directed memory graph），节点是自然语言 thought 单元（含角色/内容字段），边带 rationale，节点有 active/inactive 标志。这是一种 **结构化、层级、依赖感知的非参数化外部记忆**。

|组件|论文符号|框架符号|实现|
|---|---|---|---|
|记忆载体|$G_t = (V_t, E_t)$|$m_l$|自然语言 thoughts 组成的有向图|
|写入函数|$\phi(x_t, G_{t-1}) \to v_t$|$g_l$（写入分支）|独立 memory model 异步生成|
|管理函数|$\phi(G_t) \to O_t$|$g_l$（更新分支）|同一 memory model 不同 prompt|
|上下文投影|$\psi(G_{t+1})$|$C_N \leftarrow \text{render}(m_l)$|把 active thoughts 渲染回 context|
|主 agent|reasoning agent|$M_\theta$（frozen）|GLM-4.6 / DeepResearch-30B-A3B|

---

## 2. Training Procedure

- **优化对象**：独立的 memory model $\phi$（即 MemoBrain 本体），主 LLM $M_\theta$ **完全冻结**。$\phi$ 同时承担"记忆构造"和"记忆管理"两个角色，通过不同 system prompt 区分。
- **优化算法**：两阶段——
    - Stage I（构造）：**Supervised Fine-Tuning**，teacher 是 DeepSeek V3.2 生成的"标准 thought"。
    - Stage II（管理）：**Direct Preference Optimization (DPO)**，对每个 $G_t$ 采样多个候选 operation set，按下游推理表现构造 preference pair $(O^+, O^-)$。
- **训练数据**：从 InfoSeek（>50K 复杂 reasoning QA）合成而来，最终得到 37,719 条构造样本 + 3,016 条管理样本。数据合成 API 成本 $389。
- **是否冻结主 LLM**：**是**。被优化的只有 4B/8B/14B 的 memory model 参数。

**核心目标函数**：

构造阶段（Eq. 11，原文符号）： $$ \mathcal{L}_{\text{const}} = -\mathbb{E}_{x_t} \log \phi(v_t \mid x_t) $$

用框架符号改写： $$ \mathcal{L}_{\text{const}} = -\mathbb{E}_{\text{episode}} \log \phi\bigl(\text{thought} \mid \text{episode}, G_{t-1}\bigr) $$

管理阶段（Eq. 12，原文符号）： $$ \mathcal{L}_{\text{mgmt}} = -\mathbb{E} \log \sigma!\left( \log \frac{\phi(O^+ \mid G_t)}{\phi(O^- \mid G_t)} \right) $$

用框架符号改写（$O$ 即 $g_l$ 输出的动作集）： $$ \mathcal{L}_{\text{mgmt}} = -\mathbb{E} \log \sigma!\left( \log \frac{g_l(O^+ \mid m_l^{(t)})}{g_l(O^- \mid m_l^{(t)})} \right) $$

---

## 3. Reward Signal

- **奖励类型**：本质上是 **稀疏 terminal reward**——以下游 QA 是否正确为基础。但在 Stage II 中并不直接使用 reward，而是经由 reward → preference pair 的转换：对同一 $G_t$ 采多组 $O$，按下游推理结果分高低构造 $(O^+, O^-)$。
- **奖励来源**：下游 QA accuracy（用 GPT-4o-2024-08-06 作为 LLM-as-judge 评估 Pass@1）。Stage I 的 SFT label 来自 DeepSeek V3.2 teacher（监督信号，非 reward）。
- **奖励分配**：DPO 是序列级的 implicit reward，操作集 $O_t$ 整体作为一个动作单元被打分，**不做 per-step credit assignment**。
- **辅助奖励 / 正则**：DPO 自带的 KL 隐式正则；论文未引入额外辅助 reward。

---

## 4. Inference Procedure

- **记忆初始化**：每个任务开始时 $G_0 = \emptyset$（task-specific，从零开始）。这与 cross-task memory 完全不同。
- **每步流程**：
    1. 主 agent 执行一个 reasoning episode $x_t = (\tau_t, \omega_t)$（think → tool call → tool result）。
    2. **异步**：MemoBrain 调用 $\phi$ 把 $x_t$ 抽象成 thought $v_t$，并按 Dep$(v_t)$ 与既有节点连边，更新 $G_t \leftarrow G_{t-1} \cup \lbrace v_t \rbrace$。**这一步与主 agent 的下一轮推理并行进行，不阻塞主 loop。**
    3. 检查 working context 大小：若 $\lvert C_t \rvert < \text{budget}$，主 agent 直接继续；若 $\lvert C_t \rvert \geq \text{budget}$，触发 management。
    4. **管理触发时**：$\phi(G_t) \to O_t$（含若干 FOLD 和 FLUSH 操作），应用得到 $G_{t+1}$，再用 $\psi$ 投影出新 working context $C_{t+1}$。
    5. 主 agent 在 $C_{t+1}$ 上继续下一 episode。
- **额外推理策略**：
    - **异步执行**：构造阶段不阻塞主推理（Figure 3a 显示 memorization time 几乎都低于 reasoning time，并行后无端到端开销）。
    - **触发条件**：基于 token 预算（32K 或 64K）的硬阈值，不是连续概率决策。
    - **保留 inactive 节点**：被 FOLD / FLUSH 的节点不真删，只标 inactive 并保留其结构位置（便于未来可能的 reactivation——论文留作 future work）。
- **是否完全由学习驱动**：**部分学习 + 部分规则**。FOLD / FLUSH 的具体目标节点选择是学出来的；触发时机（基于 token 预算）和操作的语法格式（JSON schema）是手工规则。

---

## 5. RQ 分析

### RQ1 (What is memory?)

本文的 memory 是一个 task-specific、依赖感知的有向图 $G_t$，节点为自然语言 thought（episode 的语义抽象），边为显式依赖关系，节点有 active/inactive 状态。它是非参数化的外部记忆（T1，结构化层级图），同时存在一个独立可插拔的 memory model（T5）专门负责构造与管理。本文还引入了一个新概念 executive memory：从 cognitive 角度强调它对 working context 的主动控制，而非被动累积。

### RQ2 (Which component is optimized? Which signal is used?)

被优化的是 独立可插拔的 memory model $\phi$（O1），主 LLM 全程冻结。优化信号有两种：（1）Stage I 用 DeepSeek V3.2 teacher 提供的标准 thought做 SFT；（2）Stage II 用下游 QA 表现构造 preference pair 做 DPO。

### RQ3 (Target of Optimization)

最终追求两个目标：（1）回答准确度（G1，Pass@1）作为主要指标；（2）效率 / context 预算约束下的可持续推理（G2，把 100K+ token 的轨迹压缩到 32K/64K budget 内仍保持甚至超越基线性能）。Table 2 的 search call 数和 Figure 3b 的 token consumption 都属于 G2 的指标。

### RQ4 (Training Signal and RL Algorithm) 

Stage II 使用 DPO，可视为 RLHF 的闭式简化版本。

### RQ5 (How memory evolves, operates?)

推理时 memory 经历两类操作：

- 写入（异步并行）：每个完成的 episode → 一个 thought 节点 + 若干依赖边，单调追加到 $G_t$。
- 管理（触发式）：当 $\lvert C \rvert$ 达预算时，$\phi(G_t)$ 产出 $O_t$ = {FOLD 子轨迹, FLUSH 单节点}，应用后通过 $\psi$ 投影到新的 working context。被操作节点变 inactive 但保留在图中，结构信息不丢。

---

## Conclusion

MemoBrain 提出了一个面向工具增强 agent 的 executive memory 范式：把每一段推理 episode 异步抽象成一个带依赖关系的 thought 节点，构成一张持续生长的 memory graph；当 context 触及预算时，由独立训练的 memory model 决定哪些子图该折叠（FOLD 已完结的子任务）、哪些节点该清空（FLUSH 无用尝试），从而在固定 context budget 下维持长 horizon 推理的连贯性。该 memory model 用两阶段训练（构造用 SFT 蒸馏 DeepSeek V3.2，管理用 DPO 基于下游表现的偏好对），主推理 LLM 全程冻结。在 GAIA、WebWalker、BrowseComp-Plus 三个基准上，MemoBrain 一致提升基础 agent 的表现，且越是难任务（GAIA L3、BrowseComp-Plus）增益越大。其核心贡献是把记忆操作从 in-loop 的副产品提升为可独立训练的 copilot 模块，并让记忆显式、结构化、可主动控制。