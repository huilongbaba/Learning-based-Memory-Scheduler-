# When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning

**Source:** [https://arxiv.org/abs/2602.10560](https://arxiv.org/abs/2602.10560) (February 2026)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$Q$|$q$|问题 / 用户输入|
|$C$|$C$|长上下文（context）|
|$C_t$|无直接对应，属于 $C$ 的分块|第 $t$ 个 chunk|
|$M_t$|$m_s^{(t)}$|第 $t$ 步的文本记忆（工作记忆）|
|$\hat{M}_t$|候选 $m_s$|候选记忆|
|$\hat{A}$|$A_n$|最终回答|
|$A$|ground truth|真实答案|
|$\phi_\theta$|$\pi_\theta$（记忆管理角色）|记忆 agent（memory agent）|
|$\psi_\theta$|$\pi_\theta$（回答角色）|回答 agent（answer agent）|
|$\theta$|$\theta$|模型参数（记忆 agent 与回答 agent 共享）|
|$U_t$|新概念：$u_t$（update gate）|更新门状态（True/False）|
|$E_t$|新概念：$e_t$（exit gate）|退出门状态（True/False）|
|$r^{\text{outcome}}$|$R_{\text{outcome}}$|结果奖励|
|$r^{\text{update}}$|$R_{\text{update}}$|更新门奖励|
|$r^{\text{exit}}$|$R_{\text{exit}}$|退出门奖励|
|$r^{\text{format}}$|$R_{\text{format}}$|格式奖励|
|$T$|agent loop 步数 $N$|总 chunk 数 / 循环步数|

> **新符号说明**：论文引入的 $U_t$（update gate）和 $E_t$（exit gate）是对 RNN-like 循环记忆的门控信号，框架中无直接对应。分别用 $u_t$ 和 $e_t$ 表示。

---

## 概览

这篇文章用长上下文被切分为固定大小的 chunk 序列 $\lbrace C_1, \ldots, C_T \rbrace$，以及一个 RNN-like 的循环记忆更新流程（来源于 MemAgent），在每一步读取一个 chunk 并决定是否更新文本记忆 $m_s$。    
最后得到了一个端到端 RL 训练的单一策略模型 $\pi_\theta$，该模型同时扮演记忆 agent 和回答 agent 两个角色，具备门控记忆更新和提前退出的能力。    
主要优化了两个方面吧：回答准确度 G1。最终答案与 ground truth 的匹配度（EM）；和推理效率G2。通过 update gate 减少无效记忆更新、通过 exit gate 提前退出，实现推理加速。  

---

## 1. Problem Setting

- **记忆类型**：in-chat memory，属于 $m_s$（短期工作记忆）。记忆是一段固定预算内的文本摘要，在处理 chunk 的循环中被逐步更新。没有显式的跨会话长期记忆 $m_l$。
- **决策过程建模**：可视为一个有限步 MDP。每步 $t$，agent 观测当前 chunk $C_t$ 和上一步记忆 $M_{t-1}$，决定三个动作：是否更新记忆（$U_t$）、候选记忆内容（$\hat{M}_t$）、是否退出循环（$E_t$）。
- **状态空间 $\mathcal{S}$**：$(q, C_t, M_{t-1})$——问题、当前 chunk、上一步记忆的组合。
- **动作空间 $\mathcal{A}$**：$(U_t, \hat{M}_t, E_t)$——更新门决策（yes/no）、候选记忆文本、退出门决策（continue/end）。
- **观测空间 $\Omega$**：与状态空间一致（完全可观测）。
- **记忆的数据结构**：自然语言文本条目，受 token 预算约束（$\lvert m_s \rvert \leq 1024$ tokens），每步全量重写。

|核心组件|框架符号|论文对应|
|---|---|---|
|问题|$q$|$Q$|
|上下文|$C$|长文本被切分为 $\lbrace C_1, \ldots, C_T \rbrace$|
|工作记忆|$m_s$|$M_t$（文本记忆，受 1024 token 预算约束）|
|长期记忆|$m_l$|不存在|
|检索算法|$v$|不存在（顺序扫描 chunk）|
|策略|$\pi_\theta$|$\phi_\theta$ 和 $\psi_\theta$（同一模型不同 prompt）|
|最终回答|$A_n$|$\hat{A}$|

> 这篇文章属于T3，核心创新在于对 $g_s$（记忆更新函数）引入门控机制，而非记忆表征本身的改变。

---

## 2. Training Procedure

- **优化的组件**：主 LLM 本身（$\theta$），即 $\pi_\theta$。记忆 agent $\phi_\theta$ 和回答 agent $\psi_\theta$ 共享同一组参数，仅通过 prompt 区分角色。
- **优化算法**：Multi-Conv DAPO（GRPO 在多轮对话场景的扩展），属于 PPO 家族的 clipped policy gradient 方法。
- **训练数据来源**：在线交互——模型在训练数据上 rollout 产生多组轨迹，采样 $N=16$ 个 rollout 进行 group relative advantage 计算。训练数据与 MemAgent 一致。
- **是否冻结 LLM 参数**：否，全参数训练（基于 Qwen2.5-3B/7B-Instruct）。

**核心训练目标函数**：

论文原始公式（Eq. 3）：

$$J(\theta) = \mathbb{E} \left[ \frac{1}{\sum_{g=1}^{G} \sum_{t=1}^{T_g} \lvert o_{g,t} \rvert} \sum_{g=1}^{G} \sum_{t=1}^{T_g} \sum_{i=1}^{\lvert o_{g,t} \rvert} \left( \ell_{g,t,i}^{\text{clip}} - \beta D_{KL}(\pi_\theta \Vert \pi_{\text{ref}}) \right) \right]$$

框架符号标注：这是对策略 $\pi_\theta$ 在多轮 trajectory $T = (o_{g,1}, o_{g,2}, \ldots, o_{g,T_g})$ 上的 clipped PPO 损失，其中每个 $o_{g,t}$ 对应一次记忆更新或回答的 token 序列。

**优势函数计算**（Eq. 13）：

$$\hat{A}_{g,t,i} = \alpha \hat{A}_{g,t,i}^{\text{traj}} + (1 - \alpha) \hat{A}_{g,t,i}^{\text{turn}}$$

其中 trajectory-level advantage 基于 $(r^{\text{outcome}} + r^{\text{exit}} + r^{\text{format}})$ 的 group 相对值，turn-level advantage 基于 $r^{\text{update}}$ 的 group 相对值。默认 $\alpha = 0.9$。

> 本文的关键训练贡献在于将门控行为的学习信号（$r^{\text{update}}$ 和 $r^{\text{exit}}$）与最终回答的奖励信号解耦，并通过 $\alpha$ 参数平衡两者。

---

## 3. Reward Signal

- **奖励类型**：混合——既有 sparse terminal reward（$r^{\text{outcome}}$），也有 dense step-level reward（$r^{\text{update}}$），以及 trajectory-level 的 $r^{\text{exit}}$ 和 $r^{\text{format}}$。
- **奖励来源**：
    - $r^{\text{outcome}}$：EM（Exact Match），$\mathbb{I}(\text{is\_equiv}(A, \hat{A}))$，取值 0/1。
    - $r^{\text{update}}$：基于 ground truth evidence 标注的规则奖励（chunk 是否包含 evidence），取值 +1/-1。
    - $r^{\text{exit}}$：基于 ground truth 最后一条 evidence 所在位置的规则奖励，取值 $\lbrace -0.75, 0, -0.5 \rbrace$。
    - $r^{\text{format}}$：格式正确性检查，取值 0/1。
- **奖励分配**：
    - $r^{\text{outcome}}, r^{\text{exit}}, r^{\text{format}}$：同一 trajectory 内所有 turn 共享相同值（trajectory-level）。
    - $r^{\text{update}}$：每个 turn 独立计算（turn-level），通过 $\alpha$ 与 trajectory-level advantage 加权组合。
- **辅助奖励/正则项**：KL 散度正则 $\beta D_{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ 防止策略偏离参考模型太远。

> $r^{\text{update}}$ 需要知道每个 chunk 是否包含 evidence，这意味着训练依赖于 evidence-level 的标注信息。这在实际应用中可能是一个限制，因为并非所有任务都有现成的 evidence 标注。

---

## 4. Inference Procedure

- **记忆初始化**：$M_0 = \text{None}$（空记忆）。
- **每步决策流程**：
    1. 将 $(q, C_t, M_{t-1})$ 拼接为 prompt 输入记忆 agent $\phi_\theta$。
    2. $\phi_\theta$ 生成结构化输出：`<think>推理</think> <check>yes/no</check> <update>候选记忆</update> <next>continue/end</next>`。
    3. 若 $U_t = \text{True}$，则 $M_t \leftarrow \hat{M}_t$；否则 $M_t \leftarrow M_{t-1}$。
    4. 若 $E_t = \text{True}$ 且启用 exit gate，则跳出循环。
    5. 循环结束后，回答 agent $\psi_\theta$ 基于 $(q, M_t)$ 生成最终回答 $\hat{A}$。
- **推理时额外策略**：
    - 提供两种推理模式：w/ EG（启用退出门）和 w/o EG（禁用退出门，扫描全部 chunk）。
    - 对于需要全文信息的任务（如 multi-values），使用 w/o EG 模式。
    - 无 TopK 检索（顺序扫描），但可配合 reranking 技术将关键 evidence 提前。
- **推理策略驱动**：核心由学习得到的 $\pi_\theta$ 驱动（门控决策完全由模型生成），但 w/ EG vs. w/o EG 的选择是手工规则。

> 推理时的两种模式选择（w/ EG vs. w/o EG）目前需要人为判断任务类型，这在实际部署中引入了额外的工程决策。未来可能需要让模型自主学习何时启用退出门。

---

## 5. RQ 分析

### RQ1 (What is memory?)

GRU-Mem 的记忆是一个受 token 预算约束的固定长度文本摘要（$\lvert m_s \rvert \leq 1024$ tokens），在 RNN-like 的循环中被全量重写。属于 T3 。

### RQ2 (Which component is optimized? Which signal is used?)

优化对象是主 LLM 本身（O2），记忆 agent 和回答 agent 共享同一模型参数 $\theta$。使用 Multi-Conv DAPO（GRPO 变体）进行端到端 RL 训练。奖励信号包括四种：outcome reward（EM）、update reward（基于 evidence 标注的 step-level 奖励）、exit reward（基于最后 evidence 位置的 trajectory 奖励）、format reward。

### RQ3 (Target of Optimization)

主要优化目标是 G1（回答准确度，EM）和 G2（效率，推理时间/token 消耗）。通过 update gate 减少不必要的记忆更新以降低 token 开销，通过 exit gate 提前终止循环以减少推理时间。实验显示最高 400% 的推理加速。

### RQ4 (How memory evolves, operates?)

记忆在运行时按 chunk 顺序演化：每步读取一个 chunk，模型决定是否更新记忆（写）和是否退出循环（控制流）。更新操作是全量重写（而非增量追加）。退出后，最终记忆被传递给回答 agent 生成答案。记忆的读操作隐含在回答 agent 的 context 中，最终记忆直接作为回答 agent 的输入。

---

## Conclusion

GRU-Mem 针对 MemAgent 在长上下文推理中的两个关键缺陷，记忆爆炸和缺乏退出机制，提出了门控循环记忆框架。它在 RNN-like 的 chunk-by-chunk 记忆更新流程中引入了两个文本控制的门：update gate 决定是否在当前 chunk 上更新记忆，exit gate 决定是否在收集到足够证据后提前终止循环。通过设计针对性的奖励信号（$r^{\text{update}}$ 和 $r^{\text{exit}}$）并使用解耦的优势函数计算，模型能够在端到端 RL 训练中同时学会正确的更新和退出行为。实验表明，GRU-Mem 在多个长上下文 QA 任务上普遍优于 MemAgent，并实现了最高 400% 的推理加速。