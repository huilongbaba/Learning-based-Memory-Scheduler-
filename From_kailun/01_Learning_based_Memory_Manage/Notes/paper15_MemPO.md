# MemPO: Self-Memory Policy Optimization for Long-Horizon Agents

**Source:** [https://arxiv.org/abs/2603.00680](https://arxiv.org/abs/2603.00680) (March 2026)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$q$|$q$|用户问题 / 查询|
|$a_{gt}$|$A_n^*$|ground truth 最终答案|
|$a_{pred}$|$A_n$|预测的最终答案|
|$\tau = \lbrace s_1, s_2, \ldots, s_T \rbrace$|$T$|完整交互轨迹（trajectory）|
|$s_t^{mem}$|$m_s^{(t)}$|第 $t$ 步的记忆摘要（短期/工作记忆）|
|$s_t^{think}$|$t^{(t)}$|第 $t$ 步的推理过程（thought）|
|$s_t^{call}$|$a^{(t)}$|第 $t$ 步的工具调用（action）|
|$s_t^{resp}$|$o^{(t)}$|第 $t$ 步的环境返回（observation）|
|$\pi_\theta$|$\pi$|策略模型（policy）|
|$\theta$|$\theta$|模型参数|
|$R^T(\tau_i)$|—|轨迹级奖励（trajectory-level reward）|
|$R^M(\tau_i(s_t^{mem}))$|—|记忆级奖励（memory-level reward，框架中未定义，新增）|
|$A^T(\tau_i)$|—|轨迹级优势（trajectory-level advantage）|
|$A^M(\tau_i(s_t^{mem}))$|—|记忆级优势（memory-level advantage，新增）|

> $R^M$ 和 $A^M$ 是本文提出的记忆级奖励和优势，用于衡量每步 `<mem>` action 中保留信息的有效性。

---

## 概览

这篇文章用 agent 在多轮交互中产生的 trajectory，每步轨迹包含记忆摘要 `<mem>`、推理 `<think>`、工具调用 `<tool_call>` 和环境返回 `<information>` 四个组件。    
最后得到一个经过 RL 微调的 LLM，具备自主记忆管理能力，能在每步交互中主动压缩和重组历史信息。  
优化了 G1 最终回答的准确度（F1 / EM），同时优化了 G2，即大幅降低 token 消耗（效率）。

---

## 1. Problem Setting

- **记忆类型**：in-chat memory（单次任务交互内的记忆），属于 $m_s$（短期/工作记忆）。每步的 `<mem>` 是对之前所有交互历史的压缩摘要，推理时仅使用上一步的 `<mem>` 作为上下文，而非完整历史。不涉及 cross-chat 的 $m_l$。
    
- **决策过程建模**：隐式 MDP。每步状态 $s_t$ 包含四个组件 $(s_t^{mem}, s_t^{think}, s_t^{call}, s_t^{resp})$，agent 在给定 $(q, s_{t-1}^{mem})$ 的条件下生成 $s_t$。
    
- **状态空间 $\mathcal{S}$**：当前步的上下文 $(q, s_{t-1}^{mem})$，即用户问题加上上一步的记忆摘要。
    
- **动作空间 $\mathcal{A}$**：agent 在每步生成的输出 $s_t = (s_t^{mem}, s_t^{think}, s_t^{call})$，包含三种 action 类型。
    
- **观测空间 $\Omega$**：$s_t^{resp}$，即工具调用返回的信息。
    
- **记忆的数据结构**：自然语言文本段落，由 `<mem>...</mem>` 标签包裹。每步记忆是对之前交互历史的全量重写（非增量追加），类似于固定长度压缩记忆的思路，但未施加显式 token 长度约束。
    

|核心组件|框架符号|本文对应|
|---|---|---|
|用户输入|$q$|$q$（question）|
|短期记忆|$m_s$|$s_t^{mem}$（`<mem>` 内容）|
|长期记忆|$m_l$|不涉及|
|推理过程|$t$|$s_t^{think}$（`<think>` 内容）|
|动作|$a$|$s_t^{call}$（`<tool_call>` 内容）|
|观测|$o$|$s_t^{resp}$（`<information>` 内容）|
|策略|$\pi$|$\pi_\theta$（主 LLM 本身）|
|检索算法|$v$|不涉及（无外部检索）|


---

## 2. Training Procedure

- **优化组件**：主 LLM 本身的参数 $\theta$（O2）。直接对 Qwen2.5-7B 进行 RL 微调，使模型学会在 `<mem>` 中保留关键信息。
    
- **优化算法**：基于 GRPO 的改进版本（MemPO）。在标准 GRPO 的轨迹级优势之上，额外引入记忆级优势 $A^M$，对 `<mem>` token 提供更精细的梯度信号。
    
- **训练数据来源**：
    
    - **Behavior Cloning 阶段**：使用 GPT-4.1 在公开数据集上推理生成约 10K 条包含记忆的轨迹，过滤掉答案错误的轨迹，用于 SFT 冷启动。
    - **RL 阶段**：在线交互（online rollout），使用 HotpotQA 和 NQ 合成的 2-objective 任务作为训练集，local wiki search engine 作为工具。
- **是否冻结 LLM 参数**：否。主 LLM 参数 $\theta$ 全量参与优化。
    
- **核心训练目标函数**：
    

策略优化目标（论文公式 10，统一符号）：

$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\frac{1}{\lvert \tau_i \rvert}\sum_{k=1}^{\lvert \tau_i \rvert}\min\left(\gamma_{i,k}A_{i,k}, \text{clip}(\gamma_{i,k}, 1-\epsilon, 1+\epsilon)A_{i,k}\right) - \beta \mathcal{D}_{KL}(\pi_\theta \Vert \pi_{ref})\right]$$

其中 token 级优势 $A_{i,k}$（论文公式 9）：

$$A_{i,k} = \begin{cases} A^T(\tau_i) + A^M(\tau_i(s_t^{mem})), & \text{if } \tau_{i,k} \in \tau_i(s_t^{mem}) \ A^T(\tau_i), & \text{otherwise} \end{cases}$$

记忆级奖励 $R^M$（论文公式 4-5，框架符号 $r_m^{(t)}$）：

$$R^M(\tau_i(s_t^{mem})) = P[s^{ans} \mid \tau_i(s_t^{mem})] - \epsilon$$

其中 $P[s^{ans} \mid \tau_i(s_t^{mem})] = \sqrt[L]{\prod_{l=1}^{L} \pi_\theta(a_l \mid q, \tau_i(s_t^{mem}), a_{< l})}$

> 这篇文章的核心创新在于记忆级优势 $A^M$：通过条件概率衡量 `<mem>` 中保留信息对生成正确答案的支持程度。

---

## 3. Reward Signal

- **奖励类型**：
    
    - **轨迹级**：sparse terminal reward（$R^T$），仅在轨迹结束时基于最终答案正确性给出 0/1 奖励。
    - **记忆级**：dense step-level reward（$R^M$），在每步为 `<mem>` action 提供基于条件概率的连续奖励。
- **奖励来源**：
    
    - $R^T$：基于答案正确性（EM）+ 输出格式检查。正确且格式合规为 1，否则为 0。
    - $R^M$：基于条件概率 $P[s^{ans} \mid s_t^{mem}]$，即给定当前步记忆内容时模型生成正确答案的概率的几何平均值。使用 $P[s^{ans} \mid s_{< t}]$ 作为 bias 项 $\epsilon$。
- **奖励分配到各步/token**：
    
    - 轨迹级优势 $A^T$ 均匀分配给同一轨迹的所有 token。
    - 记忆级优势 $A^M$ 仅额外加给 `<mem>` 标签内的 token（credit assignment 机制）。
    - 两者通过加法组合：`<mem>` 内 token 获得 $A^T + A^M$，其余 token 仅获得 $A^T$。
- **辅助奖励或正则项**：KL 散度正则项 $\beta \mathcal{D}_{KL}(\pi_\theta \Vert \pi_{ref})$，防止策略偏离参考模型过远。
    

> $R^M$ 的设计本质上是用"记忆内容对正确答案的支持程度"作为记忆质量的代理指标。减去 bias $\epsilon = P[s^{ans} \mid s_{< t}]$ 使得奖励衡量的是记忆相对于前序轨迹的增量信息贡献。

---

## 4. Inference Procedure

- **推理时记忆初始化**：第一步无记忆（$s_0^{mem}$ 为空），agent 直接基于 $q$ 开始交互。
    
- **每步 agent 的决策流程**：
    
    1. **输入构造**：将 $(q, s_{t-1}^{mem})$ 作为当前步的上下文（仅使用上一步的记忆摘要，丢弃更早的所有历史）。
    2. **记忆生成**：agent 在 `<mem>...</mem>` 中输出对历史信息的压缩摘要 $s_t^{mem}$。
    3. **推理**：在 `<think>...</think>` 中进行推理 $s_t^{think}$。
    4. **工具调用**：在 `<tool_call>...</tool_call>` 中生成工具调用 $s_t^{call}$。
    5. **观测接收**：环境返回 $s_t^{resp}$（`<information>...</information>`）。
    6. **循环**：重复直到 agent 生成 `<answer>...</answer>`。
    
    形式化：$\pi_\theta(s_t \mid q, s_{t-1}^{mem})$
    
- **推理时额外策略**：无 TopK 检索（不依赖外部记忆库），无多轮 replan。核心策略是"只保留上一步的 `<mem>` 内容作为上下文"，实现上下文的固定规模控制。
    
- **推理策略驱动方式**：完全由学习得到的 $\pi_\theta$ 驱动，无手工规则。`<mem>` 的内容由模型自主决定保留什么、压缩什么。

---

## 5. RQ 分析

### RQ1 (What is memory?)

MemPO 中的记忆是 T3， in-chat 的 $m_s$（短期工作记忆），表现为每步由 LLM 自主生成的自然语言摘要（`<mem>` 标签内容）。该记忆在每步被全量重写，不涉及外部存储或跨会话记忆。记忆管理的策略（如何压缩、保留什么）被内化为模型自身的能力，但记忆的载体仍然是显式的自然语言文本（`<mem>` 块）。

### RQ2 (Which component is optimized? Which signal is used?)

优化对象是主 LLM 本身的参数 $\theta$（O2）。使用两层信号：(1) 轨迹级 sparse reward（答案正确性）；(2) 记忆级 dense reward（条件概率衡量记忆信息量）。两者通过优势加法组合。

### RQ3 (Target of Optimization)

优化目标同时涵盖回答准确度（G1，F1/EM）和效率（G2，token 消耗）。通过记忆压缩实现 token 消耗大幅降低（约 67-73%），同时 F1 提升约 7-26 个百分点。

### RQ4 (How memory evolves, operates?)

推理时，每步记忆 $s_t^{mem}$ 由模型基于上一步记忆 $s_{t-1}^{mem}$ 和当前步的新信息自主重写。上下文窗口固定为 $(q, s_{t-1}^{mem})$，历史信息通过逐步压缩传递。记忆的读写完全由模型内部决策驱动，无外部检索或手工规则。

---

## Conclusion

MemPO 提出了一种将记忆管理内化为 LLM agent 自身能力的方法，而非依赖外部记忆模块或 RAG。其核心创新是在 GRPO 框架中引入记忆级优势（memory-level advantage），通过条件概率衡量每步 `<mem>` 内容对正确答案的支持程度，从而为记忆生成提供 dense reward 信号，有效解决了长时域交互中的 credit assignment 问题。记忆的载体仍然是显式的自然语言文本，但记忆的生成、压缩和更新策略 $g_s$ 不是由规则或外部模块驱动，而是通过端到端的学习训练到主模型参数 $\theta$ 中。主模型在交互过程中自主决定记忆的更新，这种决策能力经过面向任务目标的优化。