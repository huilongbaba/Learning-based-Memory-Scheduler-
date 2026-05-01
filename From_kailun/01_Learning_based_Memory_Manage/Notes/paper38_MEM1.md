# MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents

**Source:** [https://openreview.net/pdf?id=XY8AaxDSLb](https://openreview.net/pdf?id=XY8AaxDSLb) (2026-04-11 修订，Accepted at ICLR 2026 poster，8666)

---

## 符号映射表

|论文原始符号|框架统一符号|含义|
|---|---|---|
|$Q$|$q$|任务/问题（augmented multi-objective query）|
|$S_t$ / `<IS>`|$m_s$|内部状态（Internal State），即短期记忆 + 推理痕迹的统一表征|
|$A_t$ / `<query>` / `<answer>`|$a$|agent 在第 $t$ 轮发出的动作（搜索查询或最终答案 $A_n$）|
|$O_t$ / `<info>`|$o$|环境/工具反馈（RAG 检索结果、WebShop 页面观测）|
|$\pi_\theta$|$\pi$|LLM 的策略|
|$\theta$|$\theta$|LLM 参数（被 PPO 更新）|
|$\tau = {(S_i, A_i, O_i)}$|$T$|轨迹（trajectory）|
|$r(\cdot)$|$r$|可验证奖励（EM / F1 / WebShop env reward）|
|—（不存在显式外部记忆库）|$m_l$|本文**不**维护独立的长期记忆库；RAG corpus 是环境，不是 agent 管理的 $m_l$|
|隐式：$S_{t+1} = \pi_\theta(\cdot \mid S_t, A_t, O_t)$|$g_s$|短期记忆更新函数（由主 LLM 自身在每轮 rollout 中执行）|
|检索由环境提供（Faiss + E5 / Serper API）|$v$|检索算法**不被优化**，是固定的外部工具|

---

## 概览

这篇文章用了一个增广的多目标 QA 数据集（把多个多跳问题拼成一个复合查询）和 WebShop 网页购物环境，构造出需要大量 turn 才能完成的长 horizon 交互轨迹。   
最后得到了一个端到端 RL 训练的 7B 语言模型 agent，无需任何外部记忆模块或额外参数，便可在任意长 horizon 任务中维持近乎恒定的上下文长度。  
直接优化的是任务最终答案的正确性（G1）。记忆效率不是显式 reward，但通过 rollout 中每轮裁剪掉旧的 $(S_i, A_i, O_i)$，只保留最新内部状态的强约束，agent 被结构性地逼迫学会把信息压缩进 $S_{t+1}$ 才能拿到 reward。效率变成架构带来的副收益。

---

## 1. Problem Setting

- **Memory 类型**：短期 in-chat memory $m_s$（即 internal state $S_t$）。本文**不**维护独立的 cross-chat / 长期外部记忆 $m_l$；RAG 知识库被当作环境而非 agent 管理的 memory。
- **决策建模**：MDP $(\mathcal{S}, \mathcal{A}, \pi, r)$，状态 = 当前 prompt 中保留的 token 序列，动作 = 下一个 token（用 XML 标签隔出 `<IS>`/`<query>`/`<answer>`）。
- **状态空间 $\mathcal{S}$**：$(q, S_{t-1}, A_{t-1}, O_{t-1}, S_t, A_t, O_t)$ 的 token 序列——**最多保留两轮 $S$、两轮 $A$、一轮 $O$**，旧的全部从 context 裁剪。
- **动作空间 $\mathcal{A}$**：词表 $\mathcal{V}$ 上的 token，分为三类语义：reasoning/memory（IS）、查询（query）、最终答案（answer）。
- **观测空间 $\Omega$**：RAG top-3 文档段 / Serper top-10 web snippet / WebShop 页面文本。
- **数据结构**：$m_s$ 是**自然语言 token 序列**，由 `<IS>...</IS>` 包裹；不是向量库、不是 KV 对、不是图。属于"显式、压缩、与推理融合"的形态。

|框架组件|本文实现|
|---|---|
|$m_s$|`<IS>...</IS>` 内的自然语言文本|
|$m_l$|**不存在**（RAG 是环境的一部分）|
|$g_s$|主 LLM 在每轮自身 rollout 中重写 $S_{t+1}$|
|$g_l$|不适用|
|$v$|外部 Faiss / Serper（固定，不参与训练）|
|$\pi$|$\pi_\theta$（待训练的主 LLM）|

> 所有过去信息要么内化进 $S_{t+1}$，要么彻底丢失。

---

## 2. Training Procedure

- **优化对象**：主 LLM 的全部参数 $\theta$（Qwen2.5-7B Base，actor + critic 双模型，actor lr=1e-6，critic lr=1e-5）。**不冻结**主模型，**没有**独立的辅助 memory module。
- **优化算法**：**PPO**（标准 actor-critic + clipped surrogate）。文中说选 PPO 而非 GRPO 是因为 PPO 提供 token-level advantage，对长 multi-turn 轨迹的训练更稳定。
- **训练数据**：在线 rollout——每个 batch 从 augmented 2-objective QA 数据集采样任务，agent 与 RAG/WebShop 环境实时交互产生轨迹。
- **关键技术（Masked Trajectory）**：rollout 中 context 被动态裁剪 → 每轮 $\tau_i = (S_i, A_i, O_i)$ 不属于同一条原始 trajectory。为了能正确算 PPO 的 policy gradient 和 critic value，作者把所有 $\tau_i$ "缝合" 成 $\tau_\text{full} = (S_1, A_1, O_1, \ldots, S_n, A_n)$，并施加一个**二维 attention mask**：

$$ \text{Attn}_t = \mathbf{1}_{a \in \lbrace S_{i-1}, A_{i-1}, O_{i-1}, S_i, A_i, O_i \rbrace} \times \mathbf{1}_{a \in \lbrace a_k \mid k \in \lbrace 1, \ldots, t \rbrace \rbrace} $$

确保第 $t$ 个 token 的 attention 只覆盖它生成时实际看到的 token。框架统一表述：每步 $\nabla_\theta \pi_{\theta, q}(a_t \mid s_t)$ 在 mask 后正确计算。

- **训练目标（PPO clipped surrogate）**：

$$ \mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \Big\lbrack \min\big(\rho_t(\theta) \hat{A}_t,\ \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big) \Big\rbrack $$

其中 $\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$，$\hat{A}_t$ 由 critic 估计。

- **目标（论文形式）**：

$$ \arg\max_\theta\ \mathbb{E}_{q \in \mathcal{Q},\ \tau \sim \pi_{\theta,q}} \Big\lbrack \sum_{(a_t, s_t) \in \tau} r(s_t) \Big\rbrack $$

> Masked trajectory 解决了"训练时 context 在变"与"PPO 需要一次性 forward pass 算 advantage"的冲突。未来的动态上下文管理或许可以参考这个做法。

---

## 3. Reward Signal

- **奖励类型**：**sparse terminal reward**（QA 任务每个 sub-question 答对 +1；WebShop 用环境给的 final reward）。
- **奖励来源**：
    - QA 环境：rule-based **Exact Match**（XML 格式不对直接 0 分；多目标任务按分号切分逐题计分）。
    - WebShop 环境：环境内置 reward 函数。
- **Credit assignment**：通过 PPO 的 critic + GAE 把 terminal reward 反传到 token-level advantage，无显式 dense step reward。
- **辅助奖励/正则**：**作者明确反对加 format reward**——附录 F.2 实验显示加 format penalty 会让模型走捷径（少搜索以避免格式错误），最终 EM 从 0.709 跌到 0.466。也没有 length/efficiency penalty。

---

## 4. Inference Procedure

- **记忆初始化**：第 0 轮 $S_0 = \emptyset$，prompt 仅含任务 $q$ 和系统指令。
- **每步流水线（见 Alg. 1）**：
    1. 当前 context = `[q, S_{t-1}, A_{t-1}, O_{t-1}]`（最多保留上一轮的三元组）；
    2. LLM 生成 `<IS> S_t </IS>` —— 在此 token 序列中**同时完成 reasoning 和 memory consolidation**；
    3. 接着生成 `<query> A_t </query>` 或 `<answer> A_n </answer>`；
    4. 若是 query：调用 Faiss/Serper 取回 $O_t$，附入 context；若是 answer：返回，结束；
    5. **关键裁剪**：$S_{t-1}, A_{t-1}, O_{t-1}$ 从 context 中移除，只保留 $q, S_t, A_t, O_t$ 进入下一轮。
- **额外推理策略**：
    - RAG top-3 / Serper top-10（固定 hyperparameter，不学习）；
    - 推理温度 0.01，几乎贪心；
    - HINT 注入：`<info>` 中加入 "You have {T-t} turns left." 作为 horizon 提示。
- **学习 vs. 手工**：策略 $\pi_\theta$ 完全由 RL 学到（如何写 IS、何时搜索、何时停止）；context 的裁剪规则、retrieval top-k、XML 格式规范是**手工**的硬编码 rollout 框架。

---

## 5. RQ 分析

### RQ1 (What is memory?)

文章中的记忆是一段被 `<IS>` 包裹的的自然语言 token 序列。它显式可读、每轮被 LLM 自身完全重写、与推理痕迹融为一体。MEM1 不维护任何独立的长期记忆库，所有跨轮信息要么被压进 $S_{t+1}$ 要么被丢弃。对应 T2（固定长度压缩记忆）。

### RQ2 (Which component is optimized? Which signal is used?)

优化对象是主 LLM 自身的全部参数 $\theta$（Qwen2.5-7B Base 起步，actor + critic 全量微调），属于 O2a 单 LLM 优化。训练信号是 sparse 的、可验证的最终任务 reward。

### RQ3 (Target of Optimization)

显式优化目标只有 G1（回答准确度）。效率（peak token、dependency、inference time）只是评估指标和论文卖点，不是 reward 的一部分。但 rollout 的裁剪结构使 G2（效率）成为达成 G1 的必要条件。

### RQ4 (Training Signal and RL Algorithm)

A1: PPO。论文显式选择 PPO 而非 GRPO，理由是 PPO 提供 token-level advantage，对 multi-turn 长轨迹训练更稳定。配套创新是 2D attention mask 处理动态 context 下的 PPO 梯度计算。

### RQ5 (How memory evolves, operates?)

每轮：旧的 $(S_{t-1}, A_{t-1}, O_{t-1})$ 仍可见 → LLM 生成新的 $S_t$（这是 $g_s$ 的实例化，由主 LLM 自我执行）→ 生成动作 $A_t$ → 拿到 $O_t$ → 物理删除 $(S_{t-1}, A_{t-1}, O_{t-1})$，只把 $q, S_t, A_t, O_t$ 带进下一轮。读 = 直接看 IS 中的 token；写 = 重写整个 IS；遗忘 = 上一轮的内容如果没被 LLM 主动复制进 $S_t$ 就永久丢失。没有检索机制 $v$。因为没有可检索的 memory store。

---

## Conclusion

MEM1 提出了一种"用一个统一的 internal state 同时承担推理和记忆"的方案：每一轮 LLM 自己重写一段被 `<IS>` 包裹的自然语言短文本，旧的对话历史从 context 中物理删除，agent 在长 horizon 任务上的 context 长度因此近乎恒定。训练采用 PPO，奖励信号只看最终答案对不对，但 rollout 的裁剪规则强制了 agent 必须学会把有用信息压进 IS，否则下一轮就拿不到 reward。在 16-objective QA 任务上，MEM1-7B 比 Qwen2.5-14B-Instruct 准确率高 3.5×，峰值 token 用量低 3.7×，在 WebShop 上也超过了 AgentLM-13B。  