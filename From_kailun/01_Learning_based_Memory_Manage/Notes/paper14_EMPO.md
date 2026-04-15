# Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization (EMPO²)

**Source:** [https://arxiv.org/pdf/2602.23008](https://arxiv.org/pdf/2602.23008) (ICLR 2026, Feb 2026)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$u$|$q$|任务描述 / human input|
|$\tau$|$T$|轨迹 trajectory|
|$\pi_\theta$|$\pi$|模型策略|
|$\theta$|$\theta$|模型参数（一致）|
|$a_t$|$a$|动作 action（一致）|
|$s_t$ / observation|$o$|观测 observation|
|$M$ (memory buffer)|$m_l$|长期记忆（外部 tips 缓冲区）|
|tips$_t$|$v(q, m_l)$ 的输出|检索到的记忆条目|
|Retr$(o_t; M)$|$v$|检索算法|
|$r_t$|reward|环境奖励|
|$r_{\text{intrinsic}}$|（新增）$r_{\text{int}}$|基于状态新颖度的内在奖励|
|tip-generation prompt|（新增）$p_{\text{reflect}}$|反思 prompt，用于生成 tips|
|$p$|—|memory rollout 概率（超参数）|
|$q$|—|off-policy update 概率（超参数）|
|$\delta$|—|token masking 阈值（超参数）|

---

## 概览

这篇文章利用 agent 与环境交互产生的历史 trajectory，通过 LLM 自身的反思能力生成 tips（自然语言形式的经验总结），存入外部记忆缓冲区 $m_l$，并在后续 rollout 中通过 embedding 相似度检索 top-10 条相关 tips 注入 prompt。  
最后得到一个经过 RL 训练的、内化了记忆探索能力的 LLM agent。测试时不需要外部记忆，模型本身已将 tips 中的知识通过 off-policy distillation 蒸馏进参数 $\theta$ 中。同时，该模型也保留了在新任务中通过 memory 进行 few-shot 适应的能力。  
优化的是 agent 在 multi-step 环境中的任务完成奖励（累积环境 reward），同时通过内在奖励 $r_{\text{int}}$ 鼓励状态空间的探索。

---

## 1. Problem Setting

- **记忆类型**：cross-episode memory（跨 rollout 的经验记忆），属于 $m_l$（长期记忆）。每条记忆是由 $\pi_\theta$ 在 episode 结束时通过反思 prompt 生成的自然语言 tip。
- **决策过程**：建模为标准 MDP。给定任务 $u$，agent 在每步 $t$ 观测 $s_t$，生成动作 $a_t$，获得奖励 $r_t$ 和下一状态 $s_{t+1}$。
- **状态空间 $\mathcal{S}$**：环境的文本观测（ScienceWorld 的房间描述 / WebShop 的 HTML 页面）。
- **动作空间 $\mathcal{A}$**：自然语言动作（如 "go to workshop"、"connect battery to wire"）。
- **观测空间 $\Omega$**：与 $\mathcal{S}$ 相同（fully observable text）。
- **记忆数据结构**：自然语言条目集合，配 embedding 索引，按 cosine similarity 检索，按 reward score 排序，top-10 返回。

|核心组件|框架符号|论文实现|
|---|---|---|
|任务输入|$q$|任务描述 $u$|
|策略|$\pi$|$\pi_\theta$（Qwen2.5-7B-Instruct + RL）|
|长期记忆|$m_l$|Memory buffer $M = \lbrace \text{tip}_1, \text{tip}_2, \ldots \rbrace$|
|检索算法|$v$|cosine similarity > 0.5，按 score 排序取 top-10|
|记忆更新|$g_l$|$\pi_\theta$ 在 episode 结束时生成 tip 并追加到 $M$|
|轨迹|$T$|$\tau = (u, a_1, r_1, s_1, \ldots, s_T)$|
|最终输出|$A_n$|agent 在环境中的最终 action 序列及对应 reward|


---

## 2. Training Procedure

- **优化组件**：主 LLM 本身（$\theta$），即 Qwen2.5-7B-Instruct 的全部参数。
- **优化算法**：基于 GRPO 的混合 on-policy 和 off-policy 优化。
    - **On-policy（无 memory）**：标准 GRPO，$\rho_\theta = \frac{\pi_\theta(a_t \mid s_t, u)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t, u)}$
    - **On-policy（有 memory）**：$\rho_\theta = \frac{\pi_\theta(a_t \mid s_t, u, \text{tips}_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t, u, \text{tips}_t)}$
    - **Off-policy**：rollout 时用 tips，update 时去掉 tips，$\rho_\theta = \frac{\pi_\theta(a_t \mid s_t, u)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t, u, \text{tips}_t)}$，实现 reward-guided knowledge distillation。
- **训练数据来源**：完全在线交互，无 SFT warm-start，无人工标注。
- **LLM 参数是否冻结**：否，$\theta$ 是被直接优化的目标。

**核心训练目标函数**（论文 Eq. 2）：

$$\mathcal{L} = \mathbb{E}_{u \sim p(U), \lbrace \tau^{(i)} \rbrace \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} \min\left( \rho_\theta^{(i,t)} A(a_t^{(i)}),\ \text{clip}(\rho_\theta^{(i,t)}, 1-\epsilon, 1+\epsilon) A(a_t^{(i)}) \right) \cdot \mathbf{1}_{\pi_\theta(a_t^{(i)} \mid s_t^{(i)}, u) \geq \delta} \right] - \beta D_{KL}(\pi_\theta \Vert \pi_{\text{ref}})$$

其中：

- $A(a_t^{(i)}) = \frac{R^{(i)} - \frac{1}{N}\sum_{j=1}^{N} R^{(j)}}{\sigma(R)}$（GRPO group-relative advantage）
- $R^{(i)} = \sum_{t=1}^{T} r_t^{(i)}$（trajectory return）
- $\mathbf{1}_{\pi_\theta(\cdot) \geq \delta}$：低概率 token masking，稳定 off-policy 训练

**统一符号标注**：

$$\mathcal{L} = \mathbb{E}_{q \sim p(U), \lbrace T^{(i)} \rbrace \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{NT} \sum_{i,t} \min\left( \rho_\theta^{(i,t)} A(a_t^{(i)}),\ \text{clip}(\rho_\theta^{(i,t)}, 1-\epsilon, 1+\epsilon) A(a_t^{(i)}) \right) \cdot \mathbf{1}_{\pi(a \mid o, q) \geq \delta} \right] - \beta D_{KL}(\pi \Vert \pi_{\text{ref}})$$


---

## 3. Reward Signal

- **奖励类型**：
    - **外在奖励**：sparse terminal reward（每步有 $r_t$，但主要通过 trajectory return $R = \sum_t r_t$ 使用）
    - **内在奖励**：dense step-level novelty reward $r_{\text{intrinsic}} = \frac{1}{n}$，其中 $n$ 是与当前状态相似的历史状态数量
- **奖励来源**：环境反馈（ScienceWorld 返回 -100 到 100 的分数；WebShop 根据产品匹配度给分）
- **奖励分配**：GRPO 的 group-relative advantage 机制——同一任务 $N$ 条 trajectory 的 return 做归一化，所有时间步共享同一 advantage 值（均匀分配到 trajectory 内各步）。
- **辅助奖励/正则项**：
    - 内在探索奖励 $r_{\text{int}} = 1/n$（基于状态新颖度）
    - KL 正则项 $\beta D_{KL}(\pi_\theta \Vert \pi_{\text{ref}})$（ScienceWorld 中 $\beta = 0$，WebShop 中 $\beta = 0.01$）
    - 低概率 token masking $\mathbf{1}_{\pi_\theta(\cdot) \geq \delta}$（防止 off-policy 梯度爆炸）

---

## 4. Inference Procedure

- **记忆初始化**：测试时**不使用外部记忆**（评估的是训练后的 base model $\pi_\theta$，无 tips），即 $m_l = \emptyset$。
- **每步决策流程**（训练后推理）：
    1. 观测当前环境状态 $s_t$（文本描述）
    2. $\pi_\theta$ 直接根据 $(s_t, u)$ 生成动作 $a_t$（无检索，无 tips）
    3. 执行 $a_t$，获得 $r_t, s_{t+1}$
    4. 循环直到任务完成或达到最大步数
- **OOD 适应模式**（可选）：在新任务上可开启 memory，agent 在每次试验后生成 tips 存入 $M$，后续试验通过 Retr 检索 tips 注入 prompt，**无需参数更新**，仅通过非参数记忆适应。
- **推理策略**：完全由学习得到的 $\pi$ 驱动（base inference）；OOD 适应时结合手工规则（cosine similarity threshold > 0.5, top-10 retrieval）。

---

## 5. RQ 分析

### RQ1 (What is memory?)

EMPO 中的记忆是一个纯文本非参数 tips 缓冲区（$m_l$），每条 tip 是 LLM 在 episode 结束时对轨迹的自然语言反思总结（< 100 词），配合 embedding 索引进行 cosine similarity 检索。记忆仅在训练时使用，训练完成后通过 off-policy distillation 将知识内化到模型参数中。因此，EMPO 同时涉及显式记忆（训练时的 tips buffer，T1）和隐式记忆（通过 off-policy 蒸馏内化到 $\theta$，T5）。

### RQ2 (Which component is optimized? Which signal is used?)

优化对象是主 LLM 本身（$\theta$，即 Qwen2.5-7B-Instruct 的权重），通过 GRPO 变体进行 on-policy 和 off-policy 混合更新。信号来源是环境的 task reward（trajectory return）+ 内在探索奖励（state novelty-based）。Off-policy 模式本质是 reward-guided knowledge distillation，将 memory-conditioned 策略的优势行为蒸馏到 base 策略中。

### RQ3 (Target of Optimization)

最终优化目标是 G1 任务完成度（answer/action accuracy），即 agent 在 ScienceWorld 和 WebShop 中的 task score 和 success rate。我认为也同时隐式优化了探索效率，通过内在奖励和 memory 机制使 agent 更快发现高奖励状态空间区域，但效率本身不作为显式优化目标。

### RQ4 (How memory evolves, operates?)

训练时：每个 episode 结束后，$\pi_\theta$ 根据最终状态和反思 prompt 生成 tip，追加到全局 memory buffer $M$。后续 rollout 中以概率 $p$ 触发 memory-augmented 模式，通过 embedding cosine similarity 检索 top-10 tips 注入 prompt。$M$ 有 1000 条上限（FIFO 淘汰）。测试时：默认不使用 memory；OOD 适应时可开启 memory 进行 few-shot 适应（仅非参数更新，无权重修改）。

---

## Conclusion

EMPO 提出了一种将外部记忆与 RL 训练深度结合的框架，用于解决 LLM agent 在多步环境中的探索不足问题。其核心思想是：在训练阶段，让 agent 通过反思生成 tips 并存入外部记忆，利用 memory-augmented prompting 促进探索；同时通过 off-policy 更新将 memory-conditioned 策略中的优势行为蒸馏到 base 策略中，使模型在测试时无需外部记忆即可表现优异。低概率 token masking 和内在探索奖励进一步稳定了混合训练过程。