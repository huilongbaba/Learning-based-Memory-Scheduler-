# Freshness-Aware Prioritized Experience Replay for LLM/VLM Reinforcement Learning (FreshPER)

**Source:** [https://arxiv.org/abs/2604.16918](https://arxiv.org/abs/2604.16918)（2026-04-18 修订）

## 符号映射表

| 论文原始符号 | 框架统一符号 | 含义说明 |
|---|---|---|
| $\pi_\theta$ | $\pi$ | 当前训练策略（主 LLM） |
| $\pi_\mu$ / $\pi_{\text{old}}$ | $\pi_{\text{behavior}}$ | 采集轨迹时的行为策略（与 $\pi$ 同模型族但参数滞后） |
| 轨迹 $e_j = (s, a, o, r, \log\pi_\mu)$ | $T$ | 完整 episode（多轮 prompt + response + observation + reward） |
| Replay Buffer $\mathcal{B}$ | $m_\ell^{\text{train}}$ | 训练时的轨迹存储（**注**：本文 memory 仅在训练阶段存在，不参与推理；属于 L-train 生命周期） |
| $p_i^{\text{base}}$ | — | 轨迹基础优先级（$\lvert r_i\rvert$、$\lvert \hat A_i\rvert$ 或 $\lvert\delta_i\rvert$） |
| $\Delta_i = t - t_i$ | — | 轨迹年龄（梯度步数差） |
| $\tau$ | — | 老化衰减常数（half-life $= \tau\ln 2$） |
| 优先级采样 $P(i)\propto p_i^\alpha$ | $v$ | 检索/采样算法（priority-proportional, stratified） |
| 优先级刷新（Line 6） | $g_\ell$ | 长期记忆更新函数（重算 age decay、evict 最低优先级） |

---

## 概览

这篇文章用了训练过程中的"过去 rollout 轨迹"。每条轨迹连同它当时的奖励、行为策略 log-prob、采集时刻一起被存进一个 replay buffer 里反复使用。  
最后得到了一个更高效训练好的主 LLM/VLM 策略 $\pi_\theta$（训练完成后 buffer 即可丢弃，推理阶段不再需要它）。  
在多轮 agentic 任务（搜索、Sokoban、FrozenLake 等）和数学推理任务上，提升样本效率与最终任务准确率。通过让昂贵的环境交互轨迹被多次复用，且让新鲜且信息量大的轨迹被采样得更频繁。

---

## 1. Problem Setting

- **memory 类型**：训练时轨迹回放缓冲区（training-time trajectory replay buffer）。这与典型 LLM agent memory（in-chat / cross-chat）不同——它**只在训练阶段存在**，作用对象是 $\theta$，而不是 inference 时的 $C_N$。本文把 $m_\ell^{\text{train}}$ 视作可显式 CRUD 的离散条目集合。
- **决策过程建模**：多轮 MDP $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$，其中
  - $\mathcal{S}$：完整对话历史 $s_t = q \oplus a_1 \oplus o_1 \oplus \ldots \oplus a_{t-1} \oplus o_{t-1}$
  - $\mathcal{A}$：当前 turn 的 assistant response（token 序列）
  - $\Omega$：环境返回的 observation $o_t$
  - $T$：确定性转移 $s_{t+1} = s_t \oplus a_t \oplus o_t$
  - $\gamma = 1$（episodic undiscounted）
- **记忆数据结构**：每条 buffer entry $= (\text{prompt}, \text{turns}, o, \log\pi_\mu, r, t_i, p_i)$。trajectory-level 而非 transition-level，匹配 agentic RL 的 episode-level reward 信号。
- **核心组件映射**：

| 组件 | 论文符号 | 框架符号 | 实现 |
|---|---|---|---|
| 策略 | $\pi_\theta$ | $\pi$ | DeepSpeed 训练侧模型 |
| 行为策略 | $\pi_\mu$ | — | vLLM 推理侧模型，定期同步 |
| 长期记忆 | $\mathcal{B}$ | $m_\ell^{\text{train}}$ | 50K 容量的轨迹缓冲区 |
| 检索 | priority sampling | $v$ | sum-segment-tree + stratified sampling |
| 写入/老化更新 | refresh + FIFO eviction | $g_\ell$ | 异步 CPU 线程，O(N) 扫描 |

---

## 2. Training Procedure

- **优化对象**：主 LLM 参数 $\theta$（对应 RQ2 中的 O2a 单 LLM 优化）。优先级函数本身**不学习**，是手工设计的 $p_i = p_i^{\text{base}} \cdot \exp(-\Delta_i/\tau)$。
- **优化算法**：PPO++（critic-free policy gradient + global advantage normalization + clipping）。
- **训练数据来源**：在线交互（fresh rollout）+ 回放历史轨迹（off-policy replay），二者混合。
- **LLM 是否冻结**：否，主 LLM 全参数训练。
- **核心训练目标函数**：

  on-policy 部分（PPO 标准目标）：
  
  $$\mathcal{L}_{\text{on}}(\theta) = -\mathbb{E}_{\pi_\theta}\Big[\sum_t \log \pi_\theta(a_t \mid s_t)\, \hat A_t\Big]$$
  
  off-policy replay 部分（带 IS 修正）：
  
  $$\mathcal{L}_{\text{replay}}(\theta) = \frac{1}{B}\sum_{i=1}^B w_i \cdot \ell_i$$
  
  其中 IS 权重
  
  $$w_i = \Big(\frac{1}{N \cdot P(i)}\Big)^\beta \Big/ \max_j w_j$$
  
  优先级
  
  $$p_i = \underbrace{p_i^{\text{base}}}_{\lvert r_i\rvert + \epsilon \text{ 等}} \cdot \underbrace{\exp\!\Big(-\frac{\Delta_i}{\tau}\Big)}_{\text{age decay}}$$
  
  采样分布
  
  $$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

---

## 3. Reward Signal

- **奖励类型**：稀疏 terminal reward（episode 结束时由环境/grader 给出）。
- **奖励来源**：
  - NQ Search → Exact Match grader
  - AIME / GSM8K → 数值匹配 grader
  - Sokoban / FrozenLake / CliffWalking → 环境内置 reward function
  - GeoQA → 答案匹配
- **奖励分配**：episode-level reward 通过 PPO 的 global advantage normalization 分摊到所有 token。
- **辅助奖励/正则项**：无显式 shaping reward；但在工程上发现需要 **关闭 KL 正则**（$\beta_{KL}=0$）和 **关闭熵正则**，并把 advantage clip 收紧到 $\epsilon_{\text{clip}}=0.2$，否则 off-policy replay 会发散。
- **优先级 $\ne$ 奖励**：优先级 $p_i$ 仅用于决定该轨迹被采样的概率，不直接进入 loss；进入 loss 的是 IS 权重 $w_i$ 与策略梯度。

> Reward signal 不是为 memory 系统专门设计的。

---

## 4. Inference Procedure

- **推理时记忆初始化**：**无**。FreshPER 的 buffer 仅在训练时存在，推理时不使用任何 replay 机制。
- **每步 agent 决策**：标准 LLM agent loop——观测 $\to$ 调用工具（检索/环境）$\to$ 生成 action。**没有**记忆读写步骤。
- **额外推理策略**：top-p=0.99, top-k=100, temperature=0.99；多轮 episode 上限 $H$ 由任务决定（NQ Search=5, Sokoban=10, AIME=3 等）。
- **是否手工规则**：推理由学到的 $\pi_\theta$ 完全驱动，replay 机制对推理透明。

> 贡献在于让训练阶段更有效地从过去 rollout 中学习。

---

## 5. RQ 分析

### RQ1 (What is memory?)

本文的 memory 是 训练时轨迹回放缓冲区，即一个存放完整 episode（含 prompt、turns、reward、行为 log-prob、采集时刻）的离散外部存储，结构上与 T1（非参数化外部记忆）一致，作用对象是 $\theta$ 而非 $C_N$。

### RQ2 (Which component is optimized?)

被优化对象是主 LLM 本身（O2a 单 LLM 优化），通过 PPO 在 fresh rollout + replay batch 上做混合策略梯度。memory 端的优先级函数 $p_i = p_i^{\text{base}}\cdot \exp(-\Delta_i/\tau)$ 不是学习得到的，而是手工设计的启发式（这一点与 Memory-R1 等显式训练 memory manager 的工作形成对比）。

### RQ3 (Target of Optimization)

核心目标是 G1（任务准确率）+ G2（样本效率）。其根本动机是：昂贵的 rollout 轨迹只用一次太浪费。

### RQ4 (Training Signal and RL Algorithm)

本文使用 REINFORCE++，critic-free policy gradient + 全局 advantage normalization + clipping。off-policy 修正使用 IS 权重而非 KL 约束。

### RQ5 (How memory evolves, operates?)

训练时记忆动力学清晰：
- 写入 $g_\ell$：每次 fresh rollout 后将轨迹 + 行为 log-prob + reward + 时间戳压入 buffer，初始化基础优先级。
- 更新 $g_\ell$：异步 CPU 线程每 iteration 重算所有条目的 age decay $\exp(-\Delta_i/\tau)$；reward-based base priority 在采集时固定，而 advantage/TD-based 变体则在每次更新后重算。
- 检索 $v$：sum-segment-tree 实现 $O(\log N)$ priority sampling，配合 stratified sampling 降低方差。
- 遗忘：buffer 满时 evict 最低有效优先级条目（自然对应"最老 + 最低基础优先级"）。
- 推理时：完全不使用 buffer。

---

## Conclusion

FreshPER 把经典 RL 中的 Prioritized Experience Replay 引入到 LLM/VLM 强化学习中，并指出直接照搬会失败的核心原因，即优先级陈旧（priority staleness）：billion-parameter 策略一步梯度就能让分布大幅漂移，旧轨迹的 importance ratio 早已失效，但其优先级却保持不变，从而长期主导采样。  
作者的对策是给任意 PER 基础优先级乘上一个指数老化因子 $\exp(-\Delta_i/\tau)$，并用 effective sample size 的指数下界 $\text{ESS} \le n\cdot \exp(-D_{\text{KL}})$ 给出理论动机。在 8 个任务（搜索、数学、Sokoban、FrozenLake、GeoQA 等）和三种模型规模（0.5B/3B/7B）上一致优于 on-policy 基线和不带老化的标准 PER。它是首个把 trajectory-level PER 在 LLM RL 上做通的工作，但其 memory 本质是训练时，不直接处理 agent inference 阶段的记忆问题。