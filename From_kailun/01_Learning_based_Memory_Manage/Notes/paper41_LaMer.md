# LaMer: Meta-RL Induces Exploration in Language Agents

**Source:** [arXiv:2512.16848](https://arxiv.org/abs/2512.16848) (2026-03-08 修订，Accepted at ICLR 2026 poster，6666)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$\tau$|$T$|单个 episode 的 trajectory（state-action-reward 序列）|
|$\tau^{(n)}$|$T_n$|第 $n$ 个 episode 的 trajectory|
|论文中的 $\mathcal{T}$|—|trial = 由 $N$ 个 episode 组成的序列（与框架的 $T$ 同名但语义为 meta-trajectory，需注意区分）|
|$\mathcal{H}^{(n)}$|$m_l$|inter-episode memory，包含历史 trajectories 与 reflections|
|当前 episode 内的 $(s_t, a_t, r_t, \ldots)$|$m_s$|当前 episode 的 working memory|
|$\pi_\theta$|$\pi$, $\theta$|主 LLM 的 policy 与参数|
|reflection generation|$g_l$|由 $\pi_\theta$ 自身在 episode 末端生成自然语言 reflection 并 append 到 $\mathcal{H}^{(n)}$|
|全量拼接 $\mathcal{H}^{(n)}$ 到 prompt|$v$|退化的"检索"——无 TopK / 无相似度，全量注入 context|
|$\gamma_{\text{step}}$|—|episode 内 step-level 折扣因子|
|$\gamma_{\text{traj}}$|—|episode 间 cross-episode 折扣因子（本文新引入，控制 exploration-exploitation）|
|$G_t^{(n)}$|—|跨 episode 的 discounted return|

---

## 概览

这篇文章用了 LLM agent 与环境多次交互产生的 multi-episode trial，每个 trial 包含 $N$ 个 episode。每次 episode 失败后，agent 自己生成自然语言 reflection，并把过往 trajectories 和 reflections 作为 inter-episode memory 注入下一次 episode 的 context。  
最后得到了一个用 Meta-RL 训练的 LLM agent（基于 Qwen3-4B 与 Llama3.1-8B-Instruct 验证），它在新环境中能主动探索、并通过 reflection 在 in-context 层面快速适应。测试时无需梯度更新。  
优化跨 episode 的 discounted return，平衡探索与利用，固定 trial budget 下最大化任务完成率（G2）。通过环境交互结果判断成功，棋盘状态、揭示规则、商品属性匹配、任务完成度，属于 G5。

---

## 1. Problem Setting

- **Memory 类型**：cross-episode memory（trial 内多个 episode 之间持久化的记忆），对应框架 $m_l$；当前 episode 内的 step 序列对应 $m_s$。注意这不是跨用户/跨 chat 的长期记忆，而是"同一任务多次尝试间"的工作记忆。
- **决策建模**：标准 MDP 在 multi-episode 维度上的扩展。每个 episode 内部仍是 MDP；跨 episode 层面则相当于 POMDP——因为下一个 episode 的最优策略依赖于历史观测和反思。论文将其概括为 Meta-RL 框架。
- **状态空间** $\mathcal{S}$：环境提供的文本观测（Sokoban 棋盘、MineSweeper 已揭示的 cell、Webshop 商品页面、ALFWorld 房间描述）。
- **动作空间** $\mathcal{A}$：每个环境定义的离散动作集合（如 Sokoban 的 $\lbrace$up, down, left, right$\rbrace$）。
- **观测空间** $\Omega$：MineSweeper / Webshop / ALFWorld 为 partial observable，Sokoban 为 fully observable。
- **记忆数据结构**：纯自然语言文本拼接（episodic trajectory + reflection 的 list），无向量库、无图结构、无 KV 索引。

|框架组件|论文中的实现|
|---|---|
|$m_s$|当前 episode 内的 $(s_0, a_0, r_0, \ldots, s_{T-1}, a_{T-1}, r_{T-1})$|
|$m_l$|$\mathcal{H}^{(n)} = \lbrace \tau^{(0)}, \text{reflection}^{(0)}, \ldots, \tau^{(n-1)}, \text{reflection}^{(n-1)} \rbrace$|
|$g_s$|环境状态自然演化（不可学习）|
|$g_l$|由 $\pi_\theta$ 自身生成 reflection 并 append 到 $\mathcal{H}$|
|$v$|全量拼接（无显式检索；ablation 中可裁剪为 reflection-only / trajectory-only）|
|$\pi$|$\pi_\theta(\cdot \mid s_t, \mathcal{H}^{(n)})$|

> 读取是简单的全量拼接。

---

## 2. Training Procedure

- **优化对象**：主 LLM 的参数 $\theta$（**不冻结**）；reflection 生成步骤也作为 action 的一部分参与梯度更新。属于 RQ2 的 **O2a（单 LLM 联合优化）**。
- **优化算法**：默认 GiGPO（GRPO 家族在 LLM agent 上的变体）；论文也证实兼容 PPO、RLOO、GRPO。
- **训练数据**：在线交互采样。注意一个关键约束——同一 trial 内 $N$ 个 episode 必须 sequential rollout（因为每个 episode 依赖前面 episode 的 reflection），不能并行。
- **公平性 trick**：为了与 RL 公平比较，Meta-RL 用 group size 8、$N=3$ episode；标准 RL 用 group size 24，使两者每步 gradient update 用到的总 trajectory 数相同。

**核心训练目标（Eq. 5，论文原始 + 框架符号）：**

$$J(\theta) = \mathbb{E}_{\mathcal{T} \sim \pi_\theta} \Big\lbrack \sum_{n=0}^{N-1} \gamma_{\text{traj}}^n \sum_{t=0}^{T-1} \gamma_{\text{step}}^t , r_t^{(n)} \Big\rbrack = \mathbb{E}_{\mathcal{T} \sim \pi_\theta}\big\lbrack G_0^{(0)} \big\rbrack$$

**跨 episode 的 return（Eq. 4）：**

$$G_t^{(n)} = \underbrace{g_t^{(n)}}_{\text{within-episode}} + \underbrace{\sum_{m=n+1}^{N-1} \gamma_{\text{traj}}^{m-n} , g_0^{(m)}}_{\text{cross-episode}}$$

**Policy gradient 估计（Eq. 6）：**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\mathcal{T} \sim \pi_\theta} \Big\lbrack \sum_{n=0}^{N-1} \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta\big(a_t^{(n)} \mid s_t^{(n)}, \mathcal{H}^{(n)}\big) \cdot A_t^{(n)} \Big\rbrack$$

其中 $A_t^{(n)}$ 是从 $G_t^{(n)}$ 派生的 advantage（GiGPO 风格的 group-relative 归一化）。

---

## 3. Reward Signal

- **奖励类型**：sparse trajectory-level reward（每个 episode 结束时给出一次）
- **奖励来源**：环境反馈，二元终止信号——成功 → 10，失败 → 0
- **奖励分配**：通过 $\gamma_{\text{traj}}$ 跨 episode 折扣分配；在 episode 内通过 $\gamma_{\text{step}}$ 折扣到每个 token / step
- **是否有辅助奖励**：无显式辅助奖励项；但 reflection 生成 token 也会通过 next episode 的 reward 反向更新（隐式 credit assignment）

---

## 4. Inference Procedure

- **Memory 初始化**：第一个 episode 开始时 $\mathcal{H}^{(0)} = \emptyset$
- **每步 agent 决策流程**：
    1. 接收当前 state $s_t^{(n)}$ 和 inter-episode memory $\mathcal{H}^{(n)}$
    2. LLM 在 non-thinking mode 下生成 reasoning + `<action>...</action>`
    3. 环境执行 action，返回 reward 和 next state
    4. 当前 episode 结束（成功 / 达到 step 上限）后，prompt LLM 生成 `<remark>...</remark>` reflection
    5. 更新：$\mathcal{H}^{(n+1)} = \mathcal{H}^{(n)} \cup \lbrace \tau^{(n)}, \text{reflection}^{(n)} \rbrace$
    6. 如果当前 episode 成功，整个 trial 提前终止；否则进入下一 episode
- **推理超参**：temperature = 0.7, max_output_tokens = 1024, max episode budget $N = 3$
- **是否手工规则**：除"成功即终止 trial"这一规则外，其他完全由训练后的 $\pi_\theta$ 驱动，包括何时改变策略、如何反思

> Inter-episode memory 中只保留 LLM 自己写的反思总结（丢弃原始 trajectory）反而效果更好。

---

## 5. RQ 分析

### RQ1 (What is memory?)

这篇文章将 memory 建模为 inter-episode memory $\mathcal{H}^{(n)}$，即一个外部、显式、自然语言形式的 episodic trajectory + reflection 集合。属于 T1（非参数化外部记忆）。memory 写入由 LLM 自身的 reflection 生成完成，读取通过全量拼接到 context 实现，无显式检索机制。

### RQ2 (Which component is optimized? Which signal is used?)

优化主 LLM 的参数 $\theta$（O2a：单 LLM 优化）。同一个 LLM 做决策和写反思，reflection 的生成 token 通过下一个 episode 的 reward 接收 credit。训练信号是跨 episode 的 sparse outcome reward。

### RQ3 (Target of Optimization)

最终目标是 agent 任务成功率，对应 G5（Task Success via Execution），通过环境执行（Sokoban 棋盘状态、MineSweeper 揭示规则、Webshop 商品属性匹配、ALFWorld 任务完成）客观判断。具体衡量指标是 pass@K（K=1,2,3）。论文还隐性关注 G2（效率），在多次尝试预算下达到更高成功率，体现了更好的 test-time scaling。

### RQ4 (Training Signal and RL Algorithm)

默认使用 GiGPO（GRPO 家族针对 LLM agent 多轮交互场景的变体），属于 A2。论文也明确说明框架兼容 PPO、RLOO、GRPO 等多种 critic-free / actor-critic 算法。

### RQ5 (How memory evolves, operates?)

- 演化粒度：episode 级别（不是 step 级别）
- 写入机制：episode 末由 $\pi_\theta$ 自己生成 reflection，append 到 $\mathcal{H}$
- 读取机制：全量拼接（无 TopK、无 score 函数）
- 遗忘 / 压缩：可通过 buffer truncation 实现；ablation 显示 reflection-only 优于全量保留
- 是否端到端可学：是，reflection 的 token 作为 action 的一部分通过 RL 同步训练

---

## Conclusion

LAMER 的核心创新是把"多次尝试"建模为一个 trial（含 $N$ 个 episode），并通过两个机制让 agent 学会主动探索：(1) 跨 episode 的 discounted return（由 $\gamma_{\text{traj}}$ 控制 exploration-exploitation 平衡）；(2) self-reflection 驱动的 in-context policy adaptation（无梯度更新，纯靠 prompt 改写）。在 Sokoban、MineSweeper、Webshop、ALFWorld 四个 agentic 环境上，pass@3 比最强 RL baseline 高出 11–19%，并对 harder / OOD 任务展现出更强泛化。从 memory scheduling 视角看，本文贡献在于把"reflection-augmented inter-episode memory"作为可学习的 policy adaptation 机制，并通过 RL 同步训练 reflection 的生成质量。