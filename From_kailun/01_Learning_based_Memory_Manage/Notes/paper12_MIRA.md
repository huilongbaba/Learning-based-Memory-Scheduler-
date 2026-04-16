# MIRA: Memory-Integrated Reinforcement Learning Agent with Limited LLM Guidance

**Source:** [https://arxiv.org/abs/2602.17930](https://arxiv.org/abs/2602.17930) (February 2026, ICLR 2026)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|Memory graph $\mathcal{G}$|$m_l$（长期记忆）|存储 trajectory segments 和 subgoal 分解的图结构记忆|
|Node（trajectory segment + subgoal）|$m_l$ 中的条目|图节点，包含轨迹片段、子目标描述、置信度 $c_m$、预估回报 $\hat{r}_m$|
|Utility signal $U_t$|无直接对应，提议符号 $u_t$|从 memory graph 派生的、用于 shaping advantage 的信号|
|Shaping weight $\xi_t$|无直接对应，提议符号 $\xi_t$|控制 utility 对 advantage 的影响强度，随训练衰减|
|Screening Unit|无直接对应，提议符号 $f_{\text{screen}}$|对 LLM 在线输出进行置信度过滤的模块|
|Similarity $\mathcal{s}(\cdot, \cdot)$|类似 $v$（检索相似度）|衡量 agent 行为与存储轨迹的一致性|
|Goal-alignment $\rho(\cdot, \cdot)$|无直接对应|子目标语义匹配的 Jaccard 相似度权重|
|$\pi_\theta$|$\pi$（策略）|RL agent 的策略网络|
|PPO advantage $\hat{A}_t$|标准 advantage|GAE 估计的 advantage|
|Shaped advantage $\tilde{A}_t$|无直接对应，提议 $\tilde{A}_t$|加入 utility shaping 后的 advantage|
|Environment reward $R$|reward|环境稀疏奖励（如 +1 达到目标）|

---

## 概览

这篇文章用了 LLM 离线生成的 subgoal 分解和 trajectory segments，以及 agent 自身高回报经验的 trajectory 片段，将它们共同构建成一个图结构记忆。  
最后得到了一个更好的 RL policy $\pi_\theta$。MIRA 并没有训练出一个独立的可插拔记忆管理模型，而是通过 memory graph 派生的 utility 信号来 shaping PPO 的 advantage estimation，最终产出一个在稀疏奖励环境中 sample efficiency 更高的 RL 策略。  
优化了 G1 和 G2，即 RL agent 的 cumulative environment reward（任务成功率和回报），通过 utility-shaped advantage 加速早期学习，同时保持 PPO 的长期收敛性。核心指标是 mean return 和 success rate。

---

## 1. Problem Setting

- **记忆类型**：cross-episode memory（跨 episode 的持久记忆），属于 $m_l$（长期记忆）。Memory graph 在 episode 之间持续存在并演化，存储的是跨 episode 积累的 trajectory segments 和 subgoal 结构。不涉及 in-chat memory 或 LLM context window 内的 $m_s$。
    
- **决策过程建模**：标准 MDP / POMDP。Agent 在 MiniGrid/BabyAI 等环境中以 partial observation 做决策。Memory graph 不被建模为一个独立的决策过程，而是作为 advantage estimation 的辅助信号。
    
- **状态空间 $\mathcal{S}$**：环境状态（grid world 中的 agent 位置、物体位置、门/钥匙状态等）。
    
- **动作空间 $\mathcal{A}$**：离散动作（移动、转向、拾取、开门等 MiniGrid 原生动作）。
    
- **观测空间 $\Omega$**：部分可观测的局部视野（agent 前方有限范围的 grid 视图）。
    
- **记忆数据结构**：有向图（memory graph $\mathcal{G}$）。节点存储 trajectory segment（状态-动作序列片段）和对应的 subgoal 描述、置信度 $c_m$、预估回报 $\hat{r}_m$。边编码 goal-subgoal 的层级分解关系。
    

|核心组件|框架符号|MIRA 中的具体实现|
|---|---|---|
|长期记忆|$m_l$|Memory graph $\mathcal{G}$，节点 = (trajectory segment, subgoal, $c_m$, $\hat{r}_m$)|
|短期记忆|$m_s$|无显式 $m_s$，当前 episode 的 rollout 即为工作记忆|
|检索算法|$v$|基于 similarity $\mathcal{s}$ 和 goal-alignment $\rho$ 的匹配|
|策略|$\pi$|PPO 策略网络 $\pi_\theta$|
|模型|$M$|RL agent（CNN/MLP policy + value network）|
|轨迹|$T$|Agent rollout trajectory $(o_1, a_1, o_2, a_2, \ldots)$|

> MIRA 的 memory graph 是一个带层级结构的图，节点存储的是 trajectory segment。最接近 T2，本质是 episodic trajectory 的图组织形式。

---

## 2. Training Procedure

- **优化的组件**：$\pi_\theta$（RL agent 的策略网络参数 $\theta$），同时包含 value network（critic）。Memory graph 本身不通过梯度优化，而是通过规则驱动的 CRUD 演化（添加高回报 trajectory、剪枝低使用节点）。LLM 参数完全冻结，仅作为 offline/infrequent-online 的知识源。
    
- **优化算法**：PPO（Proximal Policy Optimization），具体为 utility-shaped advantage estimation 版本的 PPO。
    
- **训练数据来源**：在线交互（agent 与 MiniGrid/BabyAI/Gymnasium 环境的实时交互 rollout）+ LLM 离线生成的 priors（subgoal 分解和 trajectory 建议）+ 少量在线 LLM 查询。
    
- **是否冻结 LLM 参数**：是。LLM 仅提供离线 prior 和偶尔的在线建议，其参数不参与优化。被优化的参数是 RL agent 的 policy network 和 value network 的权重 $\theta$。
    
- **核心训练目标函数**：
    

标准 PPO clipped surrogate objective，但 advantage 被 utility signal 增强：

论文原始公式（shaped advantage）：

$$\tilde{A}_t = \eta_t \cdot \hat{A}_t^{\text{GAE}} + \xi_t \cdot U_t$$

其中：

- $\hat{A}_t^{\text{GAE}}$ 是标准 GAE advantage
- $U_t$ 是 memory graph 派生的 utility signal
- $\xi_t$ 是衰减的 shaping 权重（训练早期大，后期趋零）
- $\eta_t$ 逐步升至 1

Utility 的计算：

$$U_t = \hat{r}_m \cdot \rho(g_{\triangleright}, \zeta_m) \cdot \mathcal{s}\big((o_t, a_t), (o_{t'}, a_{t'})_{\tau_m}\big)$$

框架符号标注：$U_t$ 即 $u_t$；$\hat{r}_m$ 是记忆节点的预估回报；$\rho$ 是目标对齐权重；$\mathcal{s}$ 是行为相似度。

PPO 的 clipped objective 保持不变，只是将 $\hat{A}_t$ 替换为 $\tilde{A}_t$。

> MIRA 的核心创新在于不修改 reward function，而是通过 utility signal shaping advantage estimation。MIRA memory graph 的构建和更新规则是手工设计的，非端到端可学习。

---

## 3. Reward Signal

- **奖励类型**：Sparse terminal reward（环境原生奖励，如到达目标 +1）。Utility signal $U_t$ 是 dense step-level 的，但它不是 reward，而是 advantage shaping 信号。
    
- **奖励来源**：环境反馈（MiniGrid/BabyAI/Gymnasium 的 task completion reward）。Utility signal 来源于 memory graph 中存储的先验 trajectory 与当前 rollout 的匹配度。
    
- **奖励如何分配到各步骤**：环境奖励仅在 episode 结束时给出（sparse）。Utility $U_t$ 在每一步根据当前 $(o_t, a_t)$ 与 memory graph 中 trajectory segment 的匹配度计算，提供 dense 的 per-step shaping 信号。分配机制为：匹配到存储 trajectory 的步骤获得非零 utility，未匹配步骤获得零 utility。
    
- **辅助奖励或正则项**：
    
    - $\xi_t$ 的衰减机制作为隐式正则，确保 LLM-derived priors 的影响在训练后期消退
    - Online LLM 输出的 soft logit injection（对 discouraged actions 添加有界惩罚）作为额外的轻量引导
    - Screening Unit 通过置信度过滤（token-level likelihood 或多次采样一致性）对 LLM 输出进行质量控制

> 将 LLM guidance 转化为 advantage shaping 信号而非 reward shaping，避免了修改 reward function 带来的理论问题。

---

## 4. Inference Procedure

- **推理时记忆如何初始化**：Memory graph 通过 LLM 离线查询初始化，提供 subgoal 分解和 trajectory segments 作为结构化 prior。LLM 在离线阶段拥有完整任务描述的访问权限。
    
- **每步 agent 的决策流程**：
    
    1. Agent 接收部分观测 $o_t$（环境局部视野）
    2. 当前 $(o_t, a_t)$ 对与 memory graph 中的 trajectory segments 进行匹配，计算 utility $U_t$
    3. $U_t$ 被整合到 shaped advantage $\tilde{A}_t$ 中用于策略更新（训练时）
    4. 策略网络 $\pi_\theta$ 基于 $o_t$ 输出动作分布，采样动作 $a_t$
    5. 若连续多个 episode 的 rollout utility 接近零，触发在线 LLM 查询，输出经 Screening Unit 过滤后注入 memory graph
- **推理时额外策略**：
    
    - Memory graph 的节点剪枝：未被使用超过固定 episode 窗口的节点被移除
    - 节点更新：当 agent 产生比已有节点回报更高的 trajectory segment 时，更新对应节点
    - Online LLM 的 adaptive triggering：仅在 utility 持续为零时触发，避免不必要的 LLM 调用
    - Soft logit injection：在线 LLM 的 control signal 通过对 discouraged actions 的 logit 添加有界惩罚来轻微偏置策略
- **推理策略驱动方式**：主要由学习得到的 $\pi_\theta$ 驱动，memory graph 的匹配和更新规则是手工设计的。推理时（训练完成后），$\xi_t$ 已衰减至零，policy 完全由学到的 $\pi_\theta$ 决定，memory graph 不再影响决策。
    

> MIRA 的一个重要特性是训练完成后 memory graph 的影响完全消退，最终推理纯粹依赖学到的 policy。这与需要在推理时持续访问记忆/检索的方法形成鲜明对比。我认为Memory graph 的角色更接近训练时的脚手架而不是推理时的知识库。

---

## 5. RQ 分析

### RQ1 (What is memory?)

MIRA 中的 memory 是一个有向 memory graph $\mathcal{G}$，节点存储 trajectory segments 和 subgoal 描述，边编码 goal-subgoal 的层级关系。这是一种显式的、可查看的、跨 episode 持久化的长期记忆 $m_l$，其内容来源于 LLM 离线生成的 prior 和 agent 高回报经验的积累。记忆以图结构组织，兼具 episodic trajectory 存储和 hierarchical goal 分解功能。

### RQ2 (Which component is optimized? Which signal is used?)

优化的是 RL agent 的策略网络参数 $\theta$（policy + value network），属于O1b：记忆辅助的任务模型，使用 PPO 算法。LLM 参数冻结，memory graph 通过规则驱动的 CRUD 演化而非梯度优化。训练信号包括：(1) 环境 sparse terminal reward（task completion），(2) memory graph 派生的 utility signal $U_t$（匹配度）用于 advantage shaping。Utility signal 在训练早期提供 dense guidance，随训练衰减。

### RQ3 (Target of Optimization)

最终优化目标属于 G1（回答/任务准确度）和 G2（效率），是 RL agent 在环境中的 cumulative reward（mean return 和 success rate），即任务完成的准确度和效率。同时，MIRA 追求 sample efficiency，即用尽可能少的 LLM 查询和环境交互达到高回报。

### RQ4 (How memory evolves, operates?)

Memory graph 在训练过程中持续演化：(1) 初始化时由 LLM 离线生成 subgoal 分解和 trajectory priors；(2) 训练中，agent 的高回报 trajectory segments 被添加或更新到图中（若新片段比已有节点回报更高）；(3) 低使用节点被定期剪枝；(4) 在线 LLM 查询在 utility 持续为零时自适应触发，输出经 Screening Unit 过滤后注入图中。训练完成后，$\xi_t \to 0$，memory graph 不再影响策略，记忆的作用完全内化到了学到的 $\theta$ 中。

---

## Conclusion

MIRA 提出了一种将 LLM 知识整合到 RL 训练中的方法：通过构建一个 memory graph 来存储 LLM 生成的 subgoal 分解和 trajectory segments，以及 agent 自身的高回报经验。从这个图中派生出一个 utility signal，用于 shaping PPO 的 advantage estimation，在训练早期提供 dense guidance 以克服稀疏奖励下的低效探索问题。随着训练推进，utility 的影响衰减至零，保证了 PPO 的收敛性质。该方法的核心贡献在于"将 LLM 查询摊销到持久记忆中"，避免了对 LLM 的持续实时依赖。