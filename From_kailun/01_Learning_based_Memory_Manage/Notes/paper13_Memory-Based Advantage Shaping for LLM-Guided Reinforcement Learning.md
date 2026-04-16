# Memory-Based Advantage Shaping for LLM-Guided Reinforcement Learning

**Source:** [https://arxiv.org/abs/2602.17931](https://arxiv.org/abs/2602.17931) (February 2026, AAAI-26)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$\mathcal{G}$|$m_l$（图结构外部记忆）|Memory graph，存储 subgoal 与 trajectory 的图结构|
|$o_{\tau_j}$|$o$（observation）|Agent 在 trajectory node $j$ 处的部分观测|
|$a_{\tau_j}$|$a$（action）|Agent 在 trajectory node $j$ 处的动作|
|$\hat{r}_j$|—|存储在图节点中的估计回报（非框架核心符号）|
|$\zeta_j$|—|节点的目标标签（final goal $g_j$ 或 subgoal $\kappa_\ell^{g_j}$）|
|$\kappa_\ell$|—|LLM 推断的 subgoal 节点|
|$g_\triangleright$|—|Agent 的目标节点|
|$U_t$|—|从 memory graph 导出的 utility 信号（**提议新符号 $u_t$**）|
|$\tilde{A}_t$|—|Shaped advantage（**提议新符号 $\tilde{A}_t$**）|
|$\xi_t$|—|Utility shaping 的衰减系数|
|$\pi_\theta$|$\pi$（policy）|RL policy network（注意：**此处非 LLM**，而是传统 RL 策略网络）|
|$\theta$|$\theta$（model parameters）|RL policy network 的参数|
|PPO|—|Proximal Policy Optimization 优化算法|
|LLM|$M$（model，作为知识源）|提供 subgoal 分解和 trajectory 先验的冻结 LLM|

> 本文的 $\pi_\theta$ 是一个传统 RL 策略网络（如 MLP），**不是** LLM。LLM 在本文中仅作为离线知识源（提供 subgoal 和 trajectory 先验），其参数完全冻结，不参与优化。
---

## 概览

这篇文章用了 LLM 提供的 subgoal 分解和 trajectory 建议，以及 agent 自身成功 rollout 的经验 trajectory。这些信息被编码进一个 memory graph $\mathcal{G}$。  
最后得到了一个 RL policy network $\pi_\theta$，该 policy 能够在稀疏奖励环境中更快地学习有效策略。Memory graph 本身也是一个持续演化的副产品。  
优化了G1，即RL agent 在环境中的累积回报（mean return）和任务成功率（success rate），同时优化了G2，即 sample efficiency，并减少了对 LLM 持续查询的依赖。

---

## 1. Problem Setting

- **记忆类型**：cross-episode memory（跨 episode 积累的经验和 LLM 先验），对应 $m_l$（长期记忆）。不存在 in-chat / cross-chat 的语义——本文处理的是传统 RL 环境（FrozenLake、DoorKey），非对话场景。
- **决策过程建模**：标准 MDP / POMDP（DoorKey 环境为部分可观测）。记忆管理本身**不**被建模为决策过程——图的更新由规则驱动（新 rollout 成功时加入节点、长期未访问节点被剪枝）。
- **状态空间 $\mathcal{S}$**：环境状态（如 FrozenLake 的网格位置、DoorKey 的 agent 位置 + 持有物品）
- **动作空间 $\mathcal{A}$**：环境动作（如上下左右移动、拾取钥匙、开门）
- **观测空间 $\Omega$**：部分观测（DoorKey 中 agent 仅观测周围有限视野）
- **记忆数据结构**：**有向图（memory graph）**，包含三类异构节点：
    - Trajectory nodes：$(o_{\tau_j}, a_{\tau_j}, \zeta_j, \hat{r}_j)$
    - Subgoal nodes：$\lbrace \kappa_\ell \rbrace_{\ell=1}^{L}$
    - Goal nodes：$\lbrace g_\triangleright \rbrace$
    - 边表示 goal–subgoal 关系

**核心组件与符号映射**：

|核心组件|论文实现|框架符号|
|---|---|---|
|长期记忆|Memory graph $\mathcal{G}$|$m_l$|
|短期记忆|当前 episode 的 rollout trajectory|$m_s$|
|检索算法|Similarity matching $\int(\cdot, \cdot)$ + Jaccard $\rho(\cdot, \cdot)$|$v$|
|记忆更新函数|基于规则的节点增删（成功 rollout → 加入、长期未用 → 剪枝）|$g_l$|
|策略|RL policy network $\pi_\theta$（非 LLM）|$\pi$|
|知识源模型|冻结的 LLM（GPT-4o 等）|$M$|

> 本文的记忆设定与 survey 中其他论文不同，这里的 memory graph 服务于传统 RL agent 在 grid-world 环境中的学习，而非 LLM 的 context 管理或跨对话记忆。记忆管理由规则驱动。
---

## 2. Training Procedure

- **优化组件**：RL policy network $\pi_\theta$（传统 actor-critic 架构）。**LLM 参数完全冻结**。
- **优化算法**：PPO（Proximal Policy Optimization），加上 utility-shaped advantage。
- **训练数据来源**：在线交互（agent 与环境的 rollout），加上离线 LLM 先验（subgoal 分解、partial trajectory）和偶尔的在线 LLM 查询。
- **LLM 参数是否冻结**：是。被优化的参数是 RL policy network 的权重 $\theta$（非 LLM 权重）。
- **核心训练目标函数**：

**论文原始公式（Shaped PPO）**：

$$\tilde{A}_t = A_t + \xi_t U_t$$

$$\mathcal{L}^{\text{shaped}}(\pi_\theta) = \mathbb{E}\left[\min\left(r_t(\theta), \text{clip}(r_t(\theta), 1 \pm \varepsilon_k)\right) \tilde{A}_t\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_k}(a_t \mid s_t)}$

**Utility 计算**：

$$U_t \doteq \hat{r}_m \cdot \rho(g_\triangleright, \zeta_m) \cdot \int\left((o_t, a_t), (o_{t'}, a_{t'})_{\tau_m}\right)$$

**框架符号标注**：

$$\tilde{A}_t = A_t + \xi_t \cdot u_t(m_l, o_t, a_t)$$

其中 $u_t$ 从 $m_l$（memory graph）中检索与当前 $(o_t, a_t)$ 最匹配的节点，计算行为相似度和目标对齐度。

---

## 3. Reward Signal

- **奖励类型**：Sparse terminal reward（来自环境本身），+ dense step-level utility signal $U_t$（来自 memory graph，但注意 $U_t$ 不是 reward，而是 advantage 的附加项）。
- **奖励来源**：
    - 环境奖励：任务完成时的稀疏回报（如到达目标位置）
    - Utility 信号：从 memory graph 中导出，衡量 agent 当前行为与存储的成功策略的对齐程度
- **奖励分配机制**：
    - 环境奖励：terminal（仅在 episode 结束时）
    - Utility $U_t$：per-step，通过 similarity matching 和 Jaccard 相似度直接为每个 $(o_t, a_t)$ 计算
- **辅助奖励或正则项**：
    - $U_t$ 本身可视为辅助信号（非 reward 而是 advantage shaping term）
    - 衰减系数 $\xi_t \in (0, 1]$ 随训练进行递减，确保 LLM prior 的影响逐渐消退
    - 无额外正则项（如 KL penalty）

---

## 4. Inference Procedure

- **记忆初始化**：Memory graph 在训练前用离线 LLM 先验初始化（subgoal 分解、partial trajectory 建议），训练过程中不断演化。推理时使用训练结束后的最终 graph 状态。
- **每步决策流程**：
    1. Agent 观测当前环境状态 $o_t$（部分观测）
    2. RL policy network $\pi_\theta$ 直接输出动作 $a_t = \pi_\theta(\cdot \mid o_t)$
    3. **推理时不使用 memory graph**——graph 仅在训练时通过 advantage shaping 影响 policy 更新，训练完成后 policy 已内化了 graph 中的知识
- **推理时额外策略**：无。推理时是纯粹的 learned policy $\pi_\theta$ 执行，不涉及检索、replan 或 LLM 查询。
- **推理策略驱动方式**：完全由学习得到的 $\pi_\theta$ 驱动，无手工规则。Memory graph 的知识在训练阶段已通过 advantage shaping "蒸馏"进 policy 参数。

> 这是本文与大多数 memory-augmented LLM 工作的一个关键区别，memory 仅在训练时使用，推理时 agent 不再访问 memory graph。作者认为记忆的作用是加速训练而非增强推理。

---

## 5. RQ 分析

### RQ1 (What is memory?)

本文的 memory 是一个图结构的外部经验存储（memory graph $\mathcal{G}$），包含 LLM 提供的 subgoal 先验和 agent 自身成功 rollout 的 trajectory 片段。每个节点编码了 observation-action 对、目标标签和估计回报。属于T2。与典型 LLM memory 不同，这里的 memory 服务于传统 RL agent 在 grid-world 环境中的策略学习。

### RQ2 (Which component is optimized? Which signal is used?)

被优化的是独立的 RL policy network $\pi_\theta$（O1b：记忆辅助的任务模型）。LLM 完全冻结，仅作为知识源。优化信号包括环境稀疏奖励（通过 PPO 的 advantage estimation）和 memory graph 导出的 utility signal $U_t$（通过 advantage shaping 融入 policy gradient）。

### RQ3 (Target of Optimization)

优化目标是任务累积回报（mean return）和任务成功率（success rate），对应 G1（回答/任务准确度）。同时，论文强调 sample efficiency（更少的环境交互次数达到相同性能），对应 G2（效率）。

### RQ4 (How memory evolves, operates?)

Memory graph 的演化由规则驱动：

- **初始化**：离线 LLM 先验（subgoal 分解、partial trajectory）构建初始图
- **写入**：agent 产生成功 rollout 时，新 trajectory 节点被加入图中；如果 rollout 对已知 subgoal 有更高回报，则更新对应节点
- **在线更新**：当 utility 信号连续多个 episode 接近零时（图中无有用指导），触发在线 LLM 查询，新建议经筛选后加入图
- **剪枝**：长期未被访问的节点被移除以保持图的紧凑
- **推理时**：memory graph 不再被使用，知识已通过训练蒸馏进 $\pi_\theta$

---

## Conclusion

本文提出了一种将 LLM 先验知识融入 RL 训练的方法：构建一个 memory graph 来存储 LLM 提供的 subgoal 分解和 agent 自身的成功经验，然后从图中导出 utility 信号来 shape PPO 的 advantage function。这种方法的优势在于：（1）不修改环境奖励，保留 PPO 收敛性；（2）仅需少量离线/偶尔在线 LLM 查询，避免对 LLM 的持续依赖；（3）utility shaping 的影响随训练进行自然衰减，最终 policy 的行为完全由环境反馈驱动。  
这篇文章与其他工作的区别在于：(1) 被优化的是传统 RL policy 而非 LLM；(2) memory 仅在训练时使用，推理时不访问；(3) 环境是 grid-world 而非 QA/对话任务。我认为作者想表示 LLM 的角色是知识提供者而非记忆使用者。与 MIRA 是同一作者，姊妹工作。MIRA 提供了更完整的理论分析和实验。