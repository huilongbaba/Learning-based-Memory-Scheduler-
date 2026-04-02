# Spurious Rewards: Rethinking Training Signals in RLVR

**Source:** [arXiv:2506.10947v2](https://arxiv.org/abs/2506.10947) [cs.AI], Preprint, February 26, 2026

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$x$|$q$|输入问题 / prompt|
|$y$|$T$|模型生成的 rollout（轨迹）|
|$y_t$|—|轨迹中第 $t$ 个 token（框架未细化到 token 级别）|
|$\pi_\theta$|$\pi$|当前策略（模型）|
|$\pi_\text{old}$|—|行为策略（上一步的策略），新增符号 $\pi_\text{old}$|
|$\pi_\text{ref}$|—|冻结的参考策略，新增符号 $\pi_\text{ref}$|
|$\theta$|$\theta$|模型参数（一致）|
|$r(x, y)$|—|奖励函数，框架未定义奖励符号，沿用 $r$|
|$\hat{A}(x, y)$|—|组内归一化优势函数，新增符号 $\hat{A}$|
|$\rho_t(y; \theta)$|—|token 级重要性比率 $\pi_\theta(y_t \mid x, y_{<t}) / \pi_\text{old}(y_t \mid x, y_{<t})$，新增|
|$\epsilon_c$|—|PPO 风格 clipping 阈值，新增|
|$G$|—|每个 prompt 的 rollout 数量（组大小）|
|$\gamma$|—|随机奖励的 Bernoulli 概率参数|

> 本文核心关注 RL 训练信号本身，而非 memory 系统。本文的核心贡献在于揭示 GRPO 的 clipping 机制如何放大预训练先验行为（可视为一种隐式的 $m_l$），而非显式的 memory 读写。

---

## 1. Problem Setting

**要解决的问题：** 本文研究 RLVR（reinforcement learning with verifiable rewards）中训练信号（奖励函数）的真正作用机制。具体而言，作者发现即使使用完全虚假的奖励信号（随机奖励、错误标签奖励），某些模型（尤其是 Qwen2.5-Math）仍能获得显著的数学推理性能提升，而其他模型家族（Llama3、OLMo2）则无法从中受益。

**形式化定义：**

- **输入：** 问题 $q$（数学推理题）
- **输出：** 最终答案 $A_n$（包含推理轨迹 $T$ 的完整生成序列）
- **策略：** $\pi(T \mid q)$，即给定问题 $q$ 时模型生成轨迹 $T$ 的概率

**任务类型与数据集：**

- 任务类型：数学推理（pass@1 / average@8 评估）
- 训练数据：DeepScaleR（Luo et al., 2025）
- 评估基准：MATH-500、AMC、AIME 2024/2025
- 模型：Qwen2.5-Math-7B/1.5B、Qwen2.5-7B/1.5B、Llama3.1-8B(-Instruct)、Llama3.2-3B(-Instruct)、OLMo2-7B(-SFT)

> 本文的独特价值在于系统性地挑战了"RLVR 的性能提升来自奖励信号本身"这一假设，通过消融实验揭示了优化算法（GRPO clipping）与预训练先验之间的交互作用。

---

## 2. Training Procedure

**训练流程：** 使用 GRPO（Group Relative Policy Optimization）进行在线 RL 训练。对每个问题 $q$，采样 $G=16$ 个 rollout ${T^{(1)}, \dots, T^{(G)}}$，计算组内归一化优势，然后通过 clipped surrogate objective 更新 $\theta$。

**核心公式（GRPO 目标函数）：**

$$J(\theta) = \mathbb{E}_{q \sim D,; T \sim \pi_\text{old}(\cdot|q)} \left[ \sum_{t=1}^{|T|} \min\left( \rho_t(T;\theta),\hat{A}(q,T),; \text{clip}\left(\rho_t(T;\theta),, 1-\epsilon_c,, 1+\epsilon_c\right) \hat{A}(q,T) \right) \right]$$

其中：

- $\rho_t(T;\theta) = \frac{\pi_\theta(y_t \mid q, y_{<t})}{\pi_\text{old}(y_t \mid q, y_{<t})}$（token 级重要性比率）
- $\hat{A}(q, T) = \frac{r(q, T) - \bar{r}_q}{\sigma_q}$（组内归一化优势函数）
- $\epsilon_c = 0.2$（clipping 阈值）
- KL 正则化系数 $\lambda = 0$（实验中禁用）

**Memory 如何参与训练：** 本文不涉及显式 memory 系统。但作者发现 GRPO 的 clipping 机制实际上充当了一种**隐式的预训练记忆放大器**：它系统性地增强已经具有高先验概率的 token（即预训练中学到的模式 $m_l$），同时抑制低概率 token。这种效应在没有信息性奖励的情况下依然存在。

**Clipping 偏差的梯度形式（§4, Eq.2）：**

$$\mathbb{E}[\nabla_\theta J(\theta)] \propto \mathbb{E}_{q, T} \begin{cases} \nabla_\theta \log \pi_{\theta,q}(y_t), & \text{if } R_\theta < 1-\epsilon_c \ 0, & \text{if } |R_\theta - 1| \leq \epsilon_c \ -\nabla_\theta \log \pi_{\theta,q}(y_t), & \text{if } R_\theta > 1+\epsilon_c \end{cases}$$

其中 $R_\theta = \pi_{\theta,q}(y_t) / \pi_{\text{old},q}(y_t)$。该公式表明：**即使奖励完全随机，clipping 也会产生非零的期望梯度——方向由预训练先验决定。**

**训练配置（§A.3）：** 8 GPU，学习率 5e-7，mini-batch size 128，rollout batch size 64，每 prompt 16 rollouts，温度 $\tau=1$，训练 300 步。

> RL 后训练主要放大预训练中已学到的行为，而非学习新能力。这篇文章表示，即使奖励为纯噪声，只要 clipping 机制存在，这种放大效应就会发生。

---

## 3. Reward Signal

**奖励信号的层级设计（§2.2）：**

|类别|奖励名称|定义|MATH-500 提升 (Qwen2.5-Math-7B)|
|---|---|---|---|
|标准|Ground Truth|$r=1$ 当且仅当答案正确|+29.1%|
|弱|Majority Vote|用 64 rollouts 的多数投票伪标签替代真实标签|+27.1%|
|弱|Format|$r=1$ 当且仅当回答包含非空 `\boxed{}`|+13.8%|
|虚假|Random|$r \sim \text{Bernoulli}(\gamma)$，独立于回答内容|+21.4%|
|虚假|Incorrect|用多数投票中**错误**的标签作为奖励标准|+24.1%|

**奖励与 Memory 的交互：** 本文的核心洞察是：奖励信号的信息量远不如预期重要。真正驱动性能提升的是 GRPO clipping 与预训练先验（隐式 $m_l$）的交互。具体表现为：

1. **对 Qwen2.5-Math 模型：** 预训练中学到的 "code reasoning"（在推理中使用 Python 代码但不执行）是一种高概率行为模式。clipping 偏差放大了这种模式（从 65% → 90%+ 频率），而 code reasoning 与正确性强相关（60.9% vs 28.0% 准确率），因此即使奖励无信息量，性能也会提升。
2. **对非 Qwen 模型：** 这些模型缺乏有效的 code reasoning 先验（No-Code 或 Bad-Code），因此 clipping 偏差无法放大有益行为，虚假奖励无效。

**关键验证实验（§4）：**

- 移除 clipping → 随机奖励失效（3 种不同的 no-clipping 变体均验证）
- 保留 clipping → 随机奖励持续有效
- 结论：clipping 是虚假奖励有效的**必要条件**

> 这一发现对 RLVR 研究社区有重要的方法论警示：许多 RLVR 方法仅在 Qwen 模型上验证，可能高估了其训练信号的真实贡献。作者建议使用虚假奖励作为 dummy baseline 来校准实验结果。

---

## 4. Inference Procedure

**推理设置（§A.4）：**

- pass@1：温度 0.0（贪心解码）
- pass@k / average@8：温度 0.6

**推理时无显式 memory 读写。** 推理时模型直接根据当前参数 $\theta$（已被 RLVR 更新）生成轨迹 $T$。RLVR 的效果体现在参数 $\theta$ 的变化中——具体来说，是高概率推理模式（如 code reasoning）被进一步强化。

**与训练阶段的差异：** 训练时采样多个 rollout 并计算组内优势；推理时为单次生成。训练的效果通过 $\theta$ 的更新永久编码到模型中。

**Code Reasoning 示例（§5, Figure 5）：** Qwen2.5-Math-7B 在推理时会生成 Python 代码并"预测"代码执行结果（实际上是自回归生成，未连接代码解释器），这种模式在 RLVR 后从 65% 增加到 90%+。

> 推理阶段的行为变化本质上是训练阶段 clipping 偏差累积效应的体现。这暗示了一种可能的替代方案：通过 prompt engineering 直接诱导 code reasoning（Table 2 显示 prompting 可带来 +15% 提升），无需 RL 训练。

---

## 5. RQ 分析

### RQ1: What is memory?

本文未涉及显式 memory。但可从隐式 memory 角度理解：预训练阶段学到的推理策略（如 code reasoning）构成了一种隐式长期记忆 $m_l$，编码在模型参数 $\theta$ 中。GRPO clipping 的作用是选择性地放大这些隐式记忆中的高概率模式。不同模型家族拥有不同的隐式 $m_l$ 内容（Qwen-Math 有 code reasoning，Llama/OLMo 没有），这决定了 RLVR 的效果。

### RQ2: How memory evolves, operates?

隐式 $m_l$ 的"演化"通过 GRPO 训练实现。Clipping 偏差使高概率行为（如 code reasoning）的概率进一步增加（Figure 9a 显示 token 概率单调上升），而低概率行为被抑制。这一过程是单调且不可逆的，在虚假奖励下，code reasoning 频率从 65% 快速增至 90%+（15 步内），且不会回落。值得注意的是，ground truth 奖励下 code reasoning 频率先升后降，暗示真实奖励能引导模型探索 code reasoning 之外的改进路径。

### RQ3: Which component is optimized? Which signal is used?

优化的组件是策略 $\pi$（即模型参数 $\theta$）。本文的核心发现是：优化信号的真正来源并非奖励函数 $r$，而是 GRPO clipping 与预训练先验 $\theta_0$ 的交互产生的梯度偏差。在随机奖励下，无 clipping 时期望梯度为零（无学习信号）；有 clipping 时，期望梯度非零且方向由 $\pi_\text{old}$ 的 token 概率分布决定。

### RQ4: Regarding online optimization

本文使用了在线 RL 训练（GRPO 是 on-policy 方法，每步用当前策略采样 rollout 并更新）。但作者发现，在虚假奖励设定下，"在线"的含义被弱化。因为奖励不依赖于生成内容，真正驱动更新的是 clipping 对先验的放大，而非对新行为的在线探索。

---

## Conclusion

这篇文章通过一组精心设计的实验，揭示了 RLVR 中一个反直觉的现象：对于 Qwen2.5-Math 模型，即使使用完全随机或故意错误的奖励信号，GRPO 训练仍能带来接近真实奖励水平的数学推理性能提升（随机奖励 +21.4% vs 真实奖励 +29.1%）。作者将这一现象追溯到 GRPO 的 clipping 机制，它产生了一种系统性的梯度偏差，会放大模型在预训练中已经学到的高概率行为模式（如 Qwen-Math 的 "code reasoning"）。关键的是，这种效应高度依赖模型——同样的虚假奖励对 Llama3 和 OLMo2 无效甚至有害，因为这些模型缺乏可被放大的有益预训练先验。

因此我认为，优化算法本身（而不仅仅是奖励信号）能够决定哪些内容会被“记住”和强化，截断偏差会系统性地偏向高先验行为，而与奖励质量无关。另外，仅在 Qwen 模型上验证的 RLVR 方法可能高估了其训练信号的真实贡献，未来研究应在多种模型上验证，并使用虚假奖励作为基线校准。