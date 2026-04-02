# The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning

**Source:** [arXiv:2505.15134](https://arxiv.org/abs/2505.15134) (21 May 2025)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$x$|$q$|输入 prompt / human input|
|$y = y_1 \dots y_{\|y\|}$|$T$|输出轨迹（trajectory）|
|$y_t$|$a$|第 $t$ 步生成的 token（action）|
|$y_{< t}$|$C$|生成第 $t$ 步时的上下文（已生成的前缀）|
|$\pi_\theta$|$\pi$|由参数 $\theta$ 参数化的自回归 LLM 策略|
|$\theta$|$\theta$|模型参数（一致）|
|$r(y)$|—|奖励函数（本文中为负熵，无外部标签）|
|$z_t$|—|模型在第 $t$ 步产生的 logit 向量（EM-INF 特有）|
|$\pi_{\text{ref}}$|—|KL 正则化中的参考策略（初始基座模型）|
|$H(\cdot)$|—|Shannon 熵|
|$V$|—|词表（vocabulary）|

> 本文不涉及显式的外部记忆模块（$m_s$, $m_l$）或检索组件（$v$, $R$）。论文的核心贡献在于通过熵最小化改变策略 $\pi$ 本身，而非操纵记忆结构。下文将从"隐式记忆"的角度对其进行 RQ 分析。

---

## 1. Problem Setting

**要解决的问题：** 预训练 LLM 在推理任务上已经具备相当能力，但这些能力未被充分挖掘。本文探索：**仅通过熵最小化（Entropy Minimization, EM），不使用任何标注数据，能否提升 LLM 在数学、物理和编程推理任务上的表现？**

**形式化定义：**

- **输入：** prompt $q$
- **输出：** 自回归生成的轨迹 $T = (a_1, \dots, a_{\lvert T \rvert})$，其中 $\pi(a_t \mid q, a_{< t})$ 为每步 token 的条件分布
- **目标：** 最小化策略 $\pi$ 的熵 $H(\pi)$，使模型将概率质量集中于其最置信的输出

**任务类型与数据集：**

- 数学推理：MATH-500, AMC, AIME 2024, Minerva Math, OlympiadBench
- 编程：LeetCode, LiveCodeBench-v2
- 科学推理：SciCode, UGPhysics
- 价值对齐（反例）：IndVal

> 本文的核心假设是"模型置信度与正确性正相关"。这在推理任务上成立，但在价值对齐等任务上失效（§5），构成了方法的适用边界。

---

## 2. Training Procedure

本文提出三种后训练方法，前两种涉及参数更新，第三种仅在推理时操作。

### 2.1 EM-FT：直接 token 级熵最小化微调（§3.1）

类似于监督微调，但**不使用标签**。从策略 $\pi$ 中采样轨迹，直接最小化 token 级熵：

$$\hat{H}_{\text{tok}}(\pi) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{\lvert T^i \rvert} H\bigl(\pi(\cdot \mid a^i_{< t})\bigr)$$

其中 $H(\pi(\cdot \mid a_{< t})) = -\sum_{j \in V} \pi(j \mid a_{< t}) \log \pi(j \mid a_{< t})$。

- 采样 $N=1$ 条轨迹即可获得与 GRPO/RLOO（$N=4$，需标签）相当的性能
- 梯度直接对熵求导：$\frac{1}{N}\sum_i \sum_t \nabla_\theta H(\pi(\cdot \mid a^i_{< t}))$

### 2.2 EM-RL：以负熵为唯一奖励的强化学习（§3.2）

使用 REINFORCE 风格的策略梯度，但奖励信号**完全来自熵**，无任何外部标签。

**EM-RL-sequence**（轨迹级熵）：

$$r_{\text{traj}}(T) = \sum_{t=1}^{\lvert T \rvert} \log \pi(a_t \mid a_{< t}) = \log \pi(T)$$

（论文符号：$r_{\text{traj}}(y) = \log \pi_\theta(y)$ ）

**EM-RL-token**（token 级熵）：

$$r_{\text{tok}}(T) = -\sum_{t=1}^{\lvert T \rvert} H\bigl(\pi(\cdot \mid a_{< t})\bigr)$$

（论文符号：$r_{\text{tok}}(y) = -\sum_t H(\pi_\theta(\cdot \mid y_{< t}))$ ）

两种奖励均使用 RLOO baseline 减方差，并加入小系数（$\beta=0.001$）KL 正则化 $\text{KL}[\pi | \pi_{\text{ref}}]$ 以防止策略偏离基座模型过远。

### 2.3 训练设置

- 数据：35K 数学 prompt（Numina）+ 25K 编程 prompt（Eurus-2），**均不使用标签**
- 模型：Qwen2.5-Math-7B（数学）、Eurus-2-7B-SFT（编程）
- $N=4$ rollouts, batch size 512, lr $1\times10^{-6}$
- 使用 Verl 框架，4×GH200 GPU
- 基于验证集 early stopping

**Memory 如何参与训练：** 本文不引入显式的外部记忆。可以将预训练权重 $\theta$ 视为一种"隐式长期记忆"$m_l$——EM 的本质是**强化预训练阶段已编码在参数中的知识**，而非注入新知识。

> EM-FT 与 EM-RL 虽然目标相同（最小化 $H(\pi)$），但优化路径不同：EM-FT 直接对熵求导，EM-RL 通过策略梯度间接优化。Table 1 清晰对比了两者的梯度形式。

---

## 3. Reward Signal

|方法|奖励 / 损失|是否需要标签|与 memory 的关系|
|---|---|---|---|
|EM-FT|直接最小化 $\hat{H}_{\text{tok}}$（损失函数）|否|强化 $\theta$ 中已有的置信模式|
|EM-RL-sequence|$r = \log \pi(T)$（轨迹对数概率）|否|奖励高概率轨迹，巩固参数记忆|
|EM-RL-token|$r = -\sum_t H(\pi(\cdot \mid a_{< t}))$（负 token 熵）|否|奖励每步确定性高的生成|
|SC-RL（baseline）|$r = \text{freq}(a_i) / N$（多数投票频率）|否（但需答案提取）|间接降低答案分布熵|

**核心洞察：** 负熵本身就是一种有效的伪奖励信号，因为对于已具备推理能力的模型，置信度与正确性具有正相关性。这使得无标签训练成为可能。

**与 RL reward 的类比：** EM-FT 并非标准 RL，但其优化目标（最小化熵）等价于 EM-RL 中的期望奖励最大化 $\mathbb{E}_{T \sim \pi}[r(T)] = -\hat{H}(\pi)$，只是绕过了策略梯度，直接对目标函数求导。

> 值得注意的是，熵最大化（鼓励探索）在无标签场景下会导致策略崩溃，而熵最小化（鼓励确定性）则能稳定收敛。这与传统 RL 中熵正则化促进探索的范式形成对比。

---

## 4. Inference Procedure

### EM-INF：推理时 logit 优化（§4）

**核心思路：** 不更新模型参数 $\theta$，而是将每步的 logit 向量 $z_t \in \mathbb{R}^{\lvert V \rvert}$ 视为可优化的自由参数，通过梯度下降最小化其诱导分布的熵。

**目标函数：**

$$\mathcal{L}_{\text{EM-INF}} = \max\Bigl(-\sum_{j \in V} \sigma(z_t)_j \log \sigma(z_t)_j,; \delta\Bigr)$$

其中 $\sigma$ 为 softmax，$\delta \in (0.1, 0.5)$ 为最小熵阈值（防止退化为 greedy decoding）。

**流程：**

1. 模型前向传播，产生 logit $z_t$
2. 冻结 $\theta$，对 $z_t$ 做 5–15 步梯度下降以最小化熵
3. 使用优化后的 $z_t$ 进行采样解码（非 greedy，因为 Prop.1 表明 top logit 不变）
4. 计算复杂度：$O(n)$ 次前向传播（与常规解码相同），而自洽/迭代精炼需要 $O(Nn)$

**与训练阶段的差异：**

- EM-FT / EM-RL 更新参数 $\theta$；EM-INF **不更新任何参数**
- EM-INF 在每个 token 的生成步独立优化，无需训练数据
- EM-INF 对任务类型无假设（不需要答案提取），适用于所有场景

**关键结果：** Qwen2.5-32B + EM-INF 在 SciCode 主问题上达到 10.7% 准确率，超越 GPT-4o（9.2%）、Claude 3 Opus（4.7%）等前沿模型（Table 4），且计算效率为自洽方法的 3 倍（Figure 1）。

> EM-INF 与 adaptive temperature scaling 的关键区别在于：温度缩放保持 logit 的相对顺序不变，而 logit 优化可以改变非 top logit 的排序（Prop.1）。这在高熵（高不确定性）场景下尤为重要，因为重新排序可能引导出更优的推理链。

---

## 5. RQ 分析

### RQ1: What is memory?

本文章不引入显式记忆模块。但可以将预训练参数 $\theta$ 理解为一种隐式的长期记忆 $m_l$，模型在预训练阶段将推理能力编码在参数中。EM 的核心论点是：这种隐式记忆中已经包含了足够的推理能力，只需通过熵最小化加以"锐化"即可释放。

### RQ2: How does memory evolve and operate?

- EM-FT / EM-RL： 通过训练更新 $\theta$，使隐式记忆中的置信模式得到强化。高熵（不确定）的推理路径被抑制，低熵（确定）的路径被巩固。这是一种"记忆锐化"过程而非"记忆扩充"。
- EM-INF： 不改变 $\theta$（记忆不变），而是在推理时通过 logit 优化临时调整输出分布，可视为对工作记忆 $m_s$ 的即时干预。

### RQ3: Which component is optimized? Which signal is used?

- 优化组件： EM-FT 和 EM-RL 优化模型参数 $\theta$；EM-INF 优化 logit $z_t$（不涉及 $\theta$）
- 优化信号： 均为模型自身的熵，无外部标签、无奖励模型、无人类反馈。负熵作为伪奖励信号，利用"置信度 ≈ 正确性"的假设。

### RQ4: Regarding online optimization

EM-INF 是一种在线优化方法。 它在推理时逐 token 进行 logit 优化，无需预先收集数据或进行模型训练。每个用户请求独立处理，无需对特定任务集做优化。其计算开销可忽略（logit 优化等价于优化一个 $|V|$ 参数的单层网络）。EM-FT 和 EM-RL 则为离线训练方法，不涉及在线优化。

---

## Conclusion

我认为从内存角度看，这篇文章证明了一个看似简单的信号（模型自身输出分布的熵）可以在完全无标签的条件下显著提升 LLM 的推理能力。作者提出三种方法：EM-FT 通过直接最小化 token 级熵进行无监督微调；EM-RL 将负熵作为唯一奖励进行强化学习，是一个典型的自监督奖励信号的例子，它不需要外部验证；EM-INF 在推理时优化 logit 以降低熵，无需更新任何参数。从在线的角度来看，EM-INF 是最有价值的组件，它在每个解码步骤中优化 logits，而无需修改 $\theta$ 本质上是一种测试时间适应形式。我的观点就是预训练参数 $\theta$ 熵最小化可以作为一种长期记忆的形式，它是一种强化机制，可以强化模型已知的知识，而不是注入新的知识。