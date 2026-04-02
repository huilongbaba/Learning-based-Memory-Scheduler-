# Learning to Reason without External Rewards

**source:** [arXiv:2505.19590v3](https://arxiv.org/abs/2505.19590) ICLR 2026

---

## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$q$|$q$|输入问题 (input query)|
|$o$ / $o_i$|$a$|模型生成的输出 (generated output)，对应框架中的 action|
|$\pi_\theta$|$\pi$|当前策略 (policy)|
|$\pi_{\text{ref}}$|—|参考策略 (reference policy)，框架未显式覆盖，建议符号 $\pi_{\text{ref}}$|
|$\pi_{\theta_{\text{old}}}$|—|行为策略 (behavior policy)，即上一轮迭代的策略|
|$\theta$|$\theta$|模型参数|
|$u(q, o)$ / $u_i$|—|内在奖励信号 (intrinsic reward)，建议新符号 $r_{\text{int}}$，表示 intrinsic reward|
|Self-certainty|—|自确信度，基于 $\text{KL}(U \| p_{\pi_\theta})$ 的置信度度量，建议符号 $\text{SC}$|
|$\hat{A}_{i,t}$|—|优势估计 (advantage estimate)|
|$G$|—|每个 query 采样的候选输出数量 (group size)|
|$o_{<i}$|$C$|已生成的 token 序列，构成当前 token 的上下文|
|$V$|—|词表 (vocabulary)|
|$T$ (trajectory) 概念隐含于 rollout|$T$|轨迹，论文中体现为对每个 $q$ 采样 $G$ 条完整输出序列|


---

## 1. Problem Setting

**要解决的问题：** RLVR (Reinforcement Learning with Verifiable Rewards) 依赖外部可验证奖励（如数学题的标准答案、代码的测试用例），成本高且受限于特定领域。本文提出 **RLIF (Reinforcement Learning from Internal Feedback)**，探索 LLM 能否仅凭自身内在信号提升推理能力，无需任何外部奖励或标注数据。

**形式化定义：**

- **输入：** 问题 $q \sim P(Q)$（如 MATH 数据集中的数学题，仅需题目文本，不需要答案）
- **输出：** 模型生成的回答 $a \sim \pi_\theta(\cdot | q)$
- **目标：** 最大化内在奖励 $r_{\text{int}}(q, a)$，同时约束策略不偏离参考策略 $\pi_{\text{ref}}$

**任务类型与数据集：**

- 训练数据：MATH 训练集 (7,500 题，仅用题目)、Codeforces 代码竞赛题
- 评估：GSM8K, MATH500 (in-domain 数学)；LiveCodeBench-v6, CRUXEval-O (out-of-domain 代码)；MMLU-Pro, AlpacaEval (通用能力)

> 该设定的核心价值在于完全消除了对 gold answer 的依赖，使得 RL 训练可扩展到缺乏标准答案的开放领域。

---

## 2. Training Procedure

**训练流程概述：**

INTUITOR 基于 GRPO (Group Relative Policy Optimization) 框架，将外部奖励替换为 self-certainty 内在信号。流程如下：

1. 对每个问题 $q$，使用行为策略 $\pi_{\theta_{\text{old}}}$ 采样 $G$ 个候选输出 $a_1, \dots, a_G$
2. 对每个输出 $a_i$ 计算 self-certainty 分数 $u_i = \text{SC}(a_i | q)$
3. 通过组内归一化计算优势估计 $\hat{A}_{i,t}$
4. 使用 GRPO 的 clipped policy gradient 更新策略 $\pi_\theta$

**核心公式：**

Self-certainty 定义（论文 Eq. 2，框架符号标注）：

$$\text{SC}(a \mid q) := \frac{1}{\lvert a \rvert} \sum_{i=1}^{\lvert a \rvert} \text{KL}\bigl(U \Vert\, p_{\pi_\theta}(\cdot \mid q, a_{< i})\bigr) = -\frac{1}{\lvert a \rvert \cdot \lvert V \rvert} \sum_{i=1}^{\lvert a \rvert} \sum_{j=1}^{\lvert V \rvert} \log\bigl(\lvert V \rvert \cdot p_{\pi_\theta}(j \mid q, a_{< i})\bigr)$$

其中 $U$ 为词表上的均匀分布，$a_{<i}$ 对应上下文 $C$。

GRPO 目标函数（论文 Eq. 3，Section 3.2）：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q, {a_i}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \left( \min\left[ c_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}_\epsilon(c_{i,t}(\theta)) \hat{A}_{i,t} \right] - \beta D_{\text{KL}}(\pi_\theta | \pi_{\text{ref}}) \right) \right]$$

其中 $c_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} | q, a_{i,<t})}{\pi_{\theta_{\text{old}}}(a_{i,t} | q, a_{i,<t})}$ 为重要性权重。

优势估计（论文 Eq. 3）：

$$u_i = \text{SC}(a_i | q), \quad \hat{A}_{i,t} = \frac{u_i - \text{mean}({u_1, \dots, u_G})}{\text{std}({u_1, \dots, u_G})}$$

**Memory 如何参与训练：** 本文不显式使用外部 memory ($m_l$)。Self-certainty 可理解为模型在生成过程中对其隐式 working memory ($m_s$，即 KV cache / 已生成上下文) 质量的自评估。奖励信号来源于模型对每个 token 位置输出分布的 "锐度" 的平均值。

**关键设计选择：**

- 使用 **online** self-certainty（奖励模型随策略共同演化），避免 offline 固定模型导致的 reward hacking（Section 5.4）
- KL 惩罚 $\beta = 0.005$（默认），防止策略偏离过远（Appendix B.1）
- Group size $G = 7$（默认），每步处理 128 个问题

> Self-certainty 使用 $\text{KL}(U | p)$（mode-seeking）而非熵 $H(p)$（mode-covering），论文指出这使其对生成长度的偏差更小。我觉得这是一个有意思的设计选择，与 concurrent work EM-RL 的熵最小化形成对比。

---

## 3. Reward Signal

**奖励信号：** Self-certainty $\text{SC}(a | q)$，即模型在生成每个 token 时输出分布与均匀分布之间 KL 散度的平均值。值越高表示模型越 "确信"。

**奖励如何与 memory 交互：** Self-certainty 是逐 token 计算并取平均的过程奖励（process reward），而非仅基于最终结果的结果奖励（outcome reward）。它评估的是模型在整个生成轨迹 $T$ 上对上下文 $C = (q, a_{<i})$ 的理解程度。

**与 RL reward 的关系：**

- 在 GRPO 框架中，self-certainty 分数直接替代外部 binary reward（正确/错误），经过组内 z-score 归一化后作为优势估计
- 这是一个 **连续、密集、过程感知** 的奖励信号，对比 RLVR 的 **二元、稀疏、结果导向** 奖励
- 论文实验表明（Section 5.4, Figure 8），经过 INTUITOR 训练后，self-certainty 对正确/错误回答的区分能力显著增强（Mann-Whitney U test, $p = 8.2 \times 10^{-24}$, $r = 0.45$）

**与替代信号的对比（Appendix B.4）：**

- 熵最小化 (EM)：导致重复退化和模型崩溃
- 随机奖励：严重降低性能
- Log probability（unnormalized）：偏向短生成，导致退化
- Normalized log probability：偏向长生成，被快速利用
- Self-certainty 在所有替代方案中表现最稳定

> Self-certainty 作为奖励的一个隐含假设是：模型的置信度与输出质量正相关。这在预训练充分的模型上成立，但在预训练不足的模型上可能失败，例如 Llama3.2-3B-Base 在 MATH 上训练失败。

---

## 4. Inference Procedure

**推理时流程：**

- 使用 greedy decoding，与训练时的采样策略不同（训练时 temperature = 0.9）
- 推理时不涉及 self-certainty 计算或任何特殊的 memory 读/写机制
- 使用与训练相同的 chat-style prompting format（MMLU-Pro 除外）

**与训练阶段的差异：**

- 训练时：对每个 $q$ 采样 $G$ 个候选并计算 self-certainty 做相对比较
- 推理时：直接 greedy decode 单条输出，无需 self-certainty 评分
- 推理时观察到的涌现行为：模型自发产生 pre-reasoning（先用自然语言推理，再生成结构化输出），这在训练 prompt 中并未要求（Section 5.2, Figure 5）

> 推理阶段的简洁性是 INTUITOR 的优势之一——训练完成后不需要额外的 reward model 或 verifier。涌现的 long-form reasoning 行为是一个值得深入研究的现象。

---

## 5. RQ 分析
### RQ1: What is memory?

本文未显式定义或使用外部 memory。隐式地，模型的 working memory 体现为生成过程中的上下文 $(q, a_{<i})$，即 KV cache。Self-certainty 衡量的正是模型对该隐式 working memory 的 "理解程度"。

### RQ2: How memory evolves, operates?

隐式 memory（上下文）在生成过程中逐 token 增长。INTUITOR 的训练鼓励模型生成更长、更详细的推理链，从而丰富上下文信息。涌现的 pre-reasoning 行为表明模型学会了主动构建更有效的 "working memory"。无外部 memory 的更新机制。

### RQ3: Which component is optimized? Which signal is used?

优化的组件是策略网络参数 $\theta$（即 LLM 自身）。使用的信号是 self-certainty，一种基于 $\text{KL}(U | p_{\pi_\theta})$ 的内在置信度度量，通过 GRPO 的 advantage-weighted policy gradient 进行优化。

### RQ4: Regarding online optimization

我认为是涉及的。 INTUITOR 使用 online self-certainty（奖励由当前策略模型计算，而非固定的 base model）。Section 5.4 和 Figure 7 明确对比了 online vs. offline 两种模式：offline 模式在约 100 步后遭遇 reward exploitation（模型学会在答案后附加已解决的问题以膨胀 certainty），而 online 模式因奖励信号与策略共同演化而避免了这一问题。


---

## Conclusion

这篇论文提出了 RLIF (Reinforcement Learning from Internal Feedback) 范式，用模型自身的 self-certainty（基于输出分布与均匀分布的 KL 散度）替代 RLVR 中的外部可验证奖励，实现了完全无监督的 RL 训练。self-certainty 本质上是在评估模型对其上下文（隐式 working memory）的理解质量。涌现的 pre-reasoning 行为可以理解为模型学会了主动构建更有效的 "working memory" 来提升自身确信度。训练过程中涌现了类似 DeepSeek-R1 的长链推理行为，且 online self-certainty 机制有效防止了 reward hacking。我认为它提出了一种不依赖外部信号的 online 优化方案，并且文章中还通过 online vs offline 的对比实验清晰展示了 co-evolving reward 对防止 reward exploitation 的关键作用。