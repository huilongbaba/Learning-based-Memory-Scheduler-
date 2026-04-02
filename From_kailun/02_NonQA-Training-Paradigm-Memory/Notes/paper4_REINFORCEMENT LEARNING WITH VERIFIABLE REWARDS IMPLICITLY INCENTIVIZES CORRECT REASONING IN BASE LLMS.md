# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs

**Source:** [arXiv:2506.14245v2](https://arxiv.org/abs/2506.14245) (Preprint, arXiv cs.AI, October 2025)

---

## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$q$|$q$|question prompt，一致|
|$c_i$ (CoT in response $y_i$)|$t$|推理链/思维轨迹 (thought / reasoning trace)|
|$a_i$ (final answer in $y_i$)|$A_n$|最终答案|
|$y_i = (c_i, a_i)$|$(t, A_n)$|一条完整响应 = 推理轨迹 + 最终答案|
|$\pi_\theta$|$\pi$|模型策略 (policy)，由参数 $\theta$ 决定|
|$\theta$|$\theta$|模型参数，一致|
|$G$ responses $Y = {y_1, \dots, y_G}$|${T_1, \dots, T_G}$|一组采样轨迹 (trajectories)|
|$R(y_i) = \mathcal{I}_{\text{Ans}}(a_i)$|—|可验证奖励，基于答案正确性的二值奖励|
|$\hat{A}(y_i)$|—|GRPO 优势函数 (advantage)，本框架未预定义|
|$p_c^\theta$|—|生成正确 CoT 的概率，本文核心分析对象|
|$\alpha$|—|$P(A_n \text{ correct} \mid t \text{ correct})$，正确推理导致正确答案的概率|
|$\beta$|—|$P(A_n \text{ correct} \mid t \text{ incorrect})$，错误推理碰巧正确答案的概率|

**新概念说明：**

- 本文核心分析围绕 CoT 正确性（$\mathcal{I}_{\text{CoT}}$）与答案正确性（$\mathcal{I}_{\text{Ans}}$）的解耦展开，这不涉及传统意义上的外部 memory 组件（$m_s$ 或 $m_l$），而是分析 RL 训练如何隐式地改变模型内部的推理路径分布。
- CoT-Pass@K：论文提出的新评估指标，要求最终答案与中间推理步骤同时正确才计为通过。

---

## 1. Problem Setting

**论文要解决的问题：** RLVR（使用可验证奖励的强化学习）是否真正提升了 LLM 的推理能力，还是仅仅提高了采样效率？此前 Yue et al. (2025) 提出假说——所有正确推理路径已存在于基座模型中，RLVR 只是调整采样概率并以牺牲推理容量为代价。本文系统性地反驳该假说，论证 RLVR 能隐式激励正确推理。

**输入/输出形式化定义：**

- **输入：** 问题 $q$（数学题或编程题）
- **模型输出：** 响应 $y_i = (t_i, A_{n,i})$，其中 $t_i$ 为推理轨迹（CoT），$A_{n,i}$ 为最终答案
- **奖励：** $R(y_i) = \mathcal{I}_{\text{Ans}}(A_{n,i}) \in {0, 1}$，仅基于答案正确性，由确定性验证器给出

**任务类型与数据集：**

- 数学推理：AIME 2024/2025, MATH-500, AMC23, Minerva
- 代码推理：LiveCodeBench v1–v6
- 基座模型：Qwen2.5-32B → DAPO-Qwen-32B（RLVR 后）；DeepSeek-R1-Distill-Qwen-7B → AceReason-Nemotron-7B / Skywork-OR1-7B

>这篇文章不涉及外部 memory（$m_s$, $m_l$）的使用，其核心贡献在于理论解释 RLVR 如何通过 answer-only reward 隐式优化推理路径质量，这可视为对模型"内部推理记忆"（implicit reasoning memory encoded in $\theta$）的优化过程。

---

## 2. Training Procedure

**训练流程形式化描述：**

本文采用 GRPO（Group Relative Policy Optimization）算法，训练流程如下：

1. 对每个问题 $q$，从策略 $\pi_\theta$ 中采样 $G$ 条响应 $Y = {y_1, \dots, y_G}$
2. 对每条响应计算可验证奖励 $R(y_i) = \mathcal{I}_{\text{Ans}}(A_{n,i})$
3. 计算组内归一化优势：

$$\hat{A}(y_i) = \frac{R(y_i) - \mu_Y}{\sigma_Y}$$

其中 $\mu_Y = \frac{1}{G}\sum_{j=1}^{G} R(y_j)$，$\sigma_Y = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(R(y_j) - \mu_Y)^2}$

4. 策略梯度更新：

$$\nabla_\theta J(\theta) \approx \frac{1}{G}\sum_{i=1}^{G} \hat{A}(y_i) \nabla_\theta \log \pi_\theta(y_i \mid q)$$

**Memory 如何参与训练：**

本文不使用显式的外部 memory。但从 memory 视角来看，RLVR 的作用可以理解为：通过策略梯度优化 $\theta$，隐式地重塑了模型参数中编码的推理模式分布——增强正确推理路径的概率，抑制错误推理和猜测路径。这等价于对模型"参数化记忆"的重新组织。

**核心理论（Theorem 1, §4）：**

在 **Logic Prior 假设** 下（$\alpha > \beta$，即正确 CoT 比错误 CoT 更可能产生正确答案），GRPO 满足：

$$E[\hat{A}(y_i) \mid t_i \text{ correct}] \to \frac{(1 - p_c)(\alpha - \beta)}{\sigma} > 0$$

$$E[\hat{A}(y_i) \mid t_i \text{ incorrect}] \to \frac{-p_c(\alpha - \beta)}{\sigma} < 0$$

其中 $p_c = P_{\pi_\theta}(\mathcal{I}_{\text{CoT}}(t) = 1)$。因此，即使奖励仅基于答案正确性，正确 CoT 获得正优势、错误 CoT 获得负优势，$p_c$ 单调递增。

> Logic Prior 假设（$\alpha > \beta$）是定理成立的关键前提。作者也讨论了该假设失败的情况（§4 Discussions on failure modes）：当基座模型保留了预训练中的错误知识偏差时，可能导致错误 CoT 被意外强化，这可能是 R1-Zero 出现可读性差、语言混合等问题的根因。

---

## 3. Reward Signal

**奖励信号：**

- 二值可验证奖励：$R(y_i) = \mathcal{I}_{\text{Ans}}(A_{n,i}) \in {0, 1}$
- 数学题：提取答案 token 与 ground truth 比对
- 代码题：实际执行生成代码并验证输出正确性

**奖励如何与 memory 交互：**

奖励不直接操作任何 memory 组件。其作用机制完全通过策略梯度间接实现：正确答案的奖励信号经 GRPO 优势归一化后，统计上倾向于强化具有正确推理轨迹 $t$ 的响应、抑制错误推理的响应。这一隐式激励机制是本文的核心理论贡献。

**CoT 质量验证（§6）：**

论文进一步通过 SFT 实验验证了 RLVR 后 CoT 的质量提升：在 DAPO 训练问题上，用不同阶段模型生成的 CoT 数据对基座模型做 SFT，发现后期 CoT 数据训练出的模型泛化性能更好，甚至能用 SFT 近似复制 RLVR 模型的 Pass@1 性能。

> 仅使用 answer-level reward 而非 process reward（如 PRM）就能隐式提升 CoT 质量，这一发现对 reward 设计有重要启示，即过程奖励可能不是必需的，但前提是基座模型已具备足够的"逻辑先验"。

---

## 4. Inference Procedure

**推理时的流程：**

推理阶段无特殊 memory 读/写操作。模型直接从优化后的策略 $\pi_\theta$ 中采样响应 $y = (t, A_n)$。

**评估创新——CoT-Pass@K：**

论文提出 CoT-Pass@K 指标，在采样 $K$ 条响应时，仅当某条响应的 CoT $t$ 和答案 $A_n$ 同时正确时才计为通过。实践中使用 DeepSeek-R1-0528-Qwen3-8B 作为 CoT 验证器，对每条 CoT 进行 3 次独立验证，采用 any-correct / all-correct / majority-correct 三种聚合策略以控制 false positive 和 false negative。

**与训练阶段的差异：**

- 训练时：仅使用 answer-level reward（$\mathcal{I}_{\text{Ans}}$）
- 评估时：额外引入 CoT-level 验证（$\mathcal{I}_{\text{CoT}}$），但该验证不参与训练

> CoT-Pass@K 的引入揭示了 Pass@K 在数学推理评估中的不可靠性，基座模型可能通过错误推理碰巧猜中简单格式的答案（如整数），导致 Pass@K 虚高。我认为这一指标设计思路同样值得在其他推理评估场景中推广。

---

## 5. RQ 分析

### RQ1: What is memory?

这篇文章未引入显示memory组件，但可将模型参数 $\theta$ 视为隐式的参数化记忆。RLVR 通过优化 $\theta$ 来重塑模型内部存储的推理模式分布，增强正确推理路径、抑制错误推理和猜测行为。

### RQ2: How memory evolves, operates?

$\theta$ 的演化由 GRPO 策略梯度驱动：正确 CoT 获得正优势被强化，错误 CoT 获得负优势被抑制，$p_c$（生成正确 CoT 的概率）单调递增。训练动态分析显示 $P(CC|CA)^{(q)}$（正确答案中正确 CoT 的比例）从训练初期即开始提升，且该能力可泛化至未见过的测试问题。

### RQ3: Which component is optimized? Which signal is used?

- 优化组件： 模型参数 $\theta$（即策略 $\pi_\theta$）
- 优化信号： answer-level 二值可验证奖励 $R(y_i) = \mathcal{I}_{\text{Ans}}(A_{n,i})$，经 GRPO 组内归一化后作为优势函数
- 核心发现： 尽管训练信号仅涉及答案正确性，但在 Logic Prior 假设下，GRPO 梯度会隐式激励正确推理路径

### RQ4: Regarding online optimization

本文的 RLVR 训练本质上就是在线优化：每一轮从当前策略 $\pi_\theta$ 中采样新的响应，计算奖励和优势，再更新策略。论文复现了 DAPO 的训练过程，观察到大多数训练问题在约 400 步后 $P(CA)^{(q)} \to 1$，但仍有约 30% 的正确答案伴随着有缺陷的 CoT（$P(CC|CA)$ 中位数约 0.7），揭示了纯 answer reward 在线优化的局限性。

---

## Conclusion

本文系统性地回应了"RLVR 是否真正提升 LLM 推理能力"这一争议。作者首先通过引入 CoT-Pass@K 指标（同时验证答案和推理过程的正确性），在数学和代码任务上发现 RLVR 确实扩展了推理能力边界，而非仅仅提升采样效率。在理论层面，作者证明了在"逻辑先验"假设（正确推理更可能导致正确答案）下，GRPO 的策略梯度会自动赋予正确 CoT 正优势、错误 CoT 负优势，从而隐式激励正确推理——即使奖励信号仅基于最终答案。训练动态实验进一步证实，这种正确推理的激励从训练早期即开始，且能泛化到测试集。最后，通过 SFT 实验验证了 RLVR 后生成的 CoT 质量显著提升，甚至可以用 SFT 近似复现 RLVR 模型的性能。