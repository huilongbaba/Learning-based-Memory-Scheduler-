# Spurious Rewards: Rethinking Training Signals in RLVR

**Source:** [arXiv:2506.10947v2](https://arxiv.org/abs/2506.10947) (February 26, 2026)

---

## 符号映射表

|论文符号|框架符号|说明|
|---|---|---|
|$x$|$q$|输入 prompt / question|
|$y$|$T$|模型生成的 rollout（trajectory）|
|$y_t$|—|trajectory 中第 $t$ 个 token|
|$\pi_{\theta}$|$\pi$|当前策略|
|$\pi_{\text{old}}$|—|行为策略（上一步的策略），框架未显式区分|
|$\pi_{\text{ref}}$|—|冻结参考策略，框架未覆盖，建议新符号 $\pi_{\text{ref}}$|
|$\theta$|$\theta$|模型参数（一致）|
|$r(x, y)$|—|reward 函数，框架中未显式定义，建议 $r(q, T)$|
|$\hat{A}(x, y)$|—|group-relative advantage，框架未覆盖，建议 $\hat{A}(q, T)$|
|$\rho_t(y; \theta)$|—|token 级 importance ratio $\pi_\theta(y_t \mid x, y_{< t}) / \pi_{\text{old}}(y_t \mid x, y_{< t})$，框架未覆盖|
|$\epsilon_c$|—|PPO 风格的 clipping 阈值，框架未覆盖|
|$G$|—|每个 prompt 的 rollout 数量，框架未覆盖|
|$\gamma$|—|随机奖励的 Bernoulli 概率参数，框架未覆盖|

**注意：** 本文不涉及 memory 系统（$m_s$, $m_l$, $v$, $R$ 等均未出现）。论文聚焦于 RLVR 训练信号本身的性质，而非 memory 操作。

---

### A. QA 依赖度分析

本文的 reward 设计覆盖了从标准 ground-truth 到完全无信息的 random reward 的光谱。标准设置中，reward 是严格的 QA 结果：$r(q, T) = \mathbb{1}(\hat{y} = y^*)$，即 100% 来自 QA。然而论文的核心发现是：**即使将 reward 替换为与正确答案无关（random）甚至负相关（incorrect label）的信号，Qwen2.5-Math 模型仍然能获得显著的性能提升**（random reward 在 MATH-500 上提升 21.4%，接近 ground-truth 的 29.1%）。

去掉 QA 问题后，训练流程并不崩溃——至少对 Qwen 系列模型而言。这说明 RLVR 的真正训练信号可能部分来自 **GRPO 优化算法本身的 clipping bias**，而非 reward 的语义内容。具体而言，clipping 机制会系统性地放大模型预训练阶段已有的高概率行为（如 code reasoning），即使 reward 完全随机。

隐含的非 QA 奖励成分包括：format reward（只要输出包含 `\boxed{}` 即给正奖励）和 Python reward（只要输出包含 `python` 字符串即给正奖励），这些都是纯格式/行为层面的信号，与答案正确性无关。

> Spurious reward 的有效性高度依赖模型，在 Llama3 和 OLMo2 上，spurious reward 几乎无效甚至有害，说明 pretraining prior 才是决定因素。

---

### B. Memory 价值盲区

本文**不涉及 memory 系统**，因此以下分析是间接推断：

- **能否激励存储"当前无对应问题、未来可能有用"的信息？** 不适用。本文没有 memory 操作。但论文揭示的核心现象——RLVR 主要放大预训练中已有的行为模式——暗示：如果将此机制迁移到 memory 场景，RL 训练很可能只会强化"已经在预训练中学到的记忆模式"，而无法激励模型发现新的记忆策略。
    
- **Delete/Update 是否由 QA 表现驱动？** 不适用。
    
- **是否区分不同类型的记忆价值？** 不适用。但论文对 code reasoning vs. natural language reasoning 的区分提供了一个类比：不同的"推理策略"在预训练中有不同的先验概率，RLVR（尤其是带 clipping 的 GRPO）会选择性放大高先验策略。类似地，不同类型的 memory operation 也可能有不同的先验概率，RL 训练可能只放大"最容易做的记忆操作"而非"最有价值的记忆操作"。
    

> 虽然本文不直接讨论 memory，但其"RLVR 放大预训练先验而非学习新能力"的结论对 memory RL 研究有深刻警示：如果 memory operation 的价值完全由 QA reward 定义，那么 RL 可能只是在放大预训练中已有的记忆模式，而非真正学会"什么值得记"。

---

### C. QA 评估的局限性

论文的评估指标全部是 QA 准确率（MATH-500 pass@1、AMC avg@8、AIME avg@8），没有任何非 QA 指标。

然而，论文的实验本身就是对 QA 评估局限性的有力说明：

- **Code reasoning frequency** 是论文发现的一个与 QA 准确率高度相关但本质上非 QA 的指标。Qwen2.5-Math-7B 在 RLVR 训练中 code reasoning 频率从 65% 上升到 90%+，这一行为变化无法被 QA 准确率单独捕捉。
- **如果构造"关键信息存在但从不被提问"的测试：** 本文没有直接做此实验，但论文暗示这类测试会暴露 spurious reward 的本质——模型并没有学到新的推理能力，只是被 clipping bias 推向了预训练中已有的高先验行为。
- **消融实验揭示了 QA 指标无法捕捉的现象：** 去除 clipping 后，random reward 不再产生 QA 准确率提升（Figure 4），但这并不意味着模型没有变化——只是变化不再系统性地偏向高先验行为。此外，论文发现 AIME2025（模型未见过的题目）上 spurious reward 的增益大幅减少，说明 QA 指标在不同分布上的表现不一致。

> 论文虽然只用 QA 评估，但其核心发现恰恰说明了 QA 指标的欺骗性：在 Qwen 模型上用 random reward 得到的 +21.4% QA 提升，并不代表真正的推理能力提升，而只是预训练行为的放大。这是 QA 评估范式最鲜明的反例之一。

---

### D. 非 QA 范式的可能性

- **方法中可脱离 QA 独立评估的组件：** Code reasoning frequency 是一个完全不依赖 QA 的行为指标，论文证明它与 QA 准确率高度相关但可以独立测量。此外，token probability（$\pi_{\theta,x}(y)$ 的均值）和 lexical repetition rate 也是非 QA 指标。
    
- **框架能否接入非 QA reward？** 是的，论文已经展示了多种非 QA reward 的可行性：
    
    - Format reward：$r = \mathbb{1}(\text{response contains } \backslash\text{boxed}\lbrace\rbrace)$
    - Python reward：$r = \mathbb{1}(\text{response contains "python"})$
    - No-repetition reward：$r = \mathbb{1}(\text{no string repeated} > 10 \text{ times})$
    - Random reward：$r \sim \text{Bernoulli}(\gamma)$
    
    这些 reward 无需修改 GRPO 框架即可直接使用。
    
- **Memory operation 设计是否暗示了某种内在质量标准？** 不直接适用，但论文揭示的 clipping bias 机制暗示了一个内在标准：**行为与预训练先验的一致性**。GRPO 的 clipping 机制本质上在做"放大高先验行为、抑制低先验行为"，这本身就是一种不依赖外部 reward 的内在质量信号。
    

> 论文展示的 format reward、Python reward、no-repetition reward 都是行为层面的非 QA 信号，且在特定模型上有效。这为 memory 领域设计非 QA reward（如 memory utilization rate、retrieval consistency、information coverage）提供了直接的方法论参考。

---

### 5. RQ 分析

**Q1: QA reward 是否是 memory 的充分训练信号？**

这篇文章提供了强有力的反面证据：QA reward 甚至不是 RLVR 数学推理训练的必要信号。在 Qwen2.5-Math 上，random reward（零信息量）和 incorrect reward（负信息量）都能产生接近 ground-truth reward 的性能提升。论文将此归因于 GRPO clipping bias 对预训练先验的系统性放大。这意味着：在特定模型上，QA reward 的"训练信号"可能大部分来自优化算法的偏置而非 reward 本身的语义内容。对 memory 场景而言，这个发现更加令人担忧——如果 QA reward 连"是否在学习 reward 给出的信号"都无法保证，那么通过 QA reward 训练 memory operation 的有效性就更加可疑。

**Q2: QA benchmark 是否是 memory 的充分评估标准？**

论文间接回答了这个问题：QA benchmark 可以被 spurious reward "欺骗"。在 Qwen 模型上，random reward 训练后 MATH-500 准确率提升 21.4%，但这并非真正的能力提升，而是预训练行为（code reasoning）被 clipping bias 放大的结果。论文在 AIME2025（out-of-distribution）上的实验进一步证实了这一点：spurious reward 的增益在 OOD 数据上大幅缩水。这说明 QA benchmark 容易过度估计训练的真实效果，尤其是当模型的预训练数据与评测数据高度重叠时。

**Q3: 能否构造非 QA 驱动的 memory 训练数据与信号？**

这篇文章未直接讨论 memory，但提供了构造非 QA 信号的具体实例：format reward、Python reward、no-repetition reward 都是纯行为层面的信号，不依赖答案正确性。更重要的是，论文揭示了 GRPO clipping 本身就是一种 intrinsic reward，它系统性地偏好高先验行为，无需外部 reward 即可产生训练信号。这为设计 memory 领域的 intrinsic reward 提供了启发：可以考虑基于 memory operation 的先验一致性、信息增益、或反事实贡献来构造非 QA 信号。但论文也警示：这类信号的有效性高度依赖模型的预训练先验，必须在多个模型上验证。

---

### Conclusion

这篇文章通过一组精心设计的实验，揭示了 RLVR 中一个反直觉的现象：对于 Qwen2.5-Math 模型，即使使用完全随机或故意错误的奖励信号，GRPO 训练仍能带来接近真实奖励水平的数学推理性能提升（随机奖励 +21.4% vs 真实奖励 +29.1%）。作者将这一现象追溯到 GRPO 的 clipping 机制，它产生了一种系统性的梯度偏差，会放大模型在预训练中已经学到的高概率行为模式（如 Qwen-Math 的 "code reasoning"）。关键的是，这种效应高度依赖模型——同样的虚假奖励对 Llama3 和 OLMo2 无效甚至有害，因为这些模型缺乏可被放大的有益预训练先验。

因此我认为，优化算法本身（而不仅仅是奖励信号）能够决定哪些内容会被“记住”和强化，截断偏差会系统性地偏向高先验行为，而与奖励质量无关。另外，仅在 Qwen 模型上验证的 RLVR 方法可能高估了其训练信号的真实贡献，未来研究应在多种模型上验证，并使用虚假奖励作为基线校准。如果问答奖励甚至无法可靠地表明学习更好的推理，它又如何能表明学习记住什么呢？从答案到记忆操作的积分分配路径甚至比答案到推理策略的路径更长。