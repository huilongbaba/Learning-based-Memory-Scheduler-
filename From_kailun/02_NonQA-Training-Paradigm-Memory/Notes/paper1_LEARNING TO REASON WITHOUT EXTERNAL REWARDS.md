# Learning to Reason Without External Rewards

**Source:** [arXiv:2505.19590](https://arxiv.org/abs/2505.19590) (Published as a conference paper at ICLR 2026, arXiv v3: 2 Mar 2026)

---

## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$q$|$q$|input query / question|
|$o$ (output)|论文中 $o$ 同时承担框架中 $t$（reasoning trace）和 $A_n$（final answer）的角色，因为 INTUITOR 不区分推理过程与最终答案||
|$o_{< i}$|可视为 $m_s$（working memory）的序列展开形式：生成第 $i$ 个 token 时，前 $i-1$ 个 token 构成当前工作记忆||
|$\pi_\theta$|$\pi$|policy of the model|
|$\theta$|$\theta$|model parameters|
|$\pi_{\text{ref}}$|—|reference policy（框架未定义，建议新增符号 $\pi_0$ 表示初始参考策略）|
|$u(q, o)$|—|intrinsic reward signal（框架未定义，建议新增符号 $r_{\text{int}}$ 表示 intrinsic reward）|
|Self-certainty|—|基于 $\text{KL}(U \Vert p_{\pi_\theta})$ 的 confidence metric（框架未定义，建议新增符号 $\sigma$ 表示 self-certainty score）|
|$G$|—|group size，即每个 query 采样的候选输出数量|
|$\hat{A}_{i,t}$|—|advantage estimate（标准 RL 概念，框架未显式定义）|

**注意：** 本文不涉及外部长期记忆 $m_l$、检索算法 $v$、或 RAG 系统 $R$。论文的"memory"仅以 KV cache 形式隐式存在于 context window 内（即 $m_s$）。

---

## A. QA 依赖度分析

**该论文的 reward 是否 100% 来自 QA 结果？**

不是。这是本文的核心创新：INTUITOR 将 reward 完全替换为模型自身的 self-certainty 分数 $\sigma$，定义为均匀分布 $U$ 与模型 next-token 分布之间的平均 KL 散度：

$$\sigma(o \mid q) := \frac{1}{\lvert o \rvert} \sum_{i=1}^{\lvert o \rvert} \text{KL}(U \Vert p_{\pi_\theta}(\cdot \mid q, o_{< i}))$$

训练过程中不使用任何 gold answer，不进行 answer matching，reward 完全由模型内部信号决定。

**去掉 QA 问题后，训练流程是否完全崩溃？**

不会崩溃。INTUITOR 仅需要 question/prompt $q$（不需要对应的 answer），训练在 MATH 数据集的 question-only 部分上即可运行。但 $q$ 的分布仍然重要——训练仍然需要有意义的 prompt 来触发推理。

**是否存在任何隐含的非 QA 奖励成分？**

Self-certainty 本身就是一个完全非 QA 的奖励信号。它奖励的是模型对自身输出的"确信程度"，而非输出是否匹配某个标准答案。此外，KL penalty $\beta \cdot D_{\text{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$ 也是一个隐含的正则化信号，防止策略过度偏离初始分布。

> INTUITOR 是目前少数完全脱离 QA reward 的 RL 训练方法之一。但值得注意的是，评估仍然 100% 依赖 QA 指标（GSM8K, MATH500, LiveCodeBench accuracy 等）。这形成了一个有趣的不对称：训练信号是非 QA 的，但成功标准仍由 QA 定义。

---

## B. Memory 价值盲区

**方法能否激励存储"当前无对应问题、未来可能有用"的信息？**

不适用。INTUITOR 不涉及显式的 memory 操作（store / retrieve / delete / update）。模型的"记忆"仅限于 context window 内的 KV cache（$m_s$），不存在持久化的外部记忆 $m_l$。因此，不存在"选择性存储"的决策过程。

**Delete/Update 是否也完全由 QA 表现驱动？**

不适用。框架中没有 memory operation。

**是否区分了不同类型的记忆价值（informational / procedural / contextual）？**

未显式区分。但论文的实验结果隐含地表明 INTUITOR 能同时提升多种能力：procedural knowledge（数学推理步骤、代码生成流程）、contextual understanding（instruction following）。Self-certainty 作为 reward 似乎能间接激励模型组织更好的"内部工作记忆"——即更长、更结构化的 reasoning trace $t$。

> **评注：** 虽然 INTUITOR 不直接处理 memory 问题，但其 self-certainty 信号暗示了一种 intrinsic quality metric：模型对自身输出越确信，说明其内部"工作记忆"组织得越好。这种思路可以迁移到 memory operation 的 reward 设计中，例如，memory 操作后模型 self-certainty 的变化可以作为该操作质量的代理指标。

---

## C. QA 评估的局限性

**评估中是否有任何非 QA 指标？**

部分有。AlpacaEval 2.0（Length-Controlled Win Rate）评估的是 instruction following 质量，由 GPT-4.1 判断，而非简单的 answer matching。此外，论文还报告了 response length evolution（Figure 3）、reasoning emergence（Figure 6）、self-certainty 分布分离度（Figure 8, Mann-Whitney U test）等定性/定量分析，但这些不作为主要评估指标。

**如果构造"关键信息存在但从不被提问"的测试，该方法预期表现如何？**

不适用。INTUITOR 不涉及信息的选择性存储和检索。但从其设计哲学推断：self-certainty 奖励的是"模型对输出的确信度"，而非"输出是否回答了特定问题"。理论上，如果 self-certainty 被用于训练 memory 系统，它可能会激励存储那些能提升模型整体确信度的信息，即使这些信息不对应任何具体问题。

**消融实验是否揭示了 QA 指标无法捕捉的现象？**

是的，论文发现了多个 QA accuracy 无法完全捕捉的现象：

1. **Emergent reasoning**：INTUITOR 训练的模型会自发生成推理步骤（Figure 5, 6），即使 prompt 未要求。这种行为改善了代码生成质量，但 QA accuracy 无法衡量推理过程本身的质量。
2. **Self-certainty 分离度**：Figure 8 显示 INTUITOR 训练后的模型能更好地区分自身正确和错误的回答（$r = 0.45$, $p = 8.2 \times 10^{-24}$），这是一种"自我校准"能力，QA accuracy 无法捕捉。
3. **Cross-domain generalization pattern**：Figure 4 显示 LiveCodeBench 性能在 MATH500 accuracy 停滞后仍继续上升，说明 QA accuracy 在单一 benchmark 上无法反映模型整体能力的持续增长。

> **评注：** Self-certainty 的分离度是一个特别值得关注的非 QA 指标。它衡量的是模型"知道自己知道什么"的能力。一个好的 memory 系统应该知道哪些记忆是可靠的、哪些需要更新。

---

## D. 非 QA 范式的可能性

**方法中是否有可脱离 QA 独立评估的组件？**

是的。Self-certainty 本身就是一个完全独立于 QA 的指标。它可以在无 ground truth 的情况下评估：

- 模型对自身输出的确信程度
- 不同输出候选的相对质量（self-certainty ranking）
- 训练过程中模型内部信号的稳定性（online vs. offline annotator, Figure 7）

**框架能否接入非 QA reward？需要哪些修改？**

INTUITOR 本身就是一个接入非 QA reward 的框架。其设计具有高度模块化：在 GRPO 的 advantage computation 中，只需替换 $u_i$ 的计算方式即可接入任何 intrinsic reward。论文在 Section 6 中明确提出可以将 self-certainty 与其他信号（如 RLHF、RLVR、formatting reward）组合使用。

对于 memory 场景的可能修改：

1. 将 $u_i$ 替换为 memory operation 后的 self-certainty 变化量：$\Delta\sigma = \sigma_{\text{after}} - \sigma_{\text{before}}$
2. 引入 memory utilization rate 作为额外的 intrinsic reward
3. 使用 counterfactual contribution：比较有/无特定记忆时的 self-certainty 差异

**Memory operation 设计是否暗示了某种内在质量标准？**

本文不涉及 memory operation，但 self-certainty 本身暗示了一个强有力的内在质量标准：**好的输出 = 模型自己确信的输出**。这个标准可以推广到 memory：**好的记忆操作 = 能提升模型后续输出确信度的操作**。

> **评注：** INTUITOR 的最大贡献在于证明了 intrinsic reward（self-certainty）可以替代 external reward（QA accuracy）驱动 RL 训练。这为 memory 系统的非 QA 训练提供了直接的技术路径。关键问题是：self-certainty 在 memory-intensive 场景中是否仍然是一个好的代理指标？例如，模型可能对错误的记忆非常确信（confident but wrong），这需要进一步研究。

---

## 5. RQ 分析

### Q1: QA reward 是否是 memory 的充分训练信号？

这篇文章用实验证明了 QA reward 不是训练推理能力的必要条件：self-certainty 作为唯一 reward 信号，可以在数学推理上匹配 GRPO（使用 gold answer）的表现，并在代码生成上实现更好的跨域泛化。但论文未直接讨论 memory 场景。从 self-certainty 作为 process-level reward（跨所有 token 计算，而非仅看最终答案）这一特性来看，它比 QA 的 0/1 binary reward 更能捕获推理过程的质量，部分缓解了 credit assignment 路径过长的问题。

### Q2: QA benchmark 是否是 memory 的充分评估标准？

论文未直接涉及此问题。

### Q3: 能否构造非 QA 驱动的 memory 训练数据与信号？

这篇文章提供了一个非 QA intrinsic reward 的成功案例：self-certainty。论文证明了以下关键点：

1. **Intrinsic reward 可行**：self-certainty 作为唯一信号即可驱动有效训练
2. **Online 信号优于 offline**：reward 信号需要与 policy 共同演化以防止 reward hacking（Figure 7）
3. **信号可组合**：Section 6 和 Appendix B.7 探讨了将 self-certainty 与 golden-answer reward 组合的可能性

对于 memory 场景，self-certainty 可以作为一种 intrinsic reward 的候选（如 information gain 的代理）。但论文未涉及长期交互、持续规划、偏好跟踪等 memory-specific 的数据构造。

> **评注：** INTUITOR 对 Q3 的贡献最大，它证明了 intrinsic reward 可以工作。但从 memory 角度看，self-certainty 衡量的是"模型对当前输出的确信度"，而非"记忆操作对未来的价值"。将 self-certainty 扩展到 temporal credit assignment（跨多轮交互的记忆价值评估）是一个重要的开放问题。

---

## Conclusion

这篇论文提出了 RLIF (Reinforcement Learning from Internal Feedback) 范式，用模型自身的 self-certainty（基于输出分布与均匀分布的 KL 散度）替代 RLVR 中的外部可验证奖励，实现了完全无监督的 RL 训练。self-certainty 本质上是在评估模型对其上下文（隐式 working memory）的理解质量。涌现的 pre-reasoning 行为可以理解为模型学会了主动构建更有效的 "working memory" 来提升自身确信度。训练过程中涌现了类似 DeepSeek-R1 的长链推理行为，且 online self-certainty 机制有效防止了 reward hacking。我认为它提出了一种不依赖外部信号的 online 优化方案，并且文章中还通过 online vs offline 的对比实验清晰展示了 co-evolving reward 对防止 reward exploitation 的关键作用。
对于 memory 研究，我认为最大的启示是：intrinsic model signal 可以替代 external QA reward 驱动 RL 训练，这为构造非 QA 的 memory 训练信号提供了概念验证和技术基础。