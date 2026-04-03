# The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning

**Source:** [arXiv:2505.15134](https://arxiv.org/abs/2505.15134) (May 2025)

---

## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$x$|$q$|输入 prompt / human input|
|$y = y_1 \dots y_{\lvert y \rvert}$|$T$|输出 trajectory（token 序列）|
|$\pi_\theta$|$\pi$|模型策略（autoregressive LLM policy）|
|$r(y)$|—|reward function（框架中无独立符号，此处保留 $r$）|
|$\theta$|$\theta$|模型参数（一致）|
|$V$|—|词表（vocabulary），框架未覆盖，保留 $V$|
|$z_t$|—|第 $t$ 步的 logit 向量，框架未覆盖，保留 $z_t$|
|$\pi_{\text{ref}}$|—|KL 正则化中的参考策略，保留原符号|

**新概念：**

- 论文中的 Shannon entropy $H(\pi_\theta)$ 作为 standalone reward 使用，可视为一种 **intrinsic reward signal**，建议符号 $r_{\text{int}}$（intrinsic reward），以区别于外部 QA-based reward。

---

### A. QA 依赖度分析

- **Reward 是否 100% 来自 QA？** 否。本文的核心贡献恰恰在于 reward **完全不来自 QA**。EM-RL 使用负熵 $r_{\text{traj}}(y) = \log \pi_\theta(y)$（trajectory-level）或 $r_{\text{tok}}(y) = -\sum_t H(\pi_\theta(\cdot \mid y_{< t}))$（token-level）作为唯一的 reward 信号，不需要任何 labeled data 或 answer verification。
- **去掉 QA 后训练是否崩溃？** 不会。EM-FT 和 EM-RL 的训练流程完全不依赖 QA 问题的标注答案。EM-INF 甚至不需要训练数据或参数更新。
- **是否存在非 QA 奖励成分？** 是——entropy minimization 本身就是一种纯粹的 intrinsic reward，基于模型对自身输出的 confidence，与外部 QA 正确性无关。

> 这是本次 survey 中罕见的完全脱离 QA reward 的训练范式。但需要注意，其评估仍然 100% 依赖 QA（Pass@1 accuracy）。训练信号的非 QA 性与评估指标的 QA 依赖形成了有趣的不对称。

---

### B. Memory 价值盲区

- **能否激励存储"当前无问题、未来可能有用"的信息？** 本文不涉及显式的 memory 存储机制（无 $m_s$ 或 $m_l$ 操作）。Entropy minimization 作用于模型的 token-level 分布，强化的是模型对已有 pretraining knowledge 的 confident retrieval，而非外部记忆的写入/读取。从间接角度看，EM 通过加强模型的 deterministic reasoning，可能更好地利用 pretraining 阶段编码在 $\theta$ 中的隐式记忆。
- **Delete/Update 是否由 QA 驱动？** 不适用——本文没有 memory operation 设计。
- **是否区分不同类型的记忆价值？** 否。论文关注的是 policy 层面的 confidence sharpening，不涉及记忆类型的区分。

> 本文的 "memory" 完全是隐式的，存储在 $\theta$ 中的 pretraining knowledge。EM 的成功暗示：对于 reasoning tasks，模型的隐式记忆可能已经足够，问题在于如何更好地 elicit 而非如何额外存储。这对 learnable memory 研究提出了一个基线性问题：在加入外部 memory 之前，是否已经充分利用了模型的隐式能力？

---

### C. QA 评估的局限性

- **是否有非 QA 评估指标？** 没有。所有评估指标均为 Pass@1 accuracy（math: exact match; coding: test case passing）。论文在 Table 5 中测试了 individualistic value reasoning（IndVal），但仍然使用 accuracy 作为指标。
- **"关键信息存在但不被提问"的测试？** 未涉及。论文的实验设计不涉及这一场景。
- **消融实验是否揭示了 QA 无法捕捉的现象？** 部分是。论文发现 EM 在 IndVal 任务上无效（accuracy 无变化），但在 math/coding 上有效。这表明 QA accuracy 可能无法区分"模型更 confident"和"模型更 correct"——EM 在 IndVal 上同样增强了 confidence，但并未提高 correctness，说明 confidence 与 correctness 的相关性是 task-dependent 的。

> EM 的 failure mode（§5）实际上暴露了 QA 评估的一个盲区：QA accuracy 无法度量 "confidence calibration quality"。一个模型可能在 EM 后变得更自信但在某些任务上并不更正确，QA accuracy 只能看到最终结果而看不到这种 calibration 变化。

---

### D. 非 QA 范式的可能性

- **可脱离 QA 独立评估的组件？** EM 本身（token-level entropy、trajectory-level entropy）是完全可以独立于 QA 测量的指标。论文虽未显式做此评估，但 entropy 的变化量、entropy 与 correctness 的相关性等都可以成为独立评估维度。
- **框架能否接入非 QA reward？** 本文框架天然支持非 QA reward——entropy 本身就是非 QA 的。若将 EM 视为一个 reward design template，可以扩展为其他 intrinsic reward，如 information gain、mutual information、prediction consistency 等。
- **Memory operation 设计是否暗示内在质量标准？** EM 的设计暗示了一个内在质量标准：**低熵 = 高质量**。这是一种与 QA 正确性无关的 quality proxy。论文的 Proposition 1 还区分了 temperature scaling（保持 logit 顺序）与 logit optimization（可改变非 top logit 顺序），暗示"质量"不仅是更集中的分布，还涉及 logit 间的相对关系重组。

> EM 为非 QA reward 设计提供了一个极简但有效的范例。其成功表明：即使不知道"正确答案"是什么，也可以通过 intrinsic signal 改善模型行为。这对 memory 研究的启示是：memory operation 的 reward 是否可以类似地基于 intrinsic signal（如 memory utilization entropy、retrieval confidence）而非下游 QA 表现？

---

### 5. RQ 分析

**Q1: QA reward 是否是 memory 的充分训练信号？**

这篇文章提供了反面证据：EM-RL 在没有任何 QA reward 的条件下，仅使用负熵作为 reward，就在 math 和 coding 任务上达到了与 GRPO/RLOO（使用 60K labeled examples）可比甚至更优的性能。这表明 QA reward 并非 post-training 中唯一有效的训练信号，至少在 reasoning 领域，intrinsic reward（如 entropy）可以是一种充分的替代。但论文也指出 EM 的成功依赖于 pretrained model 的能力足够强（Llama-3.1-8B 上效果较差），因此 intrinsic reward 的充分性是有条件的。

**Q2: QA benchmark 是否是 memory 的充分评估标准？**

本文未直接涉及此问题。

**Q3: 能否构造非 QA 驱动的 memory 训练数据与信号？**

这篇文章为这一问题提供了具体的方法论支撑。EM-FT 和 EM-RL 展示了一种完全非 QA 的训练范式：使用 unlabeled prompts + intrinsic entropy reward。具体的 intrinsic reward 设计包括 trajectory-level entropy $r_{\text{traj}}(y) = \log \pi_\theta(y)$ 和 token-level entropy $r_{\text{tok}}(y) = -\sum_t H(\pi_\theta(\cdot \mid y_{< t}))$。这些信号不依赖于外部标注、answer extraction 或 output verification，可以直接应用于任何生成任务（包括 code generation 等 answer extraction 困难的场景）。对 memory 训练的启示：可以设计类似的 intrinsic reward，如 memory retrieval 的 entropy（低 entropy = 检索结果更集中 = 更好的记忆组织）或 memory utilization rate 等。

---

### Conclusion

我认为从内存角度看，这篇文章证明了一个看似简单的信号（模型自身输出分布的熵）可以在完全无标签的条件下显著提升 LLM 的推理能力。作者提出三种方法：EM-FT 通过直接最小化 token 级熵进行无监督微调；EM-RL 将负熵作为唯一奖励进行强化学习，是一个典型的自监督奖励信号的例子，它不需要外部验证；EM-INF 在推理时优化 logit 以降低熵，无需更新任何参数。从在线的角度来看，EM-INF 是最有价值的组件，它在每个解码步骤中优化 logits，而无需修改 θ 本质上是一种测试时间适应形式。我的观点就是预训练参数 θ 熵最小化可以作为一种长期记忆的形式，它是一种强化机制，可以强化模型已知的知识，而不是注入新的知识。
从 Non-QA Memory 研究视角看，本文的最大价值在于证明了 intrinsic reward（entropy）作为 standalone 训练信号的可行性，为 memory operation 的非 QA reward 设计提供了方法论启示。