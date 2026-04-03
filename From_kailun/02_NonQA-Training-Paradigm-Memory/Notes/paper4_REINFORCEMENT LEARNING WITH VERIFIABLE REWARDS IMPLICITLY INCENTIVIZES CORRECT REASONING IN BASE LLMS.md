# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs

**Source:** [arXiv:2506.14245v2](https://arxiv.org/abs/2506.14245) (2025-06, revised 2025-10)

---

## 符号映射表

| 论文原始符号         | 框架符号                 | 说明                                                                                 |
| -------------- | -------------------- | ---------------------------------------------------------------------------------- |
| $q$            | $q$                  | question / prompt                                                                  |
| $\pi_\theta$   | $\pi$                | 模型策略，参数为 $\theta$                                                                  |
| $\theta$       | $\theta$             | 模型参数                                                                               |
| $y_i$          | $T$ (trajectory)     | 第 $i$ 条采样响应（包含 CoT + answer）                                                       |
| $c_i$          | $t$ (thought)        | 响应 $y_i$ 中的 CoT 推理链                                                                |
| $a_i$          | $A_n$ (final answer) | 响应 $y_i$ 中的最终答案                                                                    |
| $R(y_i)$       | —                    | 基于答案正确性的二元奖励                                                                       |
| $\hat{A}(y_i)$ | —                    | GRPO 标准化优势值                                                                        |
| $G$            | —                    | 每个 prompt 的采样组大小                                                                   |
| $p_c^\theta$   | —                    | 当前策略生成正确 CoT 的概率                                                                   |
| $\alpha$       | —                    | 正确 CoT 产生正确答案的条件概率：$P(\mathcal{I}_{\text{Ans}}=1 \mid \mathcal{I}_{\text{CoT}}=1)$ |
| $\beta$        | —                    | 错误 CoT 产生正确答案的条件概率：$P(\mathcal{I}_{\text{Ans}}=1 \mid \mathcal{I}_{\text{CoT}}=0)$ |

**新概念：**

- **CoT-Pass@K**：论文提出的新评估指标，要求答案和推理过程均正确才算通过。可记为 $\text{CoT-Pass@K}$，本质是对 $t$（thought）与 $A_n$（final answer）做联合正确性验证。
- **Logic Prior assumption**：预训练 LLM 已建立的知识与逻辑先验，使得正确 CoT 比错误 CoT 更有可能导向正确答案（即 $\alpha > \beta$）。可记为 $\mathcal{P}_{\text{logic}}$。

---

### A. QA 依赖度分析

本文的 reward **100% 来自 QA 结果**。具体而言，reward 定义为：

$$R(y_i) = \mathcal{I}_{\text{Ans}}(a_i) \in \lbrace 0, 1 \rbrace$$

即完全由最终答案是否匹配 ground truth 决定。GRPO 优势通过组内标准化计算：

$$\hat{A}(y_i) = \frac{R(y_i) - \mu_Y}{\sigma_Y}$$

去掉 QA 问题后，训练流程**完全崩溃**——没有 ground truth 就无法计算 $R(y_i)$，也无法得到有效梯度。

然而，论文的核心理论贡献恰恰揭示了一个**隐含的非 QA 奖励成分**：Theorem 1 证明在 Logic Prior 假设下，GRPO 的梯度会**隐式地**对正确 CoT 赋予正优势、对错误 CoT 赋予负优势，即使 reward 仅基于答案正确性。这意味着 answer-correctness reward 中**蕴含了对推理过程质量的隐式激励信号**，条件是预训练已经建立了足够强的 $\alpha - \beta$ gap。

> 这里的"隐式激励"本质上依赖预训练建立的 Logic Prior（$\alpha > \beta$）。如果预训练质量不足导致 $\alpha \leq \beta$，则 GRPO 可能反向强化错误 CoT。论文自身也承认这是 R1-Zero 可读性差和语言混合问题的根源。

---

### B. Memory 价值盲区

- **能否激励存储"当前无对应问题、未来可能有用"的信息？** 不能。RLVR 的训练循环是：给定 $q$ → 采样 $G$ 条响应 → 用 answer correctness 打分 → 更新 $\pi_\theta$。每一轮梯度更新都绑定于具体的 $(q, \text{ground truth})$ 对。不存在任何机制鼓励模型存储"暂时无用但未来可能有用"的信息。
    
- **Delete/Update 是否由 QA 表现驱动？** 本文不涉及显式的 memory operation（无 store/retrieve/delete/update 操作）。模型的"记忆"完全体现在参数 $\theta$ 的隐式变化中，而这种变化 100% 由 QA reward 驱动。
    
- **是否区分不同类型的记忆价值？** 未区分。论文只区分了 CoT 正确性（$\mathcal{I}_{\text{CoT}}$）和答案正确性（$\mathcal{I}_{\text{Ans}}$），不涉及 informational / procedural / contextual 等记忆类型的分类。
    

> 本文的 memory 完全是参数化的隐式记忆（$\theta$ 的变化），没有任何外部记忆 $m_l$ 或工作记忆 $m_s$ 的显式操作。从 Non-QA Memory 视角看，RLVR 框架中模型学到的推理模式可视为一种 procedural memory 的隐式改进，但论文未从这个角度分析。

---

### C. QA 评估的局限性

- **是否有非 QA 指标？** 有。论文提出了 **CoT-Pass@K**，这是对标准 Pass@K 的显著改进。CoT-Pass@K 要求 CoT 推理过程和最终答案同时正确才计为通过，使用 LLM-as-a-CoT-Judge（DeepSeek-R1-0528-Qwen3-8B）进行 CoT 正确性验证。此外还引入了 $P(CC \mid CA)^{(q)}$（正确答案中包含正确 CoT 的比例）作为训练过程中的诊断指标。这些虽然超越了纯 answer-correctness 评估，但**本质上仍服务于 QA 任务**——评估的是"能否正确推理出答案"，而非记忆的独立价值。
    
- **如果构造"关键信息存在但从不被提问"的测试？** 该方法无法应对此类测试。RLVR 的优化目标完全是提高 answer pass rate，不会激励模型保留未被查询的信息。
    
- **消融实验是否揭示了 QA 指标无法捕捉的现象？** 是的，论文的核心发现之一就是：**Pass@K 指标无法区分"正确推理得到答案"和"碰巧猜对答案"**。在 AIME 等难题上，base LLM 的 Pass@K 可以追上甚至超过 RLVR 模型，但 CoT-Pass@K 揭示了持续存在的显著差距。此外，Figure 4 显示即使 $P(CA)^{(q)}$ 接近 1.0，$P(CC \mid CA)^{(q)}$ 的中位数仍然只有约 0.7，说明 answer-correctness reward 无法完全消除错误推理。
    

> CoT-Pass@K 的提出本身就是对"QA 评估局限性"的有力证据。但它依赖 LLM verifier（DeepSeek-R1-0528-Qwen3-8B），而该 verifier 本身的可靠性有限（论文承认存在 false positive/negative），这构成了一个递归的验证问题。

---

### D. 非 QA 范式的可能性

- **可脱离 QA 独立评估的组件？** CoT-Pass@K 和 $P(CC \mid CA)$ 可以部分脱离 answer correctness 独立评估推理质量。特别是 LLM-as-a-CoT-Judge 范式可以独立判断 CoT 的逻辑正确性，不依赖最终答案。此外，Section 6 用 SFT 泛化性能作为 CoT 质量的代理指标，也是一种间接的非 QA 评估方式。
    
- **框架能否接入非 QA reward？** 理论上可以。Theorem 1 的核心只要求 reward 能区分"好"和"坏"的响应。如果将 $R(y_i)$ 替换为基于 CoT 过程质量的 reward（如 process reward model），框架在数学上仍然成立。论文在 Discussion 中提到了 process reward modeling (Lightman et al., 2024; Uesato et al., 2022; Wang et al., 2024) 作为可能的方向。但论文本身**未实验任何非 QA reward**。
    
- **Memory operation 设计是否暗示内在质量标准？** 不存在显式的 memory operation。但 Logic Prior 假设（$\alpha > \beta$）暗示了一个内在质量标准：预训练建立的知识和逻辑先验本身就是一种"好推理"的内在度量。GRPO 利用这个先验将 answer-correctness 信号分解为对 CoT 质量的隐式反馈。
    

> 论文展望中提到的 "new algorithmic paradigms" 和 "directly incentivize correct reasoning paths" 实际上指向了 process reward 或 intrinsic reward 的方向，但作者选择留在 outcome-based reward 的框架内。这为 Non-QA reward 的研究留下了明确的开放空间。

---

### 5. RQ 分析

**Q1: QA reward 是否是 memory 的充分训练信号？**

论文的立场是：QA reward 对推理训练是有效的（在 Logic Prior 成立时），但不是充分的。Theorem 1 证明 answer-correctness reward 可以隐式激励正确推理，但 Figure 4 揭示了明确的局限：即使训练问题的 $P(CA)^{(q)}$ 接近 1.0，仍有约 30% 的正确答案伴随错误 CoT。论文明确承认 "we may not have a chance to mitigate them purely based on answer correctness as the reward"。从 credit assignment 角度看，answer → CoT 的信号路径确实很长，论文通过 $\alpha - \beta$ gap 的存在性来缓解这一问题，但未讨论该 gap 在更复杂任务中是否足够。

**Q2: QA benchmark 是否是 memory 的充分评估标准？**

我认为论文隐式否定了这一点。CoT-Pass@K 的提出直接表明标准 QA 指标（Pass@K）不够充分，它无法区分"真正理解"和"碰巧猜对"。论文发现在 AIME 难题上，base LLM 可以通过多次采样"猜对"答案，使得 Pass@K 虚高。但论文的替代方案（CoT-Pass@K）仍然在 QA 框架内——它评估的是"推理 + 回答是否正确"，而非"是否记住了应该记的信息"。对于"该记但没被问"的评估能力，本文无任何贡献。

**Q3: 能否构造非 QA 驱动的 memory 训练数据与信号？**

本文未直接涉及此问题。 论文的全部训练数据和信号均为 QA 驱动（17k 数学问题 + answer correctness reward）。但论文在 Discussion (A.7) 中提供了间接相关的展望：呼吁 process reward modeling、新的 RLVR 算法、以及"更直接激励正确推理路径"的方法。这些方向与 intrinsic reward（如 information gain, counterfactual contribution）在精神上是一致的，但论文未具体讨论这些概念。Section 6 的 SFT 实验也暗示了一种可能性：用 RLVR 产生的高质量 CoT 数据作为非 RL 的训练信号，但这仍然依赖 QA 作为上游数据生成的驱动。

---

### Conclusion

这篇文章系统性地回应了"RLVR 是否真正提升了 LLM 推理能力"的争论。作者首先通过引入 CoT-Pass@K 指标（同时检验答案和推理过程的正确性），揭示了此前 Pass@K 实验中被忽略的"幸运猜中"现象，并在数学和代码两个领域展示了 RLVR 训练后推理能力边界的实质性扩展。在理论层面，作者证明了在"Logic Prior"假设（正确 CoT 比错误 CoT 更可能导向正确答案）下，GRPO 算法会隐式地给正确推理赋予正优势、给错误推理赋予负优势，从而即使 reward 仅基于答案正确性，也能逐步提升 CoT 的质量。训练动态分析和 SFT 复制实验进一步验证了这一结论：RLVR 从训练早期就开始改善推理质量，且其产生的 CoT 数据质量远高于 base LLM 的 CoT，甚至可以通过简单的 SFT 复制出接近 RLVR 模型的性能。