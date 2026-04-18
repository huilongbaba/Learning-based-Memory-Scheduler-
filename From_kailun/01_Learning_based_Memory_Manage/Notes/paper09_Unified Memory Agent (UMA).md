# Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning (UMA)

**Source:** [https://arxiv.org/abs/2602.18493](https://arxiv.org/abs/2602.18493) (2026-02-13)

---

## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$\mathcal{M}_t = (m^{\text{core}}, \mathcal{B})$|$m_l$|双组件长期记忆：core summary + Memory Bank|
|$m^{\text{core}}$|$m_l^{\text{core}}$|始终在 context 中的核心摘要|
|$\mathcal{B}$|$m_l^{\text{sem}}$|结构化 KV 条目集合（Memory Bank）|
|$x_t$|$q$ / chunk input|当前输入焦点（Phase I 为 chunk $c_i$，Phase II 为 query $q$）|
|$h_t$|$m_s$|即时交互历史（短期 working memory）|
|$C = \lbrace c_1, \dots, c_n \rbrace$|输入流|长文档分块后的 chunk 序列|
|CRUD actions|$a$|create / update / delete / reorganize 记忆操作|
|$r_{\text{tool}}$|工具执行反馈|即时的工具调用正确性奖励|
|$r_{\text{outcome}}$|最终回答奖励|QA 阶段的 EM/F1 正确性奖励|
|Future Utility Signal|记忆步奖励|从后续 QA reward 反向传播到记忆操作步的信号|
|Task-Stratified GRPO|优化算法|按任务类型分层归一化 advantage 的 GRPO 变体|
|Qwen3-4B-Instruct|$\pi_\theta$|策略模型（主 LLM）|

---

## 概览

这篇文章将长文档分块为 chunk 序列，agent 在流式处理每个 chunk 时通过 CRUD 操作主动维护一个双组件记忆（core summary + Memory Bank），并在最终 QA 阶段利用该记忆回答问题。整个过程产生的 trajectory 包含交替的记忆操作步和 QA 步。    
最后得到一个端到端训练的统一策略 $\pi_\theta$，既负责记忆的主动维护，也负责基于记忆的问答。做了Memory-R1没做的事情。  
优化最终 QA 回答的准确度 G1，同时通过 Task-Stratified GRPO 将 QA 的 outcome reward 反向传播到记忆操作步，实现记忆维护与下游问答的联合优化。

---

## 1. Problem Setting

- **记忆类型**：in-chat memory（单次长上下文交互内的记忆），同时覆盖 $m_s$（短期交互历史 $h_t$，周期性清空）和 $m_l$（持久化的双组件记忆 $\mathcal{M}_t$）。
- **决策过程建模**：显式建模为 **MDP**。状态包括当前记忆 $\mathcal{M}_t$、当前输入焦点 $x_t$、交互历史 $h_t$。
- **状态空间 $\mathcal{S}$**：$s_t = (\mathcal{M}_t, x_t, h_t)$，其中 $\mathcal{M}_t = (m^{\text{core}}, \mathcal{B})$。
- **动作空间 $\mathcal{A}$**：统一为单一动作空间，按阶段划分——Phase I 包含 CRUD 操作（create / update / delete / reorganize），Phase II 包含检索操作（BM25/Embedding retrieval）和最终回答生成。
- **观测空间 $\Omega$**：等同于状态空间（完全可观测 MDP），每步可看到当前 chunk/query + 记忆 + 历史。
- **记忆数据结构**：双组件结构：
    - $m^{\text{core}}$：紧凑的全局摘要文本（始终在 context 中）
    - $\mathcal{B}$：结构化 KV 条目集合，支持显式 CRUD 操作

|核心组件|框架符号|描述|
|---|---|---|
|Core Summary|$m_l^{\text{core}}$|全局上下文摘要，始终在 prompt 中|
|Memory Bank|$m_l^{\text{sem}}$|KV 条目集合，支持 CRUD|
|Interaction History|$m_s$|近期交互缓冲，周期性清空|
|Chunk Stream|输入流|文档分块序列 $C = \lbrace c_1, \dots, c_n \rbrace$|
|Unified Policy|$\pi_\theta$|同时驱动 Phase I 和 Phase II|

> UMA 的创新在于将记忆维护（Phase I）和问答（Phase II）统一在同一个 policy 中端到端训练，而非像 Memory-R1 那样使用两个独立 agent。这样记忆操作能直接受到下游 QA 结果的指导。

---

## 2. Training Procedure

- **优化组件**：主 LLM 本身 $\pi_\theta$（Qwen3-4B-Instruct）。单一模型同时学习记忆管理和问答。
- **优化算法**：**Task-Stratified GRPO**——GRPO 的变体，将 trajectory 中的步骤按任务类型（Memory 步 vs. QA 步）分层，在各组内独立归一化 advantage，避免异质 reward 的混淆。
- **训练数据来源**：离线构造的复合数据集，包含三部分：
    1. HotpotQA 衍生的多跳检索数据
    2. MemAlpha 通用长上下文数据
    3. Ledger-QA 合成状态追踪数据
- **是否冻结 LLM 参数**：否，主 LLM 的全部参数参与优化。
- **核心训练目标**：

GRPO 目标函数（框架符号标注）：

$$\mathcal{L}(\theta) = -\mathbb{E}_{q \sim \mathcal{D}} \left[ \frac{1}{G} \sum_{g=1}^{G} \frac{1}{\lvert T^{(g)} \rvert} \sum_{t} \min\left( \rho_t^{(g)} \hat{A}_t^{(g)}, \text{clip}(\rho_t^{(g)}, 1-\epsilon, 1+\epsilon) \hat{A}_t^{(g)} \right) \right]$$

其中 advantage 按任务类型分层计算（Task-Stratified）：

$$\hat{A}_{\text{mem}}^{(g)} = \frac{r_{\text{mem}}^{(g)} - \mu_{\text{mem}}}{\sigma_{\text{mem}}}$$

$$\hat{A}_{\text{qa}}^{(g)} = \frac{r_{\text{qa}}^{(g)} - \mu_{\text{qa}}}{\sigma_{\text{qa}}}$$

记忆步的 reward 包含 Future Utility Signal（从后续 QA reward 反向传播）：

$$r_{\text{mem}}^{(g)} = r_{\text{tool}}^{(g)} + \text{FutureUtility}(r_{\text{qa}})$$

---

## 3. Reward Signal

- **奖励类型**：混合型——QA 步使用 sparse terminal reward（最终回答正确性），记忆步使用半密集信号（工具执行反馈 + Future Utility Signal）。
- **奖励来源**：
    - $r_{\text{tool}}$：工具调用的即时执行反馈（CRUD 操作是否格式正确、执行成功）
    - $r_{\text{outcome}}$：基于 EM（Exact Match）的 QA 正确性评分
- **奖励分配机制**：
    - QA 步：直接使用 $r_{\text{outcome}}$
    - 记忆步：通过 **Future Utility Signal** 将后续 QA 步的 reward 反向传播到记忆操作步，实现长程 credit assignment
    - 两类步骤在各自的 group 内独立归一化 advantage（Task-Stratified）
- **辅助奖励/正则项**：$r_{\text{tool}}$ 作为辅助信号提供即时反馈；KL 散度正则项约束策略偏移。

---

## 4. Inference Procedure

- **记忆初始化**：$m^{\text{core}}$ 初始化为空文本，$\mathcal{B}$ 初始化为空集合。
- **每步决策流程**：
    1. **Phase I（Sequential Memory Maintenance）**：对每个 chunk $c_i$，agent 观测当前 chunk + 现有记忆 → 决定执行 CRUD 操作（create 新条目 / update 已有条目 / delete 过时条目 / reorganize 重组）→ 更新 $\mathcal{M}_t$ → 处理下一个 chunk
    2. **Phase II（Hybrid Retrieval-Augmented QA）**：处理完所有 chunk 后，输入焦点切换为 query $q$ → agent 同时使用结构化检索（从 Memory Bank）和原始上下文检索（BM25/Embedding）→ 综合证据生成最终回答
- **推理时额外策略**：混合检索策略（structured retrieval from Memory Bank + raw-context retrieval via BM25/Embedding），确保信息互补。
- **策略驱动方式**：完全由学习得到的 $\pi_\theta$ 驱动，Phase I 的 CRUD 决策和 Phase II 的检索/回答策略均由同一策略网络输出，无手工规则。

---

## 5. RQ 分析

### RQ1 (What is memory?)

UMA 使用 T2：$\mathcal{M}_t = (m^{\text{core}}, \mathcal{B})$，包含一个始终在 context 中的 core summary（类似 Mem-$\alpha$ 的 $m_l^{\text{core}}$）和一个支持 CRUD 的结构化 KV Memory Bank（类似 $m_l^{\text{sem}}$）。同时使用 $m_s$（交互历史 $h_t$）作为短期记忆。这是显式记忆的多组件表征。

### RQ2 (Which component is optimized? Which signal is used?)

UMA 优化 O2 主 LLM 本身：直接对 Qwen3-4B-Instruct 的全部权重 $\theta$ 进行端到端 RL 训练（Task-Stratified GRPO）。单一模型同时负责记忆维护和问答，不使用独立辅助模型。

### RQ3 (Target of optimization)

UMA 的优化目标为 G1 回答准确度：以 QA 的 EM 正确性作为主要 reward signal。同时通过 Future Utility Signal 将 QA reward 传播到记忆步，但最终优化目标仍是下游回答的准确性。

### RQ4 (How memory evolves, operates?)

记忆在 Phase I 中随 chunk 流式处理而持续演化：每处理一个 chunk，agent 可执行 CRUD 操作主动更新 Memory Bank 和 core summary。Phase II 中记忆状态冻结，仅用于检索。记忆的读写策略完全由 RL 训练得到的 $\pi_\theta$ 驱动，无手工规则。

---

## Conclusion

UMA（Unified Memory Agent）提出了一个端到端的强化学习框架，将记忆的主动维护与下游问答统一在同一个 LLM 策略中联合优化。它维护一个双组件记忆结构（core summary + Memory Bank），在流式处理长文档时通过 CRUD 操作主动整理信息，而非被动依赖查询时的检索。训练时使用 Task-Stratified GRPO 解决记忆步与 QA 步之间异质 reward 的 credit assignment 问题，通过 Future Utility Signal 让记忆操作的质量直接受到下游回答准确性的指导。实验表明，UMA 在动态状态追踪任务上大幅超越长上下文和 RAG baseline，同时在标准检索基准上保持竞争力，证明了"主动的、学习到的状态管理"相比"被动检索"的优越性。