# Fine-Mem: Fine-Grained Feedback Alignment for Long-Horizon Memory Management

**Source:** [arXiv:2601.08435v1](https://arxiv.org/abs/2601.08435) (2026-01-13 修订)

---

## 符号映射表

|论文原始符号|框架符号|含义说明|
|---|---|---|
|$C = \lbrace c_1, \ldots, c_T \rbrace$|$\lbrace o_1, \ldots, o_T \rbrace$|流式历史信息块（streaming chunks），可视为按时序到达的观测|
|$c_t$|$o_t$|单个 history chunk（如一段对话）|
|$\pi_\theta$ (Memory Manager)|$g_l$（参数化的长期记忆更新函数） + 独立 manager 模块|可学习的记忆管理器，参数为 $\theta$|
|$\pi_{\text{reason}}$ (Reasoning Agent)|$M$（主模型 / 推理 LLM）|冻结的推理 agent，根据查询和检索结果生成答案|
|$M_t$|$m_l^{(t)}$|第 $t$ 步后的长期记忆状态（一组离散记忆条目）|
|$m_i = \lbrace id_i, \text{content}_i, s_i \rbrace$|$e_i = (\text{key}_i, \text{content}_i, \text{meta}_i)$|单条记忆条目（id, 内容, 写入步数）|
|$P_t$|$a_t$（manager 的 action）|第 $t$ 步生成的一组操作（INSERT/UPDATE/DELETE/SKIP）|
|$\mathcal{T}$|状态转移函数|把操作 $P_t$ 应用到 $m_l^{(t-1)}$ 上得到 $m_l^{(t)}$|
|$q_j$|$q$|下游任务查询|
|$\mathcal{M}_j = \text{Retrieve}(q_j, M_T)$|$v(q, m_l)$|检索算法（本文使用 BM25）|
|$a_j$|$\hat{A}_N$ / $\hat{a}$|推理 agent 的最终答案|
|$r_{\text{global}}$|rollout 级 outcome reward|全局 QA 准确率|
|$r_{\text{chunk}}^{(t)}$|step-level dense reward|chunk 级 QA 准确率（CSR 提供的过程奖励）|
|$r_{\text{EARA}}^{(t)}$|redistributed step reward|经 evidence 归因后分配到第 $t$ 步的奖励|
|$\phi(m)$|来源映射|把 memory item $m$ 反向映射到生成它的步 $t$|
|$N_t$|Normalized Evidence Contribution|第 $t$ 步在所有下游 QA 中的累计证据贡献|
|$\beta$|attribution factor|在 uniform reward 与 evidence-driven reward 间插值|

---

## 概览

这篇文章用了流式历史信息，一个冻结的 reasoning agent，以及一个可训练的小型 manager。它额外构造了两套 QA 数据：(1) 用 GPT-4o-mini 为每个 chunk 自动生成 5 个事实型 chunk-level QA，作为局部过程信号；(2) 原 Mem-α 的 global QA，作为全局结果信号。  
最后得到了一个独立的、可插拔的 memory manager，擅长决定 INSERT / UPDATE / DELETE / SKIP 四种原子操作来维护一个单层平铺的外部记忆库。主 reasoning LLM 不动。  
通过两类细粒度反馈缓解 RL 训练 memory manager 时的 reward sparsity 与 credit assignment 问题：用 chunk-level QA 提供 step-level dense reward；用 evidence-anchored 重分配把 global QA reward 按"哪条记忆被检索利用"反向归因到具体写入步。最终目标仍是下游 QA 的回答准确度（兼顾 memory 长度的压缩效率）。

---

## 1. Problem Setting

- **记忆类型**：跨 session 的长期记忆 $m_l$。论文不维护显式的 $m_s$（每步是无状态决策），但通过流式增量处理实现"长程记忆累积"。
- **决策建模**：MDP（实际上接近 contextual bandit + sequential aggregation）。每步状态 $s_t = (c_t, M_{t-1})$，动作 $a_t = P_t$ 是一个操作列表，转移确定性地由 $\mathcal{T}$ 决定。
- **状态空间 $\mathcal{S}$**：$\lbrace (c_t, M_{t-1}) \rbrace$，即"当前 chunk + 当前已积累的记忆库"。
- **动作空间 $\mathcal{A}$**：四种原子操作 $\lbrace$ INSERT, UPDATE, DELETE, SKIP $\rbrace$ 的有限组合（一步可包含多个操作），以 JSON 函数调用格式输出。
- **观测空间 $\Omega$**：完全可观测（$o_t = s_t$）。
- **记忆数据结构**：**单层** flat 列表，每条 $m_i = \lbrace id_i, \text{content}_i, s_i \rbrace$（id、自然语言内容、写入步数）。论文显式回避多层结构，理由是"多层结构需要精心设计的监督信号否则反而拖累 manager"。
- **检索机制 $v$**：固定的 BM25（论文承认这是一个 limitation，未来计划用 dense / graph retrieval 替换）。

|框架组件|论文实例|
|---|---|
|长期记忆 $m_l$|单层平铺列表 $M_t = \lbrace m_i \rbrace$|
|记忆条目|$\lbrace id, \text{content}, \text{step} \rbrace$|
|写入函数 $g_l$|学习得到的 $\pi_\theta$（Qwen3-4B，输出 INSERT/UPDATE/DELETE/SKIP）|
|检索函数 $v$|BM25（top-k）|
|主模型 $M$|冻结的 Qwen3-32B reasoning agent|
|上下文 $C_N$|$C_N = (q_j, \text{Retrieve}(q_j, M_T))$，无累积|

---

## 2. Training Procedure

- **优化对象**：单一可学习组件——独立的 memory manager $\pi_\theta$（Qwen3-4B / 1.7B / Llama3.2-3B 均验证过）。**主 reasoning LLM 完全冻结**。这是典型的"独立可插拔模块训练"（对应 RQ2 的 O1）。
- **优化算法**：**GRPO**（group-relative 优势归一化），每个 step 采 $G=8$ 个 rollouts。
- **训练数据**：
    - Mem-α 训练语料 + 自动构造的 chunk-level QA（共 30215 条 QA，覆盖 567 个 instances），来自 SQuAD / HotpotQA / PerLTQA / LME-Train / NLU / TREC / PubMed-RCT / BookSum 等多任务混合。
    - QA 由 GPT-4o-mini 生成，由 Qwen3-32B verifier 过滤（必须仅凭对应 chunk 即可正确回答），每 chunk 保留 5 条。
    - 训练时只在 20% 随机抽样的 global QA 上算 reward，已证实与 full set 性能接近。
- **核心目标函数**（论文原始 + 框架符号）：

$$ \mathcal{L}_{\text{policy}} \approx \sum_G \log \rho_\theta(P_t \mid M_{t-1}, c_t) \cdot A_t $$

即 $\sum_G \log \rho_\theta(a_t \mid m_l^{(t-1)}, o_t) \cdot A_t$，其中 $\rho_\theta = \pi_\theta(P_t \mid \cdot) / \pi_{\text{old}}(P_t \mid \cdot)$ 是重要性比，$A_t$ 是 group-relative advantage。

- **总奖励**（这是本文的核心创新点之一）：

$$ r_t = r_{\text{EARA}}^{(t)} + r_{\text{fmt}}^{(t)} + w_1 \cdot r_{\text{chunk}}^{(t)} + w_2 \cdot r_{\text{comp}} $$

权重 $w_1 = 0.5$, $w_2 = 0.05$, $\beta = 0.5$。

- **GRPO 优势归一化**：

$$ A_{t,j} = \frac{r_{t,j} - \mu_{\text{group}}^{(t)}}{\sigma_{\text{group}}^{(t)} + \epsilon} $$

> 避免每次 rollout 时都用 LLM 评估造成的延迟

---

## 3. Reward Signal

本文的 reward 结构是**多层级混合**，是其核心贡献的最直接体现。

- **$r_{\text{global}}$**（rollout 级，sparse）：$M_T$ 完成后，对所有 global QA 求平均准确率： $$r_{\text{global}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}[\pi_{\text{reason}}(\cdot \mid M_T, q_i) = y_i]$$ 来源：EM / SubEM / LLM-as-Judge / Source-based 等指标。
    
- **$r_{\text{chunk}}^{(t)}$**（step 级，dense）：在第 $t$ 步执行 $P_t$ 之后立即用 reasoning agent 在 $M_t$ 上回答**该 chunk 自有的** QA： $$r_{\text{chunk}}^{(t)} = \frac{1}{K} \sum_{j=1}^K \mathbb{I}\big[\pi_{\text{reason}}(\cdot \mid M_t, q_j^{(t)}) = y_j^{(t)}\big]$$ $K = 5$。这是密集过程奖励，避免 sparse terminal reward 的训练困难。
    
- **$r_{\text{EARA}}^{(t)}$**（global 重分配，dense）：把 $r_{\text{global}}$ 通过 evidence 反向归因到具体的写入步： $$r_{\text{EARA}}^{(t)} = (1-\beta) \cdot \frac{r_{\text{global}}}{T} + \beta \cdot N_t$$ 其中 $N_t = \sum_{j=1}^n \sum_{m \in \mathcal{M}_j, \phi(m)=t} \frac{s_j}{\lvert \mathcal{M}_j \rvert \cdot n}$ 是 step $t$ 的"被检索利用次数 × 该次回答正确度"的加权和。论文证明 $\sum_t r_{\text{EARA}}^{(t)} = r_{\text{global}}$，即重分配是 reward-conserving 的。
    
- **辅助项**：
    
    - $r_{\text{fmt}}^{(t)}$：每条操作是否符合 JSON schema 的合法率。
    - $r_{\text{comp}} = 1 - L(M_T) / \sum_t L(c_t)$：终端的压缩率奖励。
- **Credit assignment 机制**：本文最关键的发明就是"用检索追溯"做归因——既不是 uniform 平摊（PPO 风格），也不是纯 group baseline（朴素 GRPO），而是**结合 evidence-based attribution 与 uniform baseline 的软融合**。$\beta = 0.5$ 时性能最佳；$\beta=1$（纯 evidence）会导致信号稀疏并损害 OOD 泛化；$\beta=0$（纯 uniform）则等价于不做归因。
    

---

## 4. Inference Procedure

- **记忆初始化**：$M_0 = \emptyset$（空记忆库）。
    
- **每步流水线**（构造记忆阶段）：
    
    1. 观测当前 chunk $c_t$ 和已有记忆 $M_{t-1}$；
    2. Manager $\pi_\theta$ 基于系统提示词以 JSON 函数调用格式输出操作集合 $P_t$；
    3. 状态转移 $\mathcal{T}$ 执行 $P_t$ 得到 $M_t$；
    4. 重复直至所有 chunk 处理完毕，得到最终记忆 $M_T$。
- **下游 QA 阶段**：
    
    1. 对查询 $q_j$，BM25 在 $M_T$ 上检索 top-k 相关 memory items 得到 $\mathcal{M}_j$；
    2. 冻结的 reasoning agent 基于 $(q_j, \mathcal{M}_j)$ 生成答案 $a_j$。
- **额外推理策略**：
    
    - 检索固定为 BM25 + top-2（baseline 设置；论文未在主实验报告其他 top-k）；
    - reasoning agent temperature 设为 0.1 以保证生成稳定性；
    - manager 与 reasoning agent 都运行在 non-thinking mode；
    - 记忆构造期间 manager 被指示**一次性输出所有操作**（"you will be queried only once, so make sure to call all the memory insertion functions in one turn"）。
- **手工 vs. 学习驱动**：四种原子操作的 schema 是手工定义的，但具体什么时候 INSERT / UPDATE / DELETE / SKIP、写什么内容，完全由学习得到的 $\pi_\theta$ 决定。检索环节是固定的 BM25，未学习。
    

---

## 5. RQ 分析

### RQ1 (What is memory?)

本文的记忆是 T1（非参数化外部记忆），单层 flat 列表，每条为 $\lbrace id, \text{content}, \text{step} \rbrace$ 三元组。是 cross-chat、显式可编辑的长期记忆 $m_l$，由四种原子操作（INSERT/UPDATE/DELETE/SKIP）维护；不维护独立的 $m_s$，每步决策是无状态的。

### RQ2 (Which component is optimized?)

优化对象是 O1（独立可插拔记忆管理器）：一个与主 LLM 解耦的小型 manager，主 reasoning LLM 保持冻结。训练信号是 $r_{\text{global}} + r_{\text{chunk}} + r_{\text{EARA}} + r_{\text{fmt}} + r_{\text{comp}}$ 的混合。

### RQ3 (Target of Optimization)

主目标是 G1（回答准确度），所有 reward 最终都对齐到下游 QA accuracy；同时显式包含 G2（效率）（$r_{\text{comp}}$ 鼓励压缩、降低 memory 长度）和 G6（辅助）（$r_{\text{fmt}}$ 强制 JSON schema 合法）。我认为这篇文章也可视为对 G4（写入决策质量） 的间接优化，chunk-level QA reward 直接评估"这条记忆是否被有效写入"，但这是通过下游 QA 代理，而非独立的 admission label。

### RQ4 (Training Signal and RL Algorithm)

属于 A2（GRPO），标准的 group-relative advantage normalization。论文显式引用 DeepSeekMath 的 GRPO 形式，对长度做归一化处理。

### RQ5 (How memory evolves, operates?)

- 演化方式：流式增量。每个 chunk 到来时，manager 一次性产出一组 $\lbrace$INSERT / UPDATE / DELETE / SKIP$\rbrace$ 操作，确定性地修改 $M_{t-1} \to M_t$。
- 读取（推理时）：BM25 检索 top-k 相关条目供 reasoning agent 使用。
- 写入策略学习：完全由 RL 训练得到，不依赖手工规则。
- 关键创新：通过 $\phi(m)$（"这条 memory item 在哪一步被生成"）追溯，把下游使用反向归因到具体写入步，这是 evidence-anchored attribution 的运行机理。

---

## Conclusion

Fine-Mem 是一个针对 memory manager RL 训练中 reward sparsity 和 credit assignment 两大痛点的精细化解决方案。它在不改动主 reasoning LLM 的前提下，独立训练一个小型 manager，决定流式信息每一片段（chunk）该 INSERT / UPDATE / DELETE / SKIP 进单层平铺记忆库的哪个位置。论文最有意思的两个贡献：(1) chunk-level QA 作为 step-level dense reward - 离线预构造、避免在线 LLM 评估的延迟；(2) Evidence-Anchored Reward Attribution - 用"哪条记忆在下游被检索利用"这一行为信号，把 sparse global reward 反向归因到具体写入步，实现 reward-conserving 的细粒度信用分配。在 Memalpha（ID）和 MemoryAgentBench（OOD）上分别取得 4.4% 和 7.2% 的平均提升，且在多种 manager backbone（Qwen3 / Llama3.2）和 reasoning model（Qwen3-32B / GPT-4o-mini）下保持稳健。