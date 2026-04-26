# MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards

**Source:** [arXiv:2601.05488v3](https://arxiv.org/abs/2601.05488) (2026-04-18 修订)

---

## 符号映射表

| 论文原始符号                                                                        | 框架符号                                              | 含义                                 |
| ----------------------------------------------------------------------------- | ------------------------------------------------- | ---------------------------------- |
| $\mathcal{S} = {s_1, \ldots, s_n}$                                            | 输入对话历史（$o_{1:n}$ 的来源）                             | 多轮对话 sessions                      |
| $q$（time $t_q$ 提问）                                                            | $q$                                               | 用户问题                               |
| $\mathcal{M}$（memory bank）                                                    | $m_l$                                             | 长期外部记忆（向量库 + Core block）           |
| $\mathcal{M}_{\text{core}}$                                                   | $m_l^{\text{core}}$（$m_l$ 的子组件，always-in-context） | Core 持久画像（固定容量文本块）                 |
| $\mathcal{M}_{\text{epi}}$                                                    | $m_l^{\text{epi}}$                                | Episodic 时间戳事件条目集合                 |
| $\mathcal{M}_{\text{sem}}$                                                    | $m_l^{\text{sem}}$                                | Semantic 概念/实体条目集合                 |
| $\mathcal{M}_{\text{proc}}$                                                   | $m_l^{\text{proc}}$                               | Procedural 流程/例程条目集合               |
| $\text{State}_\tau = (\mathcal{M}_{\tau-1}, s_\tau)$                          | $C_\tau = (m_l^{(\tau-1)}, o_\tau)$               | session-$\tau$ 时的决策状态              |
| $a_\tau = (a^{\text{core}}, a^{\text{epi}}, a^{\text{sem}}, a^{\text{proc}})$ | $a_\tau$（多组件复合动作）                                 | 一次写入决策（4 个并行 agent）                |
| 检索 top-$k$（text-embedding-3-small）                                            | $v$                                               | 基于语义相似度的检索算子                       |
| Memory operations                                                             | $g_l$ 的离散动作集                                      | 长期记忆更新函数                           |
| $r^{\text{task}}$                                                             | $r$（dense session-level reward）                   | 由 synthetic QA accuracy 计算的奖励      |
| $w^{(m)}$                                                                     | （新增：组件级梯度权重）                                      | contribution-aware gradient weight |
| $\hat{a}_N$ / answer                                                          | $\hat{a}_N$                                       | 最终回答                               |
| $\pi_\theta$（policy = Qwen3-4B）                                               | $\pi_\theta$                                      | 被训练的 memory-construction policy    |

---

## 概览

这篇文章用长对话的 sessions 作为训练轨迹（每个 session 让 policy 同时输出对四个 memory 组件的写入操作），并用 Claude 4.5 Opus 为每个 session 预先合成 5 个 QA 对，作为 session 级别的 dense reward 信号源。  
最后得到了一个被训练的 4B 参数 memory-construction policy（SFT + RL），它会在每个 session 处对 Core/Episodic/Semantic/Procedural 四类记忆并行决定 ADD/UPDATE/MERGE/REPLACE 等操作，把对话压缩进结构化 memory bank 供下游 RAG 使用。
主要是优化 memory-construction 操作的下游 QA 效用。核心有两点：(1) 用合成的 session 级 QA 提供 dense reward（替代以往只在 trajectory 末尾给单一 reward）；(2) contribution-aware gradient weighting，按每个 memory 组件在下游 QA 中被检索的频率，对该组件的 policy gradient 做放大，解决多组件共享同一 reward 时的 credit assignment 问题。

---

## 1. Problem Setting

- **记忆类型**：cross-chat / cross-session 的长期记忆 $m_l$。论文显式构造一个外部 memory bank $\mathcal{M}$，由四个子组件构成（Core / Episodic / Semantic / Procedural），其中 Core 始终注入 context（行为上类似 $m_s$），其余三类通过向量检索召回。
- **决策过程建模**：序贯多步决策。每个 session $\tau$ 上，policy 接收 $\text{State}_\tau = (\mathcal{M}_{\tau-1}, s_\tau)$，输出一组写入操作 $a_\tau$，得到 $\mathcal{M}_\tau$。论文未严格写成 MDP 公式，但训练目标本质上是 session-level 的 contextual policy optimization（每个 session 内独立 sample N rollouts，相邻 session 间通过共享 $\mathcal{M}_{\tau-1}$ 串联）。
- **状态空间 $\mathcal{S}$**：当前 memory bank $\mathcal{M}_{\tau-1}$ + 新到 session $s_\tau$ + top-20 检索到的相关旧 memory。
- **动作空间 $\mathcal{A}$**：四个组件并行的 typed action sets：

$$\mathcal{A}^{\text{core}} = {\text{APPEND}, \text{REPLACE}, \text{REWRITE}}$$

$$\mathcal{A}^{\text{epi}} = {\text{ADD}, \text{UPDATE}, \text{MERGE}}$$

$$\mathcal{A}^{\text{sem}} = {\text{ADD}, \text{UPDATE}, \text{SKIP}}$$

$$\mathcal{A}^{\text{proc}} = {\text{ADD}, \text{UPDATE}}$$

- **观测空间 $\Omega$**：自然语言形式的 session 文本 + 检索到的 memory 条目。
- **记忆数据结构**：
    - Core Memory：单一固定容量（5000 字符）的自然语言文本块，超限时由同一 policy 触发 compress prompt。
    - Episodic / Semantic / Procedural：自然语言条目集合，每条独立 embed 进向量数据库（text-embedding-3-small），通过语义相似度检索 top-$k$。

|组件|论文符号|框架符号|数据结构|
|---|---|---|---|
|多维记忆库|$\mathcal{M}$|$m_l$|Core block + 3 个 vector store|
|Core|$\mathcal{M}_{\text{core}}$|$m_l^{\text{core}}$|单文本块（≤5000 chars）|
|Episodic|$\mathcal{M}_{\text{epi}}$|$m_l^{\text{epi}}$|时间戳条目向量库|
|Semantic|$\mathcal{M}_{\text{sem}}$|$m_l^{\text{sem}}$|实体/概念条目向量库|
|Procedural|$\mathcal{M}_{\text{proc}}$|$m_l^{\text{proc}}$|流程条目向量库|
|检索|top-$k$ embed sim|$v$|余弦相似度 top-20|
|写入函数|memory operations|$g_l$|七种 typed actions|
|Policy|Qwen3-4B-Instruct|$\pi_\theta$|被训练的 LLM 本身|

---

## 2. Training Procedure

- **优化对象**：$\theta$ —— 主 LLM（Qwen3-4B-Instruct-2507）的全部参数。Policy $\pi_\theta$ 同时承担四个 memory agent 的角色（通过 role-specific prompt 切换），因此被训练的是同一份权重。
- **优化算法**：两阶段 —— **SFT + ADRPO**（GRPO 的 contribution-aware 扩展）。
    - **SFT 阶段**：用 Claude 4.5 Sonnet 作为 expert 在 50 个对话（约 2400 sessions × 4 agents = 9600 examples）上生成轨迹，做 imitation learning 解决 cold-start（直接 RL 时 4B 模型频繁输出 malformed JSON）。learning rate $5 \times 10^{-7}$, 10 epochs。
    - **ADRPO 阶段**：从 SFT checkpoint 开始，learning rate $1 \times 10^{-6}$，rollout = 8，clip $\epsilon = 0.2$，5 epochs，32 H20 GPU × 70 hours。
- **训练数据来源**：
    - SFT：50 个 LongMemEval 对话上由 Claude 4.5 Sonnet 滚动生成的 expert trajectories。
    - RL：另 50 个 LongMemEval 对话；每 session 通过 Claude 4.5 Opus 预生成 5 个 QA 对（共 12000 对）作为 dense reward 来源。
- **是否冻结 LLM**：否，主 LLM 全参数微调。
- **核心训练目标**：

ADRPO 在 GRPO 基础上引入组件级权重 $w^{(m)}$：

$$A_i = \frac{r_i - \mu}{\sigma + \epsilon} = \frac{\mathbb{1}[\text{valid}_i] \cdot r_i^{\text{task}} \cdot (1 - \lambda \ell_i) - \mu}{\sigma + \epsilon}$$

$$\mathcal{J}(\theta) = \mathbb{E}\Bigg[\sum_m \frac{1}{\lvert a_i^{(m)} \rvert} \sum_{k=1}^{\lvert a_i^{(m)} \rvert} \min\Big(w^{(m)} \rho_{i,k}^{(m)} A_i,\ w^{(m)} \cdot \text{clip}(\rho_{i,k}^{(m)}, 1-\epsilon, 1+\epsilon) A_i\Big)\Bigg] - \beta \cdot D_{KL}(\pi_\theta \Vert \pi_{\text{ref}})$$

其中 $\rho_{i,k}^{(m)} = \pi_\theta / \pi_{\text{ref}}$ 是组件 $m$ 在 token $k$ 处的 importance ratio，$w^{(m)} = \alpha$ 当 $m$ 是 dominant retrieval contributor 否则为 1（论文最优 $\alpha = 4$）。

---

## 3. Reward Signal

- **奖励类型**：dense step-level reward（每个 session 一次），区别于 Memory-R1 / Mem-α 的 sparse trajectory-level reward。
- **奖励来源**：synthetic session-level QA accuracy。流程：
    1. 训练前为每个 session $\tau$ 用 Claude 4.5 Opus 生成 5 个 QA 对 ${(q_j, \text{ans}_j)}$。
    2. 训练时 sample N=8 rollouts，每个 rollout 产生候选 $\mathcal{M}_\tau^{(i)}$。
    3. GPT-4.1-mini 在 $\mathcal{M}_\tau^{(i)}$ 上回答这 5 个 $q_j$，由 LLM judge（GPT-4.1-mini）对照 $\text{ans}_j$ 判定。
    4. Task reward = 平均 QA accuracy：

$$r^{\text{task}} = \frac{1}{J} \sum_{j=1}^{J} \mathbb{1}[\text{correct}(q_j)]$$

- **奖励分配**：
    - **跨 session**：每个 session 独立计算 reward 并独立做 GRPO advantage（group-relative within rollouts of the same session）。
    - **跨组件（credit assignment）**：四个组件共享同一个 $r^{\text{task}}$，但通过 $w^{(m)}$ 加权梯度做 dynamic credit assignment：

$$d = \arg\max_{m \in {\text{epi}, \text{sem}, \text{proc}}} h^{(m)},\quad w^{(m)} = \begin{cases} \alpha & m = d \ 1 & \text{otherwise} \end{cases}$$

其中 $h^{(m)}$ 是该组件在 5 个 QA 评估中被检索的次数（dominant 组件得到放大）。Core Memory 因总在 context 中故 $w^{(\text{core})} = 1$。

- **辅助奖励/正则项**：
    - **Format validity gate** $\mathbb{1}[\text{valid}]$：JSON 格式不合法、缺字段、未定义动作 → reward 直接置 0。
    - **Length penalty** $\ell \in [0,1]$：分别对 Core 与其它三类设计 piecewise-linear 罚函数（Eq. 7-8），参数 $\lambda = 0.8$, $\theta_{\min}=150$, $\theta_{\max}=400$, $\delta=200$, $[\gamma_l, \gamma_u]=[0.5, 1.3]$。防止 reward hacking。
    - **KL 正则**：$\beta \cdot D_{KL}(\pi_\theta \Vert \pi_{\text{ref}})$。

---

## 4. Inference Procedure

- **记忆初始化**：空（$\mathcal{M}_0 = \emptyset$）。逐 session 顺序处理对话历史，构建 memory bank。
- **每步决策流程**（Build phase）：
    1. 收到 session $s_\tau$ → 用 $s_\tau$ 检索 $\mathcal{M}_{\tau-1}$ 的 top-20 相关条目（context for the policy）。
    2. 同一 LLM 用 4 个 role-specific prompt 并行调用 4 次，每次输出对应 memory 类型的 JSON action（ADD/UPDATE/MERGE/...）。
    3. 应用 actions 得到 $\mathcal{M}_\tau$，新条目 embed 入向量库；Core block 超 5000 字符触发 compression。
- **回答阶段**（Answer phase）：
    - Core Memory 全量注入 context。
    - 对 question $q$ 在 Episodic/Semantic/Procedural 联合向量库中检索 top-10。
    - Answer model（默认 Claude 4.5 Sonnet，可换 GPT-4.1 / Qwen3-4B）基于 Core + retrieved 条目生成 $\hat{a}_N$。
- **额外策略**：top-$k$ 检索（build 阶段 $k=20$，answer 阶段 $k=10$）；Core block 双轮压缩兜底；UPDATE 不删除旧条目而是创建带引用的新条目（保留时间链）。
- **驱动方式**：核心决策（写什么、用什么 action）完全由学到的 $\pi_\theta$ 输出；外围有手工规则（5000 字符上限触发 compress、JSON 格式校验、检索 top-$k$ 数）。

---

## 5. RQ 分析

### RQ1 (What is memory?)

记忆是显式的、外部的、多组件的、自然语言形式的非参数化记忆库 $m_l$，由四类子组件构成（Core / Episodic / Semantic / Procedural），其中 Core 是固定容量文本块、其余三类是向量化条目集合。可被 CRUD 操作，通过语义相似度检索召回 - 主体属于 T1（非参数化外部记忆）。Core Memory 因有固定字符上限且每轮全量注入 context，行为上又具有 T2（固定长度压缩记忆）的性质。

### RQ2 (Which component is optimized?)

被优化的是主 LLM 本身，它同时扮演四个 memory agent 的角色 - O2a（单 LLM 优化）。虽然系统中看起来有 4 个 agent，但它们共享同一份权重（仅靠 prompt 切换角色）。

### RQ3 (Target of Optimization)

主目标是 G1（回答准确度），通过 synthetic session-level QA 的平均 accuracy 作为 task reward。辅助目标包括 G2（效率），length penalty 控制 memory 体积避免冗余 / reward hacking，以及 G6（补充优化目标），format validity gate 与 length penalty 共同构成辅助 shaping 项。

### RQ4 (Training Signal and RL Algorithm)

ADRPO 是 GRPO 的扩展：保留 group-relative advantage normalization 与 clipped surrogate，新增 contribution-aware gradient weighting $w^{(m)}$ 解决多组件 credit assignment - A2（GRPO）。SFT 阶段属于监督预训练，不计入 RL 类别。

### RQ5 (How memory evolves, operates?)

记忆按 session 增量演化：每收到新 session，policy 同时对四个组件做并行写入决策（七种 typed operations）。关键设计是 UPDATE = versioned add 而非 overwrite，新条目带时间戳并显式引用旧条目，使记忆形成可追溯的时间链；MERGE 则做跨 session 的总结性合并。Core Memory 通过 REPLACE 做局部更新而非整体重写。

---

## Conclusion

MemBuilder 是一个针对长对话场景的记忆构造器训练框架。它的目标是：让一个小模型学会把对话切成四类结构化记忆条目（持久画像、时间事件、概念知识、流程知识），并把这些记忆写得"有用"，即下游 QA 系统能用它们答对问题。论文的两个核心创新是：(1) 把原本只在轨迹末尾给一次的奖励，拆成每个 session 一次的密集奖励，做法是预先用 LLM 为每个 session 合成 5 个测验题；(2) 因为四个 memory 组件共享同一个奖励，作者按"哪个组件被实际检索得最多"对该组件的梯度做放大，相当于一种 retrieval-based credit assignment。实验显示 4B 模型经过 SFT+ADRPO 后在 LoCoMo 上达到 84.23%，超过用 Claude 4.5 Sonnet 直接做 memory construction 的 prompting baseline。