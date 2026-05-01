# Memory-T1: Reinforcement Learning for Temporal Reasoning in Multi-Session Agents

**Source:** [https://openreview.net/pdf?id=vQf2YR2Kpd](https://openreview.net/pdf?id=vQf2YR2Kpd) (2026-04-11 修订，Accepted at ICLR 2026 poster，8644)

---
## 符号映射表

|论文原始符号|含义|框架统一符号|
|---|---|---|
|$q$|用户查询/问题|$q$|
|$a$ / $a^*$|生成答案 / ground-truth 答案|$a$ / $a^*$（$a$ 即 $A_N$）|
|$\mathcal{M} = [(\tau_1, S_1), \ldots, (\tau_N, S_N)]$|多会话对话历史构成的 memory bank（带时间戳的 sessions）|$m_l$（长期记忆，外部非参数化条目集合）|
|$S_i$ / $u_{ij}$ / $E_{ij}$|会话 $i$ / 第 $j$ 条 utterance / 该 utterance 中提及的事件集合|$m_l$ 中的离散条目（含 $\text{content}$、$\text{meta}=\tau_i$、$\text{struct}=$ event annotations）|
|$\mathcal{C}$ (candidate pool)|经过粗筛后的候选会话集合（top-k）|$C_N$ 的前置候选集；可视为 $m_l$ 经检索 $v$ 投影到 working set 的中间产物|
|$\mathcal{S} \subseteq \mathcal{C}$|RL agent 最终精选出的 evidence 会话子集|最终用于回答的 retrieved context（$C_N$ 的核心来源）|
|$\mathcal{M}^*$|gold-standard 证据会话集合（训练时可见，推理时不可见）|ground-truth $C^*$（用于 reward 计算）|
|$I_Q = (t_{\text{start}}, t_{\text{end}})$|查询的目标时间范围|gold temporal scope（reward 信号源）|
|$\pi_\theta$|待训练的策略模型|$\pi$|
|$\pi_{\text{ref}}$|冻结的参考策略|$\pi_{\text{ref}}$|
|Retriever (BM25) + Time Filter|候选生成阶段的检索机制|$v$（检索算法 $v(q, m_l) \to \text{top-}k$）|
|GRPO objective $\mathcal{J}_{GRPO}(\theta)$|训练目标|RL training objective|
|$R = w_a R_a + w_g R_g + w_t R_t$|多级奖励函数|reward 设计|

---

## 概览

这篇文章用了带时间戳的多会话对话历史（Time-Dialog 数据集），并在训练阶段额外标注了三类信号：每个 query 的目标时间范围 $I_Q$、每条 utterance 的事件级时间标签、以及每个 query 的 gold evidence session IDs $\mathcal{M}^*$。这些精细标注仅用于训练时计算 reward，推理时不可见。  
最后得到了一个经过 RL 微调的 7B/3B 主 LLM 策略 $\pi_\theta$，它在给定 query $q$ 与候选池 $\mathcal{C}$ 后能端到端地输出"被引证据 session id 列表 + 自然语言答案"的结构化字符串。系统整体是一个两阶段的 coarse-to-fine pipeline（时间过滤 + BM25 检索做粗筛 - RL 主模型做精选与回答）。  
优化三件事：(1) 答案准确度 $R_a$（G1）、(2) 证据 grounding $R_g$（Jaccard 与 gold session 集合的对齐）（G4）、(3) 时间一致性 $R_t$（会话级时间邻近 $R_s$ + utterance 级事件时间保真 $R_f$）（G6）。

---

## 1. Problem Setting

- **记忆类型**：Cross-chat / cross-session 的长期记忆 $m_l$。论文处理的对象是过去多个 session 累积下来的、带时间戳的对话历史（memory bank）。这与短期 in-chat 工作记忆 $m_s$ 不同。
- **决策建模**：把"从 $\mathcal{C}$ 中精选证据子集 $\mathcal{S}$ + 生成答案 $a$"建模为一个 **single-turn 的 contextual bandit / 单步生成任务**，使用 GRPO 优化。严格来说不是经典的多步 MDP——agent 在一个 step 内同时输出 selected_memory 和 answer。
- **状态空间** $\mathcal{S}_{\text{state}}$：$(q, \mathcal{C})$，即 query 与候选会话池构成的 prompt。
- **动作空间** $\mathcal{A}$：策略一次性产出的结构化字符串 `{selected_memory: [session_i, ...], answer: "..."}`，可同时解析出 evidence subset $\mathcal{S} \subseteq \mathcal{C}$ 和最终答案 $a$。
- **观测空间** $\Omega$：候选池 $\mathcal{C}$ 的内容（每个 session 的文本 + 其时间戳 $\tau_i$）。
- **记忆数据结构**：自然语言条目 + 结构化时间元数据。每个 session $S_i = {(u_{ij}, E_{ij})}$，每个事件 $e_k$ 可附带语义描述 $\kappa_k$ 和时间跨度 $(t_k^{\text{start}}, t_k^{\text{end}})$。整体上是 **flat 的 session 列表 + per-session timestamp + per-utterance event annotations**（训练时才能用到 event annotations）。

|框架组件|对应论文实体|
|---|---|
|$m_l$|Memory bank $\mathcal{M}$（多会话对话历史）|
|$m_s$|论文未显式建模|
|$v$|Time Filter + BM25 检索（粗筛产生 $\mathcal{C}$）|
|$\pi$|$\pi_\theta$（Qwen2.5-3B/7B-Instruct）|
|$C_N$|$(q, \mathcal{C})$ 拼接后的 prompt|
|$A_N$|$a$（结构化输出中的 answer 字段）|

> 证据选择与答案生成耦合在一次生成里联合优化。

---

## 2. Training Procedure

- **优化对象**：主 LLM $\pi_\theta$ 的全部参数（$\theta$）—— 直接对 Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct 做端到端 RL 微调。LLM **不冻结**。检索器（BM25）和 time filter 不参与训练。
- **优化算法**：**GRPO**（Group Relative Policy Optimization），用 batch-average baseline 替代 value network。
- **训练数据**：Time-Dialog 数据集（4,065 训练 / 451 验证 / 200 测试），每条样本含 $(q, \mathcal{M}_{-}, a_{-}, I_{Q})$。属于离线轨迹 + GPT-4 辅助标注 + 人工校验的混合数据。
- **冻结哪些参数**：Reference policy $\pi_{\text{ref}}$ 冻结（用于 KL 正则）；BM25 检索器无参数；time filter 由 prompted LLM 完成（也不训练）。**$\pi_\theta$ 全参数更新**。
- **核心训练目标函数**：

GRPO objective（论文 Eq. 3）：

$$ \max_\theta ; \mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{(q,\mathcal{C}) \sim \mathcal{D},, {(\mathcal{S}_j, a_j)} \sim \pi_{\text{ref}}} \Big[ \tfrac{1}{G} \sum_{j=1}^{G} \min\big( r_j(\theta) \hat{A}_j,; \text{clip}(r_j(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_j \big) \Big] - \beta , \mathbb{E}_{(q,\mathcal{C}) \sim \mathcal{D}} \big[ D_{KL}(\pi_\theta(\cdot \mid (q,\mathcal{C})) ,\Vert, \pi_{\text{ref}}(\cdot \mid (q,\mathcal{C}))) \big]. $$

其中重要性比率：

$$r_j(\theta) = \frac{\pi_\theta((\mathcal{S}_j, a_j) \mid (q, \mathcal{C}))}{\pi_{\text{ref}}((\mathcal{S}_j, a_j) \mid (q, \mathcal{C}))}.$$

GRPO 的 advantage 估计（论文 Eq. 4）：

$$\hat{A}((q,\mathcal{C}), (\mathcal{S}_j, a_j)) = R((q,\mathcal{C}), (\mathcal{S}_j, a_j)) - \tfrac{1}{G} \sum_{j=1}^{G} R((q,\mathcal{C}), (\mathcal{S}_j, a_j)).$$

统一符号：策略 $\pi$（即 $\pi_\theta$）的更新由 group-relative advantage $\hat{A}_j$ 驱动，policy gradient 在重要性比率与 KL 约束下做 clip。

- **超参**：batch size = 32, lr = $1 \times 10^{-6}$, $K=G=8$ rollouts per prompt, KL coefficient = 0.1, max sequence length = 16k tokens, framework = VERL。

---

## 3. Reward Signal

- **奖励类型**：**dense step-level reward**（实际是每个完整 rollout 的 outcome reward，但 reward 本身由三个稠密信号合成，远比纯 sparse answer reward 信息更丰富）。整个 reward 范围 $R \in [-1, 1]$，parsing 失败时直接给 $-0.5$。
- **奖励来源**：
    - $R_a$（accuracy）：与 $a^*$ 比对，根据答案类型用不同 metric（EM / Unit-aware / $\epsilon$-EM / Hamming）。
    - $R_g$（evidence grounding）：与 gold session 集合 $\mathcal{M}^*$ 的 **Jaccard index**，scale 到 $[-1, 1]$。
    - $R_t$（temporal consistency）：分两层
        - $R_s$（session-level chronological proximity）：基于选中 session 时间戳与 $I_Q$ 的最小时间间隔 gap，套一个 logistic 软惩罚函数：

$$R_s = \frac{c}{1 + \exp(x)} - d, \quad x = \frac{\text{gap}(U, I_Q) - m}{s}, \quad R_s \in (-d, c-d].$$

```
- $R_f$（utterance-level chronological fidelity）：对每条相关 utterance 内每个事件 $e$ 打分（$+1$ 完全在 $I_Q$ 内 / $+0.5$ 部分重叠 / $-1$ 无重叠），再做两层平均：
```

$$R_f(U, I_Q) = \frac{1}{|,U_{\text{rel}},|} \sum_{u \in U_{\text{rel}}} \Big( \frac{1}{|,E_u,|} \sum_{e \in E_u} r_e(e, I_Q) \Big).$$

- 合成：$R_t = \alpha R_s + \beta R_f$（论文 $\alpha = \beta = 0.5$）。
- **总奖励**：

$$R = \begin{cases} w_a R_a + w_g R_g + w_t R_t, & \text{parsing succeeds} \ -0.5, & \text{otherwise}\end{cases}, \quad R \in [-1, 1].$$

论文最优权重 $(w_a, w_g, w_t) = (0.6, 0.2, 0.2)$。

- **Credit assignment**：reward 是对一个完整 rollout（即一个 $(\mathcal{S}_j, a_j)$ pair）整体打分，不做 token 级 credit assignment；GRPO 通过 group-relative baseline 把这个 outcome reward 转成每个 rollout 的相对 advantage。
- **辅助/正则项**：
    - parsing 失败的 $-0.5$ 罚分（structural/format reward）；
    - KL 散度对 $\pi_{\text{ref}}$ 的正则（训练稳定性）；
    - 论文的消融显示：去掉 $R_t$ 整体掉 5.1%、去掉 $R_g$ 掉 9.1%、只用 $R_a$ 掉 22.4%——证明三个 reward 是真正互补的。

---

## 4. Inference Procedure

- **记忆初始化**：直接以完整的 memory bank $\mathcal{M}$（多会话对话历史）作为输入。**$\mathcal{M}^*$、$I_Q$、event annotations 在推理时全部不可见**——这些只用来训练时算 reward。
- **每步决策流水线**（实际只有一"步"，但内部分两阶段）：
    1. **Phase 1 - Candidate Generation（粗筛）**：
        - **Time Filtering**：用一个 prompted LLM 预测 query 的目标时间窗 $(t_{\text{start}}, t_{\text{end}})$，硬过滤掉所有 $\tau_i$ 不在该窗口内的 session，得到 $\mathcal{M}_{\text{temp}} \subseteq \mathcal{M}$。
        - **Relevance Filtering**：在 $\mathcal{M}_{\text{temp}}$ 上跑 BM25 排序，取 top-$k$（论文里 $k=10$ 表现最好）形成候选池：

$$\mathcal{C} = \arg\text{top-}k_{(\tau_i, S_i): t_{\text{start}} \leq \tau_i \leq t_{\text{end}}} \big( \text{Retriever}(q, S_i) \big).$$

2. **Phase 2 - Fine-grained Selection（精选 + 回答）**：训练后的 $\pi_\theta$ 接收 $(q, \mathcal{C})$，**一次生成**结构化字符串 `{selected_memory: [session_3, session_16], answer: "19 days."}`，从中解析出 evidence subset $\mathcal{S}$ 和 final answer $a$。

- **额外推理策略**：
    - top-$k$ 检索（$k=10$ 最优，是 candidate generation 的超参）；
    - 没有显式温度调节 / 多轮 replan / self-consistency 等；
    - 不使用 Chain-of-Thought 或 ReAct 风格的多步交互。
- **驱动方式**：粗筛阶段（time filter + BM25）是**完全手工/规则式**的；精选与回答阶段**完全由学习到的 $\pi_\theta$ 驱动**——即"读什么 session"+"如何答"在同一次 forward 中由 RL 策略联合给出。

---

## 5. RQ 分析

### RQ1 (What is memory?)

论文将 memory 表征为一个带时间戳的多会话对话条目集合 $\mathcal{M}$，是显式、可寻址、外部、非参数化的长期记忆 $m_l$（典型 T1）。条目通过 BM25 + 时间过滤被检索召回。

### RQ2 (Which component is optimized? Which signal is used?)

优化对象为主 LLM $\pi_\theta$ 的全参数（属于 O2a 单 LLM 优化）。信号是三级合成 reward，答案准确度 $R_a$ + 证据 grounding $R_g$（Jaccard）+ 时间一致性 $R_t$（session 级 logistic 邻近 + utterance 级事件时间保真），加权和为 $R$。

### RQ3 (Target of Optimization)

最终目标是答案准确度 G1（多种 metric：EM / Unit-aware / $\epsilon$-EM / Hamming），同时把写入/选择决策质量 G4（evidence grounding via Jaccard）和辅助过程奖励 G6（时间一致性、format penalty）显式纳入 reward。论文不追求 token efficiency，但消融表明三个目标互补不可分。

### RQ4 (Training Signal and RL Algorithm)

GRPO（属于 A2 类），用同 prompt 下 $G=8$ 个 rollout 的 batch-average 替代 value network，配合 ratio clipping 和 KL 正则。另外，论文附录 C.4 明确比较了 PPO（A1），结论是 PPO 在 Category B/C 上比 GRPO 差 22%–30%。

### RQ5 (How memory evolves, operates?)

记忆本身在系统中是只读、不演化的。不存在 $g_l$ 或 $g_s$ 把新对话写回 memory bank 的步骤。运行时记忆操作只有读：先时间硬过滤 $v_{\text{time}}$ - BM25 排序 $v_{\text{BM25}}$ - 学习到的 $\pi_\theta$ 选 evidence subset $\mathcal{S} \subseteq \mathcal{C}$ 并直接生成答案。整个 pipeline 是 single-step 的检索-回答耦合，不包含多轮迭代或记忆更新。

---

## Conclusion

Memory-T1 关注的是"在带时间戳的多会话对话历史里，怎么挑出时间上对的那几条 session 来回答用户问题"。它的做法是把整条流水线拆成两段：先用一个简单的"时间窗 + BM25"做粗筛，再让一个用 RL 微调过的主 LLM 在候选池里同时输出"我选了哪几个 session"和"答案是什么"。创新是 reward 设计，除了答案对错，作者额外加了两个稠密信号：被选 session 的时间戳是否离 query 时间窗近（logistic 软惩罚）、被选 session 内部 utterance 提到的事件时间是否落在 query 时间窗内（事件级离散打分）。这套 reward 让一个 7B 模型在 Time-Dialog 上拿到 67.0% F1，超过 14B baseline 10 个点，并在 128k 长上下文下不崩溃。