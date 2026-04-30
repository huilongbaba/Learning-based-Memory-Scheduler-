# Beyond Stochastic Exploration: What Makes Training Data Valuable for Agentic Search (HiExp)

**Source:** [arXiv:2604.08124v1](https://arxiv.org/abs/2604.08124) (2026-04-09 修订， ACL Rolling Review)

---
## 符号映射表

| 论文原始符号 | 框架符号 | 说明 |
|---|---|---|
| $\pi_\theta$ (policy model) | $\pi_\theta$ | 被训练的搜索 agent（Qwen2.5-7B/32B Instruct） |
| HEK $= \lbrace E_1, E_2, \ldots, E_L \rbrace$ | $m_l$ | 长期记忆，分层经验知识库（外部、非参数化、离散条目） |
| $E_1$ (case-based / Instance) | $m_l$ 的 instance 层 | 实例级经验，存储具体决策点与陷阱 |
| $E_2$ (pattern) | $m_l$ 的 pattern 层 | 任务结构层级的模式化经验 |
| $E_3$ (strategy) | $m_l$ 的 strategy 层 | 高层策略性元规则 |
| $e_i$ (description), $d_i$ (title) | content, key | 单条经验条目的内容与检索键 |
| $\phi(\cdot)$ (semantic encoder, multilingual-e5-base) | $v$ 的一部分 | 嵌入函数，用于相似度检索 |
| $\text{cos\_sim}(\phi(q_t), \phi(d))$ | $v$ 的打分函数 | 检索算法 |
| $e^*$ (retrieved experience) | $\text{top-1 / top-}k$ 检索结果 | 与当前 $q_t$ 最匹配的经验条目 |
| $\mathcal{R}$ (search engine) | 外部 doc 检索（与 $m_l$ 并列） | 文档语料库（Wikipedia 2018），与 HEK 并列被使用 |
| $y$ (output sequence) | $T$ | 完整推理轨迹（含 think / tool_call / tool_response） |
| $q_t$ (intermediate query) | 第 $t$ 步的 $q$ | 中间子查询 |
| $r_{orm}$ (outcome reward) | terminal reward | 答案级稀疏奖励 |
| $\hat{A}_i$ | advantage | GRPO 的 group-relative advantage |
| $\text{Reflect}(\cdot)$, agglomerative clustering | $g_l$ | 离线的长期记忆构建函数（不在 RL 内更新） |

---
## 概览

这篇文章用了历史 rollout 轨迹（成功的和失败的各一组），让 LLM 通过对比反思（contrastive self-reflection）从中挖出"关键决策点"和"推理陷阱"，再用层级聚类把零散的 case 抽象成 pattern 和 strategy。  
最后得到了两样东西：(1) 一个三层级的外部经验知识库 HEK（E1 实例 / E2 模式 / E3 策略），离线构建，可插拔；(2) 一个用 GRPO 在"经验对齐 rollout"下训练出的搜索 agent（HiExp-Searcher），主 LLM 参数被更新。  
文章的直接优化目标是多跳 QA 的答案准确率（F1 / EM / CEM）。但更核心的卖点是训练稳定性。通过让 rollout 阶段被 HEK 引导（不再是纯随机探索），advantage 方差和梯度方差显著降低，policy 更新更稳。

---
## 1. Problem Setting

- **记忆类型**：cross-chat 的长期记忆 $m_l$（HEK），跨样本、跨 episode 持久；不涉及单对话内的 $m_s$。
- **决策建模**：多步 agentic search 被建模为 token-level MDP，但训练用 critic-free 的 GRPO（group-relative），不显式估 value。HEK 作为**外部条件变量**进入 policy。
- **状态空间** $\mathcal{S}$：当前推理上下文 + 已有检索结果 + 当前中间查询 $q_t$。
- **动作空间** $\mathcal{A}$：think token / tool_call (search) / 终止并 box 答案。
- **观测空间** $\Omega$：tool_response（含 doc 检索结果与 case-based 经验 $E_1$）。
- **记忆数据结构**：层级化的离散条目集合，每条包含 `{type, title, tags, description, thinking, qa_groups}`，由 title 充当 child chunk 做语义匹配，description 作为 parent chunk 注入推理上下文。

| 组件 | 论文实现 | 框架符号 |
|---|---|---|
| 长期记忆 $m_l$ | HEK = $\lbrace E_1, E_2, E_3\rbrace$ | $m_l$ |
| 检索算法 $v$ | multilingual-e5-base + cosine + parent-child 架构 + 阈值 0.8 / top-5 | $v$ |
| 短期记忆 $m_s$ | 当前 rollout 的 reasoning trace（context window 内） | $m_s$ |
| 长期记忆更新 $g_l$ | 离线的 contrastive distillation + agglomerative clustering | $g_l$（仅离线一次） |
| 模型 $M$ | Qwen2.5-7B/32B Instruct（policy）+ Qwen-Max（teacher，可选） | $\pi_\theta$ |

> HEK 在 RL 训练中是冻结的。

---
## 2. Training Procedure

- **优化对象**：主 LLM $\pi_\theta$ 的全部参数（即被训练的 search agent 本身）。HEK 不被反向传播更新。
- **优化算法**：GRPO（主），也验证了 GSPO；与 PPO 不同，无 value network。
- **训练数据来源**：
  - 多跳 QA：R1-Searcher stage-2 数据 + 8000 条 Musique 抽样，共约 16k；
  - 数学推理：OpenR1-Math 45k 子集；
  - HEK 本身由 policy 模型自己 rollout 出的轨迹自蒸馏得到（Section 4.3.3 显示 self-distillation 比 strong-teacher distillation 略好 +1.2%）。
- **是否冻结 LLM 参数**：**否**，policy LLM 全参数更新。HEK 嵌入器 $\phi$ 冻结。
- **核心训练目标**：

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\ \lbrace y_i \rbrace \sim \pi_{\text{old}}} \frac{1}{G}\sum_{i=1}^{G} \min\Big( r_i(\theta)\hat{A}_i,\ \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\Big) - \beta\, D_{KL}$$

其中 $r_i(\theta) = \dfrac{\pi_\theta(y_i \mid x, E_l;\ \mathcal{R})}{\pi_{\text{old}}(y_i \mid x, E_l;\ \mathcal{R})}$，关键点是分子分母都**条件于经验 $E_l$ 与检索系统 $\mathcal{R}$**——这是与 vanilla GRPO 的唯一形式差异。

- **关键 trick**：loss 计算时**对 retrieved doc snippet 与 case-based experience $E_1$ 做 mask**，避免 policy 把"被注入的内容"也当作要优化的 token，这与 Search-R1 等方法的处理一致。

---
## 3. Reward Signal

- **奖励类型**：sparse terminal reward（基于答案正确性），无 step-level 奖励。
- **奖励来源**：rule-based outcome reward $r_{orm}$（F1 / EM 与 ground truth 比对，对开放域用 LLM-as-Judge）。
- **分配机制**：通过 GRPO 的 group-relative advantage 隐式做 credit assignment，对组内每条 rollout 用 $\hat{A}_i = (r_i - \text{mean}(\mathbf{r})) / \text{std}(\mathbf{r})$。**HEK 不进入奖励，只进入 rollout 条件**。
- **辅助奖励 / 正则**：仅 KL 正则项（系数 $\beta = 10^{-3}$），无其他 shaping reward。

---
## 4. Inference Procedure

- **记忆初始化**：HEK 在训练前已离线构建完毕（Algorithm 1），训练与推理共享同一份 HEK；不依赖任何 episode 内的写入。
- **每步决策流程**：
  1. 输入 $q$ → 编码 $\phi(q) \in \mathbb{R}^d$；
  2. 在 system prompt 注入 E2/E3（top-5，开局一次）；
  3. policy 生成 `<think>` → 决定是否 `<tool_call>`；
  4. 若调用 search，外部环境返回 doc + 触发 cos-sim ≥ 0.8 的 $E_1$ case-based experience，二者一并作为 `<tool_response>`；
  5. 重复直到 policy 输出 `\boxed{·}` 终止。
- **额外推理策略**：
  - top-k = 5 for 策略级，top-1 + 阈值 0.8 for 实例级；
  - parent-child retrieval 架构（title 匹配，description 注入）；
  - 训练 / 推理 sampling temperature = 1.0。
- **是 π 驱动还是手工规则**：混合。**何时检索**完全由 $\pi$ 决策；**注入哪一级 HEK** 由手工规则（开局注入 E2/E3，中间步注入 E1）；**注入哪条具体经验**由 $\phi$ 的相似度决定，与 $\pi$ 解耦。

---
## 5. RQ 分析

### RQ1 (What is memory?)

文章里用的 HEK 是一个外部、非参数化、层级化的离散条目集合，记忆形态是经验性的。条目源于模型自己的历史 rollout（self-distillation），通过 cos-sim 检索注入到当前 context。

### RQ2 (Which component is optimized? Which signal is used?)

优化对象是主 LLM 的全部参数（O2a，单 agent）。信号是 outcome reward（最终答案 F1/EM/CEM，G1）。HEK 不被 RL 优化，构建过程是 LLM 驱动的离线自蒸馏（offline + self-reflection prompt），不含梯度更新。

### RQ3 (Target of Optimization)

主目标是回答准确度（G1）。

### RQ4 (Training Signal and RL Algorithm)

主算法 A2 GRPO。论文也额外验证了 GSPO（也是 group-relative 家族），均能与 HiExp 协同。

### RQ5 (How memory evolves, operates?)

离线一次性构建，运行时只读。$g_l$ 体现为两阶段过程：(a) 对每个训练样本做 K=多次 rollout，按 outcome 划分成功/失败两组，LLM-self-reflect 提取 $E_1$；(b) 对 $E_1$ 做 agglomerative clustering（Ward linkage，阈值 $\tau_l$），LLM 总结出 $E_2$，再对 $E_2$ 聚类得到 $E_3$。运行时 $v$ 用 cos-sim 在 $\phi$ 嵌入空间检索，不存在写入操作。这是与 Memory-R1 / A-MEM 等"运行时动态写入"路线的本质区别。

---
## Conclusion

这篇文章想表达训练数据的价值不止于其标签，整段 rollout 轨迹（尤其是成功 vs. 失败的对比）本身就是宝贵的经验。作者用 LLM 自反思 + 层级聚类，把这些原始轨迹离线提炼成一个三层级（实例/模式/策略）的外部经验库 HEK，然后在 GRPO 训练的 rollout 阶段把 HEK 检索注入到 prompt 里，让随机探索变成"经验对齐的探索"。结果是答案准确率涨 +9.7 CEM，训练方差大幅下降，且方法对算法（GRPO/GSPO）和任务（多跳 QA / 数学）都通用。代价是 HEK 一旦构建就冻结，无法随 policy 演化。