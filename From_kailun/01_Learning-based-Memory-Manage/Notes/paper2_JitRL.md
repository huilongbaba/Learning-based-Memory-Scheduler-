# Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates

**Source:** arXiv:2601.18510v1 [cs.LG], Preprint, January 2026

---

## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$s$|—|状态（state），框架中未单独定义，属于 MDP 形式化的一部分|
|$a$|$a$|动作（action），与框架一致|
|$o$|$o$|观测（observation），与框架一致|
|$\tau = (s_0, a_0, r_0, \ldots)$|$T$|轨迹（trajectory）|
|$\pi_\theta$|$\pi$|模型策略（policy）|
|$\theta$|$\theta$|模型参数，与框架一致|
|$\mathcal{M} = {(s_i, a_i, G_i)}$|$m_l$|动态经验记忆库对应框架中的长期外部记忆|
|$\mathcal{N}(s)$|—|检索返回的近邻集合，对应检索算法 $v$ 的输出|
|$z(s,a)$|—|LLM 输出 logits，框架中未定义，**建议新符号 $\ell$**（logit）|
|$\mathcal{E}$ (Evaluator)|—|LLM-based 逐步奖励评估器，框架中未定义，**建议新符号 $\mathcal{E}$**|
|$G_t$|—|折扣累积回报，标准 RL 概念，框架中未定义，**建议新符号 $G_t$**|
|$\hat{A}(s,a)$|—|优势函数估计，标准 RL 概念，框架中未定义，**建议新符号 $\hat{A}$**|
|$\hat{V}(s), \hat{Q}(s,a)$|—|非参数值函数估计|
|$\beta$|—|KL 约束温度参数|

> 框架符号体系主要面向 RAG/记忆检索场景，缺少标准 RL 中 MDP 层面的符号（如 $s$, $r$, $V$, $Q$, $A$, $G$）。

---

## 1. Problem Setting

**记忆类型：** JitRL 处理的是 **cross-episode（跨 episode）的长期记忆**，对应 $m_l$。记忆库 $\mathcal{M}$ 在多个 episode 之间持续积累，存储历史交互的 $(s, a, G)$ 三元组。同时也支持 cross-task 记忆迁移（§5.3, §5.5），即来自不同任务的经验可跨任务检索复用。

**决策过程建模：** 标准 MDP。论文在 §3 中将问题形式化为策略优化 $J(\theta) = E_{\tau \sim \pi_\theta}[R(\tau)]$，但核心创新在于将策略更新从训练时转移到推理时，通过非参数记忆检索替代梯度更新。

**状态空间 $\mathcal{S}$：** 环境的原始观测经过抽象后的结构化状态表示。WebArena 中为正则化 URL + 局部动作历史；Jericho 中为压缩的结构化文本摘要（§4.1, Appendix F）。

**动作空间 $\mathcal{A}$：** 增强候选集 $C = C_{LLM} \cup {a_i : (s_i, a_i, G_i) \in \mathcal{N}(s)}$，即 LLM 生成的候选动作与记忆中历史动作的并集（Eq. 23, Appendix F）。

**观测空间 $\Omega$：** WebArena 中为 HTML DOM 树 + 截图；Jericho 中为游戏文本描述。

**记忆的数据结构：** 非参数记忆库 $\mathcal{M} = {(s_i, a_i, G_i)}_{i=1}^{N}$，每个条目为 (状态, 动作, 折扣回报) 三元组。检索基于 Jaccard 相似度（Eq. 24），属于 **结构化自然语言条目 + 数值回报** 的混合形式。

|核心组件|论文符号|框架符号|说明|
|---|---|---|---|
|经验记忆库|$\mathcal{M}$|$m_l$|存储 $(s, a, G)$ 三元组的动态非参数记忆|
|检索机制|Top-k Jaccard|$v$|基于 Jaccard 相似度的 kNN 检索|
|基础策略|$\pi_\theta$|$\pi$|冻结的 LLM 策略|
|优化后策略|$\pi^*$|—|通过 logit 调整得到的后验策略|
|评估器|$\mathcal{E}$|—|LLM-based 逐步奖励生成器|
|状态抽象|AbstractState|—|将原始观测压缩为结构化状态|

> JitRL 的一个显著特点是将记忆视为"非参数策略分布"（§2），而非仅用于上下文学习的文本片段。这与 Reflexion、AWM 等方法形成了本质区别——后者将记忆作为 prompt 的一部分，而 JitRL 直接通过记忆调制输出分布。

---

## 2. Training Procedure

**核心立场：JitRL 是一个 training-free 框架，不涉及任何梯度更新。** LLM 参数 $\theta$ 完全冻结（§4）。

**优化的组件：** 严格来说，JitRL 不优化任何可学习参数。它在推理时通过记忆检索 + logit 调整实现"等效策略优化"。优化目标是从冻结的先验策略 $\pi_\theta$ 推导出后验策略 $\pi^*$。

**"优化"算法：** KL 约束下的闭式策略优化（§4.3）。论文证明 logit 加法更新规则是以下目标的精确闭式解（Theorem 4.1）：

$$\pi^* = \arg\max_{\pi'} \Big\lbrace \mathbb{E}_{a \sim \pi'}[\hat{A}(s, a)] - \frac{1}{\beta} D_{KL}(\pi' \Vert \pi_\theta) \Big\rbrace$$

**框架符号标注：**

$$\pi^* = \arg\max_{\pi'} \Big\lbrace \mathbb{E}_{a \sim \pi'}[\hat{A}(s, a)] - \frac{1}{\beta} D_{KL}(\pi' \Vert \pi) \Big\rbrace$$

闭式解映射到 logit 空间（Eq. 10 / 12）：

$$\ell'(s, a) = \ell(s, a) + \beta \cdot \hat{A}(s, a)$$

其中 $\ell(s,a)$ 为基础 LLM 的 logits（论文中为 $z(s,a)$）。

**数据来源：** 在线交互——agent 与环境交互产生的轨迹实时存入记忆库 $m_l$（§4.1）。

**是否冻结 LLM 参数：** 是。$\theta$ 完全冻结，所有"学习"通过记忆积累和推理时 logit 调整实现。

> JitRL 的理论贡献在于证明了 logit 加法更新等价于 KL 正则化策略优化的闭式解。这个证明将推理时的启发式操作提升为有理论保证的策略改进步骤。但值得注意的是，该证明依赖于优势函数估计的准确性，而非参数估计在记忆稀疏时可能不可靠。

---

## 3. Reward Signal

**奖励类型：** Dense step-level reward（§4.1）。每个 episode 结束后，Evaluator 对轨迹中的每一步生成标量奖励 $r_t$，再聚合为折扣回报 $G_t = \sum_{u=t}^{T} \gamma^{u-t} r_u$（Eq. 3）。

**奖励来源：** LLM-as-judge。论文使用 LLM-based Evaluator $\mathcal{E}: \tau \to {r_t}_{t=1}^T$，通过反思式评估（Reflective Step-wise Rewards）对每一步的贡献进行打分（§4.1）。评分范围为 $[-3, +3]$（Appendix G）。在 Jericho 中还结合了游戏引擎提供的环境分数。

**奖励分配机制：** 通过 LLM Evaluator 的 credit assignment 实现逐步分配，而非简单的均匀分配。Evaluator 根据每步动作对任务成功的贡献独立评分，再通过折扣因子 $\gamma$ 聚合为长期回报（§4.1）。

**辅助奖励/正则项：**

- KL 散度正则项 $\frac{1}{\beta} D_{KL}(\pi' | \pi_\theta)$，用于约束更新后的策略不偏离基础 LLM 太远（Eq. 8）
- 探索奖励：对未见动作采用 UCB 风格的乐观估计 $\hat{Q}(s,a) = \hat{V}(s) + \alpha / |\mathcal{N}(s)|$（Eq. 6）
- 优势归一化 $\tilde{A}(s,a) = A(s,a) / (\max_{a'} |A(s,a')| + \epsilon)$（Eq. 25）

> Reflective Step-wise Rewards 的质量高度依赖 Evaluator LLM 的能力。论文未讨论 Evaluator 评分的一致性和准确性问题。此外，$\gamma$ 的选择对不同任务差异较大（WebArena 用 0.1，Jericho 用 0.5），暗示需要任务相关的调优。

---

## 4. Inference Procedure

**记忆初始化：** 记忆库 $\mathcal{M}$ 初始化为空集 $\emptyset$（Algorithm 1）。agent 从零开始积累经验。

**每步决策流程（Algorithm 1, §4）：**

1. **观测 → 状态抽象：** 接收原始观测 $o$，通过 AbstractState 函数压缩为结构化状态 $s$
2. **检索：** 基于 Jaccard 相似度从 $m_l$ 中检索 top-$k$ 近邻 $\mathcal{N}(s)$
3. **值估计：** 计算状态值 $\hat{V}(s)$（Eq. 4）和每个候选动作的动作值 $\hat{Q}(s,a)$（Eq. 5/6）
4. **优势估计：** $\hat{A}(s,a) = \hat{Q}(s,a) - \hat{V}(s)$（Eq. 7），并归一化（Eq. 25）
5. **Logit 调整：** $\ell'(s,a) = \ell(s,a) + \beta \cdot \tilde{A}(s,a)$（Eq. 10）
6. **采样：** $a \sim \text{Softmax}(\ell')$
7. **Episode 结束后：** Evaluator 生成逐步奖励 → 计算折扣回报 $G_t$ → 存入 $m_l$

**推理时的额外策略：**

- Top-$k$ 检索（$k=10$）
- 候选集增强：合并 LLM 候选和记忆中的历史动作（Eq. 23）
- 探索-利用平衡：以概率 $\lambda$ 对未见动作赋予乐观估计（Eq. 6）
- Verbalized Logit 变体：对不暴露 log-probabilities 的黑盒模型，通过提示 LLM 输出置信度分数作为 logits（§5.1）
- WebArena 中使用两阶段层级匹配检索（先 URL 过滤，再 Jaccard 匹配）（Appendix F）

**策略驱动方式：** 主要由 learned（通过经验积累学到的非参数策略）驱动，但仍有手工设计的规则：状态抽象函数、检索相似度阈值、探索率 $\lambda$、候选集构建逻辑等均为手工设定。

> JitRL 提供了两种 logit 获取方式（Token-level 和 Verbalized），使其能适配黑盒 API 模型。但 Verbalized Logit 的精度可能显著低于真实 log-probabilities，论文未充分讨论两种方式的性能差异。

---

## 5. RQ 分析
### **RQ1： What is memory?**

记忆被定义为动态非参数经验库 $\mathcal{M} = {(s_i, a_i, G_i)}$，存储结构化状态-动作-回报三元组。与传统 ICL 方法将记忆作为 prompt 文本不同，JitRL 将记忆视为经验分布的样本，用于非参数值函数估计。记忆的核心作用是支撑推理时的优势估计，进而调制 LLM 输出分布。

### RQ2：How memory evolves, operates?

记忆通过"收集-评估-存储"循环演化：每个 episode 结束后，Evaluator 对轨迹进行逐步评分，计算折扣回报后存入记忆库。读操作通过 kNN 检索实现，写操作在 episode 结束时批量进行。记忆在运行时持续增长，不涉及遗忘或压缩机制。

### RQ3：Which component is optimized? Which signal is used?

不优化任何可学习参数。通过记忆检索获得的优势估计 $\hat{A}$ 直接调制 LLM 的输出 logits，实现等效策略改进。信号来源为 LLM-as-judge 的逐步奖励和环境反馈。KL 正则项确保更新后的策略不偏离基础模型。

### RQ4：Regarding online optimization

JitRL 天然支持在线持续学习，这是其核心设计目标。它实现的是 cross-episode 的在线适应，即agent 在同一任务的多次尝试中通过积累记忆持续改进，同时也支持 cross-task 迁移。由于不涉及梯度更新，不存在灾难性遗忘问题。

---

## Conclusion

这篇文章提出了一种无需梯度更新的推理时策略优化框架，使冻结参数的 LLM agent 能够在部署后持续学习。其核心思想是维护一个存储 (状态, 动作, 回报) 三元组的动态记忆库，在推理时通过 kNN 检索估计每个候选动作的优势函数，再将优势估计加到 LLM 的输出 logits 上以实现策略改进。论文从理论上证明了这种 logit 加法更新是 KL 约束策略优化的精确闭式解，并证明了值估计和策略更新的一致性收敛。在 WebArena 和 Jericho 基准上，JitRL 不仅超越了所有 training-free 基线，还优于需要昂贵训练的 WebRL 等权重更新方法，同时将成本降低了 30 倍以上。