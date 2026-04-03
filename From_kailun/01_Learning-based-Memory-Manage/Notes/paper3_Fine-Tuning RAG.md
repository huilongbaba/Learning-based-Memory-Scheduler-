# Fine-Tuning with RAG for Improving LLM Learning of New Skills

**Source:** [arXiv:2510.01375](https://arxiv.org/abs/2510.01375) (Oct 2025, published at ICLR 2026)

---

## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$s_t \in \mathcal{S}$|—|环境隐状态（POMDP），框架未显式定义|
|$o_t \in \Omega$|$o$|观测|
|$a_t \in \mathcal{A}$|$a$|动作|
|$c_t^{\text{thought}}$|$t$|推理轨迹（ReAct 中的 Thought）|
|$c_t^{\text{state}}$|—|StateAct 的显式状态表示，可视为 $m_s$ 的一种形式|
|$c_t^{\text{action}}$|$a$|环境动作|
|$g$|$q$|任务指令 / human input|
|$H_0 = \lbrace h_1, \dots, h_k \rbrace$|$m_\ell$ (hint bank)|从失败中提取的 hint 库，作为外部长期记忆|
|$\pi_\theta$|$\pi$ (base)|基础策略|
|$\pi_\theta^{\text{RAG}}$|$\pi + R$|检索增强策略|
|$\pi_\phi^{\text{SFT}}$|$\pi'$ (SFT student)|在基线成功轨迹上微调的策略|
|$\pi_\phi$ (Distilled)|$\pi'$ (distilled student)|在 RAG 轨迹（去除 hint）上蒸馏的策略|
|$\phi$|$\theta + \Delta(\phi)$|LoRA adapter 参数叠加在冻结骨干上|
|$s_\psi$|$v$|LLM re-ranking 检索算法（Qwen-2.5 7B）|
|$\tau$|$T$|轨迹|
|$R(s_t, a_t)$|—|环境奖励函数|

---

## 1. Problem Setting

- **Memory 类型：** 本文处理的是 **cross-chat memory**（跨任务的经验积累）。Hint bank 是从训练集的失败轨迹中提取并在新任务中复用的，属于 $m_\ell$（长期外部记忆）。在单个 episode 内不存在动态更新的 $m_s$，但 ReAct 的 Thought 和 StateAct 的 State 可视为隐式的 in-chat working memory。
- **决策过程建模：** POMDP $(\mathcal{S}, \mathcal{A}, T, R, \Omega, O)$，隐状态 $s_t$ 不可直接观测，agent 基于观测历史 $o_{\leq t}$ 做决策。
- **状态空间 $\mathcal{S}$：** 环境的完整世界状态（ALFWorld 中的房间布局与物品位置；WebShop 中的商品库与购物车状态）。
- **动作空间 $\mathcal{A}$：** ReAct: $(c_t^{\text{thought}}, c_t^{\text{action}})$；StateAct: $(c_t^{\text{state}}, c_t^{\text{thought}}, c_t^{\text{action}})$；WebShop 中省略 thought。环境动作为文本命令（`goto`, `take`, `search[...]`, `click[...]` 等）。
- **观测空间 $\Omega$：** 文本形式的环境反馈（ALFWorld 的房间描述；WebShop 的搜索结果/商品页面）。
- **记忆数据结构：** Hint bank 为**自然语言条目**，按任务类别（ALFWorld 6 类）或商品类别（WebShop 5 类）分区存储，每条 hint ≤120 字符，使用占位符（`{object}`, `{container}` 等）保证泛化性。

|核心组件|框架符号|论文实现|
|---|---|---|
|Human input|$q$|任务指令 $g$|
|Observation|$o$|环境文本反馈 $o_t$|
|Thought|$t$|$c_t^{\text{thought}}$（ReAct）|
|Action|$a$|$c_t^{\text{action}}$|
|Long-term memory|$m_\ell$|Hint bank $H$，按类别分区|
|Retrieval algorithm|$v$|LLM re-ranking（Qwen-2.5 7B）|
|Policy|$\pi$|$\pi_\theta$（base）/ $\pi_\phi$（distilled）|
|Trajectory|$T$|序列化的 episode $\tau$|

> 本文的 POMDP 建模是标准的，但 hint bank 的角色较为特殊，它既不是传统的 RAG 知识库（事实性文档），也不是 episodic memory（具体经历），而是从失败中提炼的程序性规则。这种 memory 更接近 "procedural long-term memory"，在框架中可考虑引入符号 $m_\ell^{\text{proc}}$ 加以区分。

---

## 2. Training Procedure

- **优化的组件：** 优化 LoRA adapter 参数 $\Delta(\phi)$，即 $\pi_\phi$ 中的低秩适配器。LLM 骨干参数 $\theta$ **完全冻结**（4-bit 量化）。
- **优化算法：** **Supervised fine-tuning（SFT）**——标准的 next-token cross-entropy（最大似然），非 RL 算法。
- **训练数据来源：** **离线轨迹**，具体为：
    - $\mathcal{D}_{\text{SFT}}$：基线 agent 的成功轨迹（无 hint）
    - $\mathcal{D}_{\text{Distillation}}$：RAG teacher 的成功轨迹，**去除 hint 字符串和 few-shot 示例**
- **冻结 LLM 参数：** 是。骨干量化为 4-bit，仅训练 16-bit 精度的 LoRA adapter，插入 attention 和 MLP 投影层。

**核心训练目标函数：**

论文原始公式（Eq. 1）：

$$\min_\phi \mathbb{E}_{x \sim \mathcal{D}} \left[ -\frac{1}{N} \sum_{i=1}^{N} \log \pi_\phi(x_i \mid x_{< i}) \right]$$

其中 $\mathcal{D} \in \lbrace \mathcal{D}_{\text{Distillation}}, \mathcal{D}_{\text{SFT}} \rbrace$。

统一符号标注：

$$\min_{\Delta(\phi)} \mathbb{E}_{T \sim \mathcal{D}} \left[ -\frac{1}{N} \sum_{i=1}^{N} \log \pi_{\theta + \Delta(\phi)}(x_i \mid x_{< i}) \right]$$

这等价于从 RAG teacher $\pi_\theta^{\text{RAG}}$ 到 student $\pi_\phi$ 的行为蒸馏：期望 $\pi_\phi(a_t \mid o_{\leq t}) \approx \pi_\theta^{\text{RAG}}(a_t \mid o_{\leq t}, H_0)$。

**训练超参数：** LR=2e-4, LoRA rank=64/16（ALFWorld/WebShop）, LoRA $\alpha$=128/32, 单 epoch, batch size=8（梯度累积）, 8-bit AdamW, WebShop 使用 label smoothing $\varepsilon=0.1$。

> 本文完全不使用 RL，仅用 SFT 蒸馏。这意味着 student 只能学到 teacher 的行为分布，无法超越 teacher 的上限。与 GRPO/PPO 等在线优化方法相比，缺乏探索能力，但训练更稳定、数据效率更高。另外，hint 的提取依赖 GPT-4o，引入了对更强模型的隐式依赖。

---

## 3. Reward Signal

- **奖励类型：** **Sparse terminal reward**——仅在 episode 结束时判定成功/失败。ALFWorld 为二值（goal reached or not）；WebShop 为 0-100 分数。
- **奖励来源：** **环境反馈**——ALFWorld 判断目标是否达成；WebShop 计算购买商品与指令的匹配分数。
- **奖励如何分配到各步骤/token：** **不做显式 credit assignment。** 本文不使用 RL，奖励信号仅用于**筛选训练数据**（仅保留成功/高分 episode），而非用于梯度计算。对 WebShop 保留 score ≥ 67 的 episode。
- **辅助奖励或正则项：** 无显式辅助奖励。WebShop 使用 token-level label smoothing（$\varepsilon = 0.1$）作为正则化手段，防止在短轨迹上过拟合。此外，LoRA 本身（低秩约束）可视为隐式正则。

> 奖励仅用于数据过滤而非优化，这是本文与典型 RL-based memory 方法的根本区别。这种 "reward-as-filter" 模式简单有效，但浪费了失败轨迹中的信号，因为失败轨迹仅用于 hint 提取（Stage B），未直接参与梯度更新。

---

## 4. Inference Procedure

- **记忆初始化：** Distilled student **不使用任何外部记忆**。推理时无 hint bank、无检索。RAG teacher 在 $t=0$ 检索 top-$k$（$k=3$）hints 并注入 prompt，之后不再检索。
- **每步决策流程（Distilled student）：**
    1. 接收观测 $o_t$（环境文本反馈）
    2. （ReAct）生成 thought $t$，推理当前状态和下一步计划
    3. 生成动作 $a_t$（环境命令）
    4. 环境返回新观测 $o_{t+1}$
    5. 重复直到任务完成或达到步数上限（ALFWorld: 50, WebShop: 15）
- **额外推理策略：** 无 TopK 检索、无温度调节、无多轮 replan。Distilled student 完全依赖内化的参数知识。RAG teacher 的检索使用 LLM re-ranking（Qwen-2.5 7B 量化模型），在候选 hint 分区内打分选 top-k。
- **推理策略来源：** Distilled student **完全由学习得到的 $\pi_\phi$ 驱动**，无手工规则。但 agent 的 prompt 框架（ReAct/StateAct 的 thought-action 格式）是预定义的结构，属于手工设计的推理脚手架。

> 推理时零检索开销是本文的核心卖点。相比 ExpeL、AutoGuide 等需要持续检索的方法，distilled student 的推理效率显著更高。但这也意味着模型无法适应训练分布之外的全新失败模式。

---

## 5. RQ 分析

### RQ1: What is memory?

本文的 memory 是从 agent 失败轨迹中自动提取的自然语言 hint 条目（$m_\ell$），按任务/商品类别分区存储。每条 hint 是祈使句形式的程序性规则（如 "Ensure the {container} is open before placing the {object}"），使用占位符保证跨任务泛化。这是一种 procedural long-term memory，非 episodic，非 parametric。

### RQ2: How does memory evolve and operate?

在训练流水线中，memory 经历 "提取 → 检索 → 蒸馏 → 消失" 的生命周期：Stage B 中从失败创建 hint（写入 $m_\ell$），Stage C 中检索 hint 辅助 teacher 生成更好轨迹（读取 $m_\ell$），Stage D 中 hint 被从训练数据中删除，student 将其内化为参数知识。推理时 memory 不存在，已被压缩进 $\theta + \Delta(\phi)$。

### RQ3: Which component is optimized? Which signal is used?

优化的是 LoRA adapter $\Delta(\phi)$，使用 next-token cross-entropy（SFT）在 RAG teacher 的成功轨迹上训练。信号来自环境的 sparse terminal reward（成功/失败），但仅用于过滤训练数据，不直接参与梯度计算。无 RL 优化。

### RQ4: Regarding online optimization

本文未直接涉及此问题。本文是纯离线的 train-then-deploy 范式。

---

## Conclusion

这篇文章提出了一种将 RAG 从推理时依赖转化为训练时教师的蒸馏流水线。核心思路是：先让基线 agent 运行并收集失败，用 GPT-4o 从失败中提取可复用的自然语言 hint，再用 hint 辅助 teacher agent 生成更高质量的成功轨迹，最后在去除 hint 的轨迹上用 SFT 训练 student，迫使其将 hint 中的知识内化为参数。蒸馏后的 student 在推理时不需要任何检索，却能达到甚至超过 RAG teacher 的性能，同时 token 开销更低。

它仅在训练期间使用内存（$m_l$ 作为临时提示库，而不是永久运行时组件），将内存提炼为参数来消除推理时的内存占用。因此，其内存生命周期为“创建→使用→销毁”，这完全颠覆了传统RAG或内存增强系统的思路。