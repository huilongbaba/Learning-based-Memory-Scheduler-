# Process Reinforcement through Implicit Rewards (PRIME)

**Source:** [arXiv:2502.01456v2](https://arxiv.org/abs/2502.01456v2) (Feb 2025, revised Sep 2025)

---

## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$x$|$q$|输入 prompt / question|
|$y$|$T$|完整的生成轨迹（response）|
|$y_t$|$a$ (在时间步 $t$)|第 $t$ 步的 token/action|
|$y_{<t}$|$C$|当前步之前的上下文（prompt + 已生成 token）|
|$\pi_\theta$|$\pi$|策略模型（policy）|
|$\theta$|$\theta$|策略模型参数（一致）|
|$\pi_\phi$|新增 $\pi_R$（隐式 PRM）|隐式过程奖励模型，本质是一个语言模型|
|$\pi_{\text{ref}}$|新增 $\pi_{\text{ref}}$|参考模型，用于计算隐式奖励的基准分布|
|$r_o(y)$|新增 $r_o$|结果级奖励（outcome reward），由验证器给出|
|$r_\phi(y_t)$|新增 $r_p$|隐式过程奖励（token 级 dense reward）|
|$A_t$|无直接对应，沿用 $A_t$|优势函数（advantage function）|
|$K$|无直接对应，沿用 $K$|每个 prompt 采样的响应数量|
|$\beta$|无直接对应，沿用 $\beta$|隐式奖励的温度/缩放系数|
|$V(y_{<t})$|无直接对应，沿用 $V$|价值模型预测的状态价值|

> 本文的核心创新不在于 memory 模块的引入，而在于将隐式过程奖励模型（Implicit PRM） 作为一种可在线更新的 dense reward 信号源。在框架语境下，Implicit PRM 可类比为一种 参数化的短期记忆 $m_s$，它编码了"当前策略分布下，每个 token 对最终正确性的贡献"这一知识，并随训练在线更新。

---

## 1. Problem Setting

### 要解决的问题

大语言模型（LLM）的强化学习训练中，**稀疏的结果级奖励**（outcome reward，仅在生成完毕后给出一个标量）存在三个核心问题：（1）鼓励"答案碰巧正确但过程错误"的伪解；（2）样本效率低；（3）信用分配困难（credit assignment）。密集的过程奖励（process reward）理论上可以缓解上述问题，但面临三大挑战（§2.2）：

- **C1：过程奖励难以定义**——步骤边界不明确，逐 token 标注成本极高
- **C2：PRM 在线更新不可扩展**——传统 PRM 需要步骤级标注，无法在 RL 训练中实时更新，导致分布偏移和奖励黑客（reward hacking）
- **C3：显式奖励建模带来额外开销**——需要单独的数据收集和训练阶段

### 形式化定义

- **输入：** prompt $q \sim D$（数学或编程问题）
- **输出：** 轨迹 $T = (a_1, a_2, \dots, a_n)$，即模型逐 token 生成的完整响应
- **目标：** 学习策略 $\pi(a_t | q, C)$ 最大化期望累积折扣回报

### 任务类型与数据集

- **任务：** 竞赛级数学推理、编程
- **评测基准：** AIME 2024, AMC, MATH-500, Minerva Math, OlympiadBench, LeetCode, LiveCodeBench
- **基座模型：** Qwen2.5-Math-7B-Base

> 本文的问题定义精准地抓住了 RL for LLM 的痛点。与 DeepSeek-R1 等工作类似，都认为密集奖励是提升推理能力的关键，但 PRIME 提出了更可扩展的解决方案。

---

## 2. Training Procedure

### 训练流程

PRIME 的训练流程分为两个阶段：

**阶段一：SFT 热身。** 在基座模型上进行监督微调，教模型学习"行动导向的思维链"（action-centric chain-of-thought）推理模式（§D）。使用 230K 数据，包含数学、编程和生物医学。

**阶段二：在线 RL 训练（核心）。** 算法流程（Algorithm 1）：

1. **初始化：** 策略模型 $\pi$、隐式 PRM $\pi_R$、参考模型 $\pi_{\text{ref}}$ 均从 SFT 模型初始化
2. **采样：** 对每个 prompt $q$，策略模型生成 $K$ 个响应 ${T^1, \dots, T^K}$
3. **结果验证：** 使用规则验证器计算结果奖励 $r_o(T^i)$
4. **在线过滤：** 过滤掉全对或全错的 prompt（保留中等难度）
5. **计算隐式过程奖励：** 通过 $\pi_R$ 和 $\pi_{\text{ref}}$ 的对数概率差得到 token 级 dense reward
6. **更新隐式 PRM：** 使用交叉熵损失在当前 rollout 上更新 $\pi_R$（在线更新）
7. **计算优势函数并更新策略：** 结合过程奖励和结果奖励计算 advantage，使用 PPO clip loss 更新 $\pi$

### Memory 如何参与训练

在框架语境下，隐式 PRM $\pi_R$ 扮演了一种**可学习的参数化记忆**角色：

- 它"记住"了当前策略分布下，每个 token 对最终结果的贡献度
- 通过**在线更新**，这种记忆随策略演化而同步演化，避免分布偏移
- 参考模型 $\pi_{\text{ref}}$ 则是一种**冻结的长期记忆** $m_l$，提供稳定的基准分布

### 核心公式

**隐式过程奖励（Eq. 3）：**

$$r_p(a_t) = \beta \log \frac{\pi_R(a_t | C)}{\pi_{\text{ref}}(a_t | C)}$$

> 论文符号：$r_\phi(y_t) = \beta \log \frac{\pi_\phi(y_t|y_{<t})}{\pi_{\text{ref}}(y_t|y_{<t})}$

**优势函数（Eq. 5）：**

$$A^i_t = \underbrace{\sum_{s=t}^{|T^i|} \gamma^{s-t} \cdot \left( r_p(a^i_s) - \frac{1}{K-1}\sum_{j \neq i} r_p(\bar{a}^j) \right)}_{\text{RLOO with implicit process rewards}} + \underbrace{r_o(T^i) - \frac{1}{K-1}\sum_{j \neq i} r_o(T^j)}_{\text{RLOO with outcome rewards}}$$

> 论文中 $\bar{r}_\phi(y^j)$ 指第 $j$ 个响应的平均隐式过程奖励。

**PPO clip 损失（Eq. 6）：**

$$L_{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( \frac{\pi(a_t|C)}{\pi_{\text{old}}(a_t|C)} A_t, ; \text{clip}\left(\frac{\pi(a_t|C)}{\pi_{\text{old}}(a_t|C)}, 1-\epsilon, 1+\epsilon \right) A_t \right) \right]$$

**隐式 PRM 在线更新损失（Eq. in Algorithm 1, step 8）：**

$$L_{\text{CE}}(\phi) = -\mathbb{E}_{(q, T, r_o)} \left[ r_o \cdot \log \sigma(r_R(T)) + (1 - r_o) \cdot \log(1 - \sigma(r_R(T))) \right]$$

其中 $r_R(T) = \beta \log \frac{\pi_R(T)}{\pi_{\text{ref}}(T)}$ 为序列级隐式奖励，$\sigma$ 为 sigmoid 函数。

> PRIME 的关键洞察在于：隐式 PRM 的训练只需要结果级标签（outcome labels），却能产出 token 级的密集奖励。这打破了"密集奖励需要密集标注"的常规假设，使在线 PRM 更新变得可行。

---

## 3. Reward Signal

### 奖励信号类型

PRIME 使用**双重奖励信号**的组合：

1. **结果奖励 $r_o$（稀疏，外部）：** 规则验证器给出，数学任务为精确匹配（0/1），编程任务为通过率
2. **隐式过程奖励 $r_p$（密集，内部）：** 由隐式 PRM 通过对数概率比计算，token 级别

### 奖励与 Memory 的交互

- 隐式 PRM $\pi_R$ 本身就是奖励的来源——它通过与参考模型 $\pi_{\text{ref}}$ 的对数概率差产生 token 级奖励
- $\pi_R$ 在训练过程中使用结果奖励 $r_o$ 作为监督信号进行在线更新（CE loss）
- 这形成了一个闭环：$r_o$ → 更新 $\pi_R$ → 产生 $r_p$ → 更新 $\pi$ → 产生新的 rollout → $r_o$

### 关键实验发现

- **密集 vs 稀疏奖励（§4.3）：** PRIME 相比仅用结果奖励的 RLOO，训练奖励提升 6.9%，样本效率提升 2.5 倍
- **在线更新至关重要（§5.1）：** 离线 PRM 的分类精度随训练进行逐渐下降（reward hacking），在线 PRM 则持续上升
- **奖励优于价值（§5.4）：** 将隐式 PRM 作为 reward model 使用远优于作为 value model 使用

> PRIME 的奖励设计优雅地绕过了 PRM 需要步骤级标注的瓶颈。从 reward shaping 的视角看（§C.3），隐式过程奖励满足基于势函数的奖励塑形（PBRS）定义，理论上不改变最优策略但加速学习。

---

## 4. Inference Procedure

### 推理时的流程

推理时，PRIME 的行为与标准自回归生成相同：

- **策略模型 $\pi$** 逐 token 生成响应
- **无需使用隐式 PRM**——PRM 仅在训练阶段使用
- 最终模型 Eurus-2-7B-PRIME 是一个独立的语言模型，推理成本与普通 LLM 无异

### 与训练阶段的差异

|方面|训练|推理|
|---|---|---|
|隐式 PRM|在线使用并更新|不使用|
|参考模型|提供基准 logprobs|不使用|
|结果验证器|提供 $r_o$ 信号|不使用|
|采样方式|每 prompt $K$ 个响应|标准采样（pass@1 或 best-of-N）|

> 推理时不需要额外模型是 PRIME 的实用优势之一。与需要 PRM 做 best-of-N 搜索的方法不同，PRIME 将密集奖励的价值"蒸馏"到了策略模型本身。

---

## 5. RQ 分析

### RQ1: What is memory?

PRIME 不显式引入 memory 模块，但隐式 PRM ($\pi_R$) 可被理解为一种参数化的过程评估记忆。它编码了"在当前策略分布下，每个 token 对最终结果的贡献"这一动态知识。参考模型 $\pi_{\text{ref}}$ 则是一种冻结的基线记忆，提供初始分布信息。

### RQ2: How memory evolves, operates?

隐式 PRM 通过在线学习持续演化——每一轮 RL 迭代中，它在当前策略的 rollout 上使用结果标签（CE loss）进行更新。这使其始终与策略分布保持同步，避免了离线 PRM 因分布偏移导致的精度下降。实验表明，在线更新的 PRM 分类精度从约 50% 持续上升至约 75%，而离线 PRM 从约 70% 降至约 55%。

### RQ3: Which component is optimized? Which signal is used?

**本文立场：** 两个组件被同时优化：

1. 策略模型 $\pi$：使用 PPO clip loss 优化，信号来自过程奖励 + 结果奖励的组合优势函数
2. 隐式 PRM $\pi_R$：使用 CE loss 优化，信号为结果级标签 $r_o$

关键发现：SFT 模型直接初始化 PRM 优于专门训练的 PRM，PRIME 适用于 REINFORCE、GRPO、PPO、RLOO 等多种 RL 算法。

### RQ4: Regarding online optimization

在线优化是 PRIME 的核心设计原则。论文明确论证了：

- 在线 PRM 更新有效防止了奖励黑客（reward hacking）
- 仅需结果级标签即可在线更新（无需步骤级标注），计算开销与传统 ORM 相当
- 每步训练时间比 RLOO 多 24%，但总体训练效率提升约 2 倍
- "Zero" 实验表明可直接从基座模型开始 RL，跳过 SFT

---

## Conclusion

这篇文章提出了一种在大语言模型的 RL 训练中引入密集过程奖励的可扩展方法。其核心思路是利用隐式过程奖励建模，通过策略模型与参考模型的 token 级对数概率比来推导密集奖励，只需结果级标签即可训练，从而实现了 PRM 的在线更新。这解决了传统 PRM 需要昂贵的步骤级标注、无法在线扩展、以及需要独立训练阶段这三大挑战。从 memory 的视角看，隐式 PRM 是一种随策略同步演化的参数化评估记忆，其在线更新机制是防止奖励退化的关键。