# RLIF: Learning to Reason without External Rewards (INTUITOR)

> source: https://arxiv.org/abs/2505.19590 From_kailun/02_NonQA-guided-Model-Training/Papers/RLIF.pdf

## Problem Setting

现有 RL 微调范式依赖**外部奖励信号**：RLHF 需要人类偏注标注训练奖励模型，RLVR 需要领域特定的验证器（如数学题的 gold answer、代码的 test suite）。这限制了 RL 在开放场景下的可扩展性。

本文提出 Reinforcement Learning from Internal Feedback (RLIF) 范式，用模型自身的内在信号替代外部奖励，实现无监督策略优化。具体方法 INTUITOR 使用模型自身置信度（self-certainty）作为唯一奖励信号。

|组件|符号|定义|
|---|---|---|
|输入|q|问题 / 人类输入|
|模型输出|o|生成的完整回答（含推理过程 t 和最终答案 Aₙ）|
|策略|πθ|当前策略模型|
|参考策略|πref|初始参考策略（冻结）|
|内在奖励|u(q, o)|模型内部信号，非外部验证|
|模型参数|θ|待优化参数|

RLIF 的优化目标：

$$\max_{\pi_\theta} \mathbb{E}_{o \sim \pi_\theta(q)}\left[u(q, o) - \beta \cdot \text{KL}[\pi_\theta(o|q) | \pi_{\text{ref}}(o|q)]\right]$$

与 RLHF/RLVR 的核心区别：u(q, o) 来自模型内部状态，而非外部人类标注或规则验证器。

**不涉及记忆机制**：RLIF 不建模 mₛ 或 mₗ，不涉及 RAG 系统 R 或检索算法 v。其核心关注点是**奖励信号的来源**——从外部监督转向内在反馈。

## Training Procedure

用 **GRPO**（Group Relative Policy Optimization）端到端优化，将外部奖励替换为 self-certainty 分数。

**Self-certainty 定义**：模型输出分布与均匀分布之间的平均 KL 散度：

$$\text{Self-certainty}(o|q) := \frac{1}{|o|} \sum_{i=1}^{|o|} \text{KL}(U | p_{\pi_\theta}(\cdot|q, o_{<i})) = -\frac{1}{|o| \cdot |V|} \sum_{i=1}^{|o|} \sum_{j=1}^{|V|} \log(|V| \cdot p_{\pi_\theta}(j|q, o_{<i}))$$

其中 V 为词表，U 为词表上的均匀分布。Self-certainty 值越高 → 模型越"确信"。

**训练流程**：

1. 对每个 q，用行为策略 πθ_old 采样 G 个输出 {o₁, …, o_G}
2. 对每个 oᵢ 计算内在奖励：uᵢ = Self-certainty(oᵢ|q)
3. 计算组内优势函数（替代外部奖励）：

$$\hat{A}_{i,t} = \frac{u_i - \text{mean}({u_1, \ldots, u_G})}{\text{std}({u_1, \ldots, u_G})}$$

4. 用 GRPO 目标更新 πθ：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\min\left[c_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}_\epsilon(c_{i,t}(\theta))\hat{A}_{i,t}\right]\right) - \beta \cdot \text{KL}[\pi_\theta | \pi_{\text{ref}}]\right]$$

其中 $c_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}$ 为重要性权重。

**关键设计选择**：

- Self-certainty 是 **mode-seeking** 的（KL(U‖p)），不同于 entropy（mode-covering），对长文本生成的 length bias 更小
- 使用 **online self-certainty**（奖励信号随 πθ 共同演化），而非 offline（固定基座模型打分），以防止 reward hacking
- 优势函数均匀分配到所有 token，联合优化推理过程与输出质量

**训练超参数**（Qwen2.5-3B on MATH）：

|参数|值|
|---|---|
|基座模型 M|Qwen2.5-3B|
|训练数据|MATH (7,500 题)|
|每次更新 batch|128 题|
|每题采样数 G|7|
|KL 惩罚 β|0.005|
|学习率|3×10⁻⁶|
|训练步数|58|

## Reward Signal

**奖励类型**：内在奖励（intrinsic reward），无外部监督

**奖励信号**：Self-certainty —— 模型对自身输出的置信度度量

$$u_i = \text{Self-certainty}(o_i|q)$$

**与其他奖励信号的对比**：

|奖励信号|类型|是否需要外部标注|训练稳定性|
|---|---|---|---|
|Gold answer (RLVR/GRPO)|外部、二值|✅ 需要 gold answer|稳定|
|Plurality Voting (GRPO-PV)|自洽性|❌|稳定|
|**Self-certainty (INTUITOR)**|**内在、连续**|**❌**|**稳定（online 模式下）**|
|Negative entropy (EM-RL)|内在|❌|❌ 易崩溃（repetition trap）|
|Random reward|随机|❌|❌ 严重退化|
|Log probability|内在|❌|❌ length bias 导致崩溃|

**Online vs Offline self-certainty**：

- Offline（固定基座模型打分）：约 step 100 时被策略模型 exploit（通过附加已解决问题来膨胀置信度），导致 response length 暴涨 + accuracy 崩溃
- **Online（随策略模型更新）**：奖励信号与策略共演化，避免 reward hacking，训练稳定

**奖励信号质量验证**（Mann-Whitney U 检验，MATH500）：

- INTUITOR 训练后的模型在区分自身正确/错误回答时，self-certainty 的 p-value 最低、effect size r 最大（r=0.45），优于 GRPO（r=0.35）和基座模型（r≈0）

## Inference Process

1. 给定 q，模型 πθ 直接生成输出 o（含推理过程 t + 最终答案 Aₙ）
2. 使用 greedy decoding，无需采样多个候选或后处理
3. 无外部验证、无记忆检索、无工具调用

策略完全由训练后的 πθ 驱动，推理阶段不使用 self-certainty 评分。

**涌现行为**：

- 训练后模型在 CRUXEval-O 和 LiveCodeBench 上自发产生 **pre-reasoning**（先自由推理，再生成结构化输出），类似 DeepSeek-R1 的长链推理
- 训练过程呈现阶段性：先学会生成有效代码 → 再发展出 pre-code reasoning → 最终形成 step-by-step planning + code + explanation 的完整结构

## Dataset & Evaluation

**基座模型**：Qwen2.5-1.5B / 3B / 7B / 14B，Qwen3-14B，Llama3.2-3B-Instruct，OLMo-2-7B-SFT

**训练数据**：MATH (7,500 题)，Codeforces（代码变体）

**评估基准**：

|类型|数据集|
|---|---|
|数学推理 (in-domain)|GSM8K, MATH500|
|代码推理 (out-of-domain)|LiveCodeBench v6 (LCB), CRUXEval-O|
|知识推理|MMLU-Pro|
|指令遵循|AlpacaEval 2.0|

**主要结果**（Qwen2.5-3B，训练于 MATH）：

|方法|GSM8K|MATH500|LCB|CRUXEval-O|MMLU-Pro|AlpacaEval|
|---|---|---|---|---|---|---|
|Base|0.673|0.544|0.093|0.236|0.377|3.72|
|GRPO (gold answer)|0.826|0.636|0.085|0.341|0.403|6.91|
|GRPO-PV (无 gold)|0.820|0.636|0.086|0.299|0.398|6.17|
|**INTUITOR (无 gold)**|**0.792**|**0.612**|**0.153**|**0.416**|**0.379**|**7.10**|

**关键发现**：

- In-domain（数学）：INTUITOR 略低于 GRPO，但差距不大
- **Out-of-domain（代码）**：INTUITOR 显著优于 GRPO（LCB: 0.153 vs 0.085，CRUXEval-O: 0.416 vs 0.341）
- 指令遵循：INTUITOR 在 AlpacaEval 上优于 GRPO（7.10 vs 6.91）
- 早期学习速度：INTUITOR 在 step 10 时已超越 GRPO（GSM8K: 0.811 vs 0.758）

## RQ

本文不涉及 memory（mₛ/mₗ）管理，核心贡献在于 Reward 信号设计。

### 1. 如何在没有外部奖励/标注的情况下实现 RL 微调？

提出 RLIF 范式，用模型内在信号 u(q, o) 替代外部奖励 v(q, o) 或 rφ(q, o)，具体实例化为 self-certainty 指标。

### 2. Self-certainty 作为内在奖励信号为什么有效？它和 entropy minimization 有什么本质区别？

Self-certainty 是 mode-seeking 的 KL(U‖p)，对 length bias 更鲁棒；entropy minimization 容易导致 repetition trap 和模型崩溃。Online self-certainty 还能防止 reward hacking。

### 3. 内在奖励是否能泛化到训练域之外？

是。在 MATH 上训练的 INTUITOR 在 out-of-domain 代码任务上的提升显著优于使用 gold answer 的 GRPO，表明 process-aware 的连续内在奖励比 outcome-based 的二值外部奖励具有更好的迁移能力。