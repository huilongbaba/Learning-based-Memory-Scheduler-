# Training Large Reasoning Models Efficiently via Progressive Thought Encoding

**Source:** [https://openreview.net/pdf?id=q4iJxp47CT](https://openreview.net/pdf?id=q4iJxp47CT) (2026-02-18 修订，Accepted at ICLR 2026 poster，6644)

---

## 符号映射表

|论文原始符号|框架符号|含义说明|
|---|---|---|
|KV cache $K_t, V_t$（带滑窗约束）|$m_s$|推理/rollout 时的工作记忆，长度受限于固定 cache window|
|被驱逐的 KV pair $K_e, V_e$|"$o_{< t}$ → $m_s$ 退化品"|已离开 $m_s$ 的过去观察，本应被丢弃|
|LoRA 增量 $\Delta W = A \cdot S_e \cdot B$|$g_l$（参数化更新函数）/ 部分 $m_l$（落入 $\theta$ 中）|把驱逐 token 的信息固化到模型权重|
|全局 latent query $q_g$ + 全局 token $h_g$|$m_z$（latent memory carrier）|可学习的 latent 摘要载体，"承运"驱逐内容|
|上下文状态 $S_e$|$m_z^{(t)}$|由驱逐 KV 经 cross-attention 得到的 latent 张量|
|Cache 剪枝策略 $\mathcal{D}$（滑窗）|$g_s$（短期记忆刷新函数）的简化版|决定哪些 token 被保留 / 驱逐|
|Cache-aware policy $\pi^{\mathcal{D}}_\theta(y \mid p)$|$\pi_\theta$（约束版）|在受限 cache 下运行的策略|
|Rollout 轨迹 $y = (y_1, \ldots, y_T)$|$T$（trajectory，由 $a_1, a_2, \ldots$ 组成）|thinking token 序列|
|$r(y)$（最终答案是否正确）|$r$（terminal reward）|sparse outcome reward|
|主 LLM 参数 $\theta$ + LoRA 部分|$\theta$（其中 LoRA 与 $W_Q^a, W_K^a, W_V^a, A, B, h_g$ 是被更新的部分）|被优化的参数子集|

---

## 概览

这篇文章用了 RL rollout 阶段被驱逐出 KV cache 的"思考 token"，通常滑窗 cache 会直接丢弃它们，而本文反过来，把这些 token 的 key/value 通过一次 cross-attention 浓缩成一个 latent 张量 $S_e$，再以低秩方式写进 LoRA 权重里。  
最后得到了一组带"在线自更新机制"的 LoRA 适配器：训练后，模型在固定 cache 长度下 inference 时，LoRA 权重会随着 token 不断被驱逐而持续滚动更新，从而在小 cache 下"假装"自己看到了完整上下文。  
优化目标是在严格 KV cache 预算下做 RL 训练，同时不掉点。具体优化两件事：(1) 数学推理任务的最终答案准确率。(2) 训练阶段的 peak/mean GPU 显存与 attention FLOPs。

---

## 1. Problem Setting

- **处理的记忆类型**：本文不处理传统意义上的 cross-chat $m_l$（如 RAG / episodic store），而是聚焦于**单条 rollout 内**的 in-chat 工作记忆 $m_s$（KV cache）。其创新点在于：当 $m_s$ 容量受限时，把溢出内容**单向**固化进 $\theta$（参数化 $m_l$ 的退化形态），实现"$m_s$ → $\theta$ 的 streaming 内化"。
- **决策过程建模**：MDP（与标准 GRPO 一致），但 rollout 分布从 $\pi_\theta(y \mid p)$ 替换为 cache 受限的 $\pi^{\mathcal{D}}_\theta(y \mid p)$。
    - $\mathcal{S}$：当前 prompt + 当前 KV cache 状态 + 当前 LoRA 权重 $\theta + \Delta W$
    - $\mathcal{A}$：下一个 token $y_t \in \mathcal{V}$
    - $\Omega$：解码出的 token 历史（部分可见，因 cache 滑动）
- **记忆数据结构**：双层结构。
    - 第一层 $m_s$：$L$ 个 token 的滑窗 KV cache（显式、可读、固定长度）
    - 第二层 $\Delta W$：低秩矩阵 $A, B$ + 中间 latent $S_e \in \mathbb{R}^{d_q \times d}$（隐式、不可读、参数化）

|框架组件|论文实现|
|---|---|
|$m_s$|滑动窗口 KV cache（长度 $L$，question token 永久保留作 sink）|
|$g_s$|"新 token 入栈，最旧 25% thinking token 出栈" 的滑窗规则|
|$m_l$ / $\theta'$|主模型参数 + LoRA 权重 $\theta + \Delta W$（$\Delta W$ 在 rollout 中持续更新）|
|$g_l$|$\Delta W = A \cdot S_e \cdot B$；$S_e \leftarrow \text{Normalize}(S_e + S'_e)$（流式累加）|
|$v$|不存在显式离散检索。"检索"由 cross-attention 在 $q_g$ 与 $K_e, V_e$ 之间完成（可微）|
|$C_N$|当前 cache 内 token 序列 + LoRA 隐式记忆|

> 把长 CoT rollout 的内存爆炸重新刻画为一个 cache-constrained policy optimization 问题。

---

## 2. Training Procedure

- **优化的组件**：主 LLM $\theta$ 的 LoRA 子集 + 进阶编码模块（$W_Q^a, W_K^a, W_V^a$、低秩矩阵 $A, B$、全局 token $h_g$）。基础模型权重冻结。
- **优化算法**：**Cache-aware GRPO**（标准 GRPO 在受限 rollout 分布上的直接套用），见 eq.(3)。
- **训练数据来源**：DAPO-Math-17K 数据集；rollout 在线生成轨迹，outcome reward 由数学题答案正确性提供。
- **是否冻结 LLM 参数**：是。被更新的参数仅包括 LoRA 矩阵（rank=32）和编码模块（$W_Q^a, W_K^a, W_V^a, A, B$ 与 32 个 global tokens）。
- **训练超参**：lr = $1\times 10^{-5}$；max grad norm = 1.0；rollout max length 3072；batch size 512；25% 驱逐率；8×A100 40GB。

**核心训练目标（论文原始 + 框架符号）：**

论文 eq.(3)（cache-aware GRPO）： $$\mathcal{L}^{\mathcal{D}}_{\text{GRPO}}(\theta_g;\theta_{\text{ref}}) = \mathbb{E}_{y\sim \pi^{\mathcal{D}}_{\theta_g}(\cdot \mid p)}\Big[r(y) - \beta, \mathrm{KL}\big(\pi^{\mathcal{D}}_{\theta_g}(\cdot \mid p) ,\Vert, \pi_{\theta_{\text{ref}}}(\cdot \mid p)\big)\Big]$$

框架符号下： $$\mathcal{L}(\theta) = \mathbb{E}_{T \sim \pi_\theta(\cdot \mid q,, m_s\text{-限制})}\Big[r(T) - \beta, \mathrm{KL}\big(\pi_\theta ,\Vert, \pi_{\text{ref}}\big)\Big]$$

其中 $\pi_\theta$ 在 rollout 中按 $g_l$ 流式更新：$\theta \leftarrow \theta + \Delta W$，$\Delta W = A \cdot S_e \cdot B$。

LoRA 增量由驱逐 token 算出： $$\Delta W = A \cdot \underbrace{\big(W_Q^a q_g\big)\big(W_K^a K_e\big)^{\top}\big(W_V^a V_e\big)}_{S_e} \cdot B$$

> backward 不必再展开整个 cache。

---

## 3. Reward Signal

- **奖励类型**：sparse terminal reward（最终答案正确性）。
- **奖励来源**：基于规则的对数学题答案的精确匹配，无 LLM judge、无 human feedback。属于"环境反馈 / 任务对错"。
- **奖励分配**：GRPO 标准做法——同一 prompt 下采 $n$ 条 rollout，组内做 mean/std 归一化，得到 advantage $\hat{A}_i$，按 token 均匀分配到该 trajectory 的所有 token。
- **辅助奖励或正则项**：仅有 GRPO 的 KL 正则项 $\beta \cdot \mathrm{KL}(\pi^{\mathcal{D}}_{\theta_g} ,\Vert, \pi_{\text{ref}})$，无 format/length/process reward 等其他 shaping。

---

## 4. Inference Procedure

- **记忆初始化**：cache 中预先放入 question token（永久保留作 sink）；$S_e$ 用 32 个可学习的 global token $h_g$ 初始化为 $S_e = (W_Q^a q_g)(W_K^a k_g)^\top (W_V^a v_g)$。$\Delta W$ 在解码开始前为 0。
- **每步决策流水线**：
    1. **观测 $o_t$**：当前 cache 中所有 token 的 KV
    2. **生成动作 $a_t$**：用当前 $\theta + \Delta W$ 跑一次 forward，按 $\pi^{\mathcal{D}}_{\theta + \Delta W}(\cdot \mid C_t)$ 采样下一个 token
    3. **cache 更新（$g_s$）**：把新 token 的 KV 入 cache；若 cache 满，按滑窗规则驱逐最旧 25% 的 thinking token，得 $K_e, V_e$
    4. **记忆固化（$g_l$）**：用驱逐的 $K_e, V_e$ 与 $q_g$ 做 cross-attention，得到 $S'_e$；累加 $S_e \leftarrow \text{Normalize}(S_e + S'_e)$；重算 $\Delta W = A \cdot S_e \cdot B$
    5. 回到 1，直到 EOS 或达到 max length
- **额外推理策略**：消融实验里测试了 H2O、PyramidKV、HeadKV 等更智能的 token 驱逐方法（取代滑窗），结论是有提升但训练阶段开销大，未采用；另外测试了不同驱逐率（5%–25%）和不同 global token 数量（0–64）。
- **学习 vs. 手工**：核心更新规则（什么时候驱逐、驱逐多少）是**手工启发式**（滑窗 + 25%）；可学习的部分是 cross-attention 投影矩阵和 global tokens 的内容。$\pi$ 不学"何时驱逐"，只学"如何把已驱逐内容塞进 LoRA"。

> 跨 step 持久但不跨 episode 持久。

---

## 5. RQ 分析

### RQ1（What is memory?）

本文同时使用三种记忆形态：(1) 固定长度滑窗 KV cache（T2）作为可读的 $m_s$；(2) 由 cross-attention 在驱逐 KV 上构造的 latent 摘要 $S_e$（T3）；(3) 把 $S_e$ 进一步通过 $A, B$ 低秩注入 LoRA 权重，使 $\theta$ 流式漂移（T4）。这三层是流水的：$m_s$ 溢出 - $m_z$ 浓缩 - $\theta$ 内化。

### RQ2（Which component is optimized? Which signal is used?）

优化对象是主 LLM 的 LoRA 子参数 + 进阶编码模块（$W_Q^a, W_K^a, W_V^a, A, B, h_g$），属于 O2a（单 LLM 优化）；同时由于编码模块本质是一个学习版的 attention 路径（写到 attention 计算内部、规则可学），也带有 O5（KV cache / 参数化 attention） 的属性。 信号是 rollout 终点的数学答案对错。

### RQ3（Target of Optimization）

两个目标：(1) 数学推理准确率（G1）；(2) 训练阶段 GPU peak/mean memory 与 attention FLOPs，以及推理阶段在小 cache 下的鲁棒性（G2）。其中 G1 是 reward 显式编码的，G2 通过架构 + cache 约束实现，没有进入 reward。

### RQ4（Training Signal and RL Algorithm）

A2: GRPO。

### RQ5（How memory evolves, operates?）

推理时记忆流式演化：每解码若干 token，cache 满 → 驱逐最旧 25% → 计算 $S'_e$ → $S_e$ 归一化累加 → 重算 $\Delta W$ → 用新权重继续解码。$\theta$ 因此不断向"已生成的长 CoT 内容"漂移。一次 episode 结束后 $\Delta W$ 清零，不跨 prompt 持久。$\theta$ 漂移只发生在 inference / rollout 阶段。

---

## Conclusion

这篇论文提出了 Progressive Thought Encoding (PTE)：一种在 RL 训练大推理模型时、面对 KV cache 预算限制的解决方案。常规做法（滑窗 cache）会直接丢弃旧 token，导致长 CoT 推理质量下降；本文反其道而行之，把旧 token 的 KV 先经一次 cross-attention 压成 latent 张量 $S_e$，再以低秩方式累加到 LoRA 权重 $\Delta W$。这样 LoRA 权重在 rollout 中就被驱逐 token 持续"喂养"，模型即使只看到一小段窗口，也能记住早期推理。整个机制嵌进 GRPO，成为 cache-aware GRPO。在三个开源模型 + 六个数学 benchmark 上，相对纯 LoRA + RL 平均 +19.3% 准确率、显存近乎减半，AIME 上 +23.4 绝对分。我认为这可以被当作一种在工作记忆（cache）和参数化记忆（LoRA）之间架桥的流式蒸馏。