# Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory

**Source:** [arXiv:2603.04257v1](https://arxiv.org/abs/2603.04257) (2026-03-04)

---
## 符号映射表

|论文原始符号|含义|框架符号|
|---|---|---|
|$u$|User task instruction|$q$|
|$m_0$|System prompt (fixed, never compressed)|属于 $C$ 中的固定前缀|
|$M$ / $M_{\text{work}}$|Agent context window / working context|$C$ / $C_N$（working portion）|
|IndexedSummary $\sigma = (s, \mathcal{I})$|压缩后的 in-context 状态（progress state + index map）|$m_s$（短期/工作记忆，固定长度压缩）|
|$\mathcal{D}: \text{index} \mapsto \text{content}$|External key-value experience store|$m_l$（长期记忆，带显式索引的 KV 存储）|
|ReadExperience(index)|通过 index 反引用取回 archived block|$v$（检索算法，此处是精确 key 查找而非相似度检索）|
|CompressExperience(...)|压缩 working context 并写入 $\mathcal{D}$|写操作 $g_l$ + 短期重写 $g_s$|
|$z_t$|第 $t$ 步的 thinking / reasoning trace|$t$ (thought)|
|$c_t$|第 $t$ 步的 tool call|$a$ (action)|
|$o_t$|Tool / environment output|$o$ (observation)|
|$\pi_{\text{agent}}$ / $\pi_\theta$|Agent policy|$\pi$|
|$\pi_{\text{IEM}}$|Memex policy（理论分析中）|$\pi$|
|$\theta$|被更新的模型参数（Qwen3-30B backbone）|$\theta$|
|$R$|Episode-level scalar return|reward $R$|
|$A^{(g)}$|Group-relative normalized advantage|advantage|
|$T_{\max}$, $\tau$|最大步数 / context 惩罚阈值|超参数|
|Trajectory segments ${S_0, \dots, S_k}$|按压缩边界切分的子轨迹|$T$（被切段的 trajectory）|
|$A_N$ / Finish($y$)|最终答案|$A_N$|

---

## 概览

这篇文章用的是 agent 在长时间任务中产生的完整 tool-use trajectory（thinking、tool call、tool output 一连串的历史），以及一个外挂的数据库来存原始证据。训练信号来自 agent 自己在环境里 rollout 出来的多条轨迹外加一个带权重的多项奖励。  
最后得到的是一个被微调过的主 LLM，它学会了在 agent loop 中把 CompressExperience 和 ReadExperience 当作工具来用，也可以说得到的是一个会自己管理长期记忆的 agent 策略 $\pi_\theta$。  
主要优化的是 "在有限 context budget 下完成长时任务"：既要提高任务成功率，又要压住 working context 的峰值长度、减少冗余工具调用、保证格式正确。文章中的最终指标是 task success rate 提升 + peak working context 下降。

---

## 1. Problem Setting

- **记忆类型**：处理的是 **in-chat 的长时 agent 记忆**，同时包含 $m_s$（in-context 的 IndexedSummary $\sigma$）和 $m_l$（off-context 的 external store $\mathcal{D}$）。$m_s$ 被严格约束在 $\tau_\sigma$ 以内（ALFWorld 实验中 summary 被截断为 300 tokens，整个 working context 阈值为 8K），$m_l$ 则保存 full-fidelity 原始证据。
- **决策过程建模**：建模为一个 **POMDP 式的 multi-turn tool-use decision process**，其中记忆操作（CompressExperience / ReadExperience）与环境工具、Finish 放在同一个动作空间中，属于 first-class actions。
- **状态空间 $\mathcal{S}$**：当前 context window $M = [m_0, u, M_{\text{work}}]$ 加上外部存储 $\mathcal{D}$ 的当前内容。
- **动作空间 $\mathcal{A}$**：$\mathcal{T} = \lbrace \text{CompressExperience}(\cdot), \text{ReadExperience}(\cdot), \text{Finish}(\cdot), \text{OtherTool}(\cdot) \rbrace$——环境工具与记忆工具并列。
- **观测空间 $\Omega$**：包括 tool output $o_t$、从 $\mathcal{D}$ 取回的 archived block、以及一条系统自动注入的 `ContextStatus(M, \tau)` 指示（当前 working tokens 与阈值），把 context 状态"显式化"喂给 agent。
- **记忆的数据结构**：
    - $m_s$ = IndexedSummary $\sigma = (s, \mathcal{I})$，其中 $s$ 是自然语言进度状态，$\mathcal{I} = \lbrace (\text{index}, \text{description}) \rbrace$ 是索引–描述对的有限集合。
    - $m_l$ = $\mathcal{D}: \text{index} \mapsto \text{content}$，是一个显式 KV 数据库，key 是 agent 自己起的稳定字符串（如 `ctx_locations`），value 是完整的原始内容（tool output、代码片段、日志）。**检索不是相似度匹配，而是精确 key 查找**。

**核心组件映射表：**

| 框架组件  | 本文对应                                                                        |
| ----- | --------------------------------------------------------------------------- |
| $q$   | 任务指令 $u$（ALFWorld 家务任务）                                                     |
| $C_N$ | 推理时实际喂给 LLM 的 $M = [m_0, u, M_{\text{work}}]$                               |
| $m_s$ | IndexedSummary $\sigma$（压缩后替换整个 $M_{\text{work}}$）                          |
| $m_l$ | External store $\mathcal{D}$                                                |
| $v$   | ReadExperience：按 key 精确查找 $\mathcal{D}[\text{index}]$                       |
| $g_s$ | CompressExperience 生成新的 $\sigma$                                            |
| $g_l$ | CompressExperience 写入 $\mathcal{D}[\text{index}] \leftarrow \text{content}$ |
| $\pi$ | $\pi_\theta$（主 LLM，被 RL 优化）                                                 |
| $T$   | agent loop 产生的 $(z_t, c_t, o_t)$ 序列，按 compress 切段                           |

---

## 2. Training Procedure

- **优化对象**：直接训练主 LLM 的参数 $\theta$（Qwen3-30B-A3B-Thinking-2507），LLM 本身承担所有角色——思考、工具调用、压缩、检索、回答。**没有独立的 memory manager 模型，也没有冻结 LLM**。
- **优化算法**：**GRPO 的变体 MemexRL**，带 PPO-style clip 和 KL 惩罚。
- **训练数据来源**：**在线交互**——每个 prompt 从当前策略采样 $G=8$ 个 rollout，每个 rollout 是完整的 agent-环境交互轨迹；先用 supervised 示范 warm-start，再用 RL 精炼。
- **是否冻结 LLM**：**不冻结**，优化的就是 LLM 本体的权重（使用 INT4 QAT：前向 INT4、梯度 BF16 累积）。
- **核心训练目标函数**：

Episode-level return（论文式 (1)）：

$$R = R_{\text{task}} - P_{\text{context}} - P_{\text{redundancy}} - P_{\text{format}}$$

Group-relative advantage（论文式 (2)）：

$$A^{(g)} = \frac{R^{(g)} - \text{mean}\lbrace R^{(h)} \rbrace_{h=1}^{G}}{\text{std}\lbrace R^{(h)} \rbrace_{h=1}^{G} + \epsilon}$$

PPO-clipped surrogate with KL（论文式 (3)，统一符号后）：

$$\max_\theta ; \frac{1}{G} \sum_{g=1}^{G} \sum_t \min\Big( r_t^{(g)} A^{(g)}, ; \text{clip}(r_t^{(g)}, 1-\eta, 1+\eta) A^{(g)} \Big) - \beta , D_{\text{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$$

其中 $r_t^{(g)} = \pi_\theta(a_t^{(g)} \mid s_t^{(g)}) / \pi_{\theta_{\text{old}}}(a_t^{(g)} \mid s_t^{(g)})$。

- **Segmented Trajectory Processing**（关键训练技巧）：如果 trajectory 中出现了 $k$ 次压缩，就切成 $k+1$ 段 $\lbrace S_0, \dots, S_k \rbrace$，每段独立 tokenize 和优化，但所有段共享同一个 terminal reward $R$。这样可以让"早期的压缩决策"通过 group-relative advantage 拿到来自后续段落的 credit。

---

## 3. Reward Signal

- **奖励类型**：**sparse terminal reward + dense step-level shaping penalties 混合**。$R_{\text{task}}$ 是终局稀疏信号（任务成功/失败），三项惩罚则是从整条轨迹聚合而来的 dense shaping。
- **奖励来源**：
    - $R_{\text{task}}$：**环境反馈**（ALFWorld 判定任务是否完成）。
    - $P_{\text{context}}$：程序化计算——对每一步 working context token 数超过阈值 $\tau$ 的部分累加再归一化：

$$P_{\text{context}} = \min\Big(1, ; \frac{\sum_{t=1}^{T} \max(0, C_t - \tau)}{\tau \cdot T}\Big)$$

- $P_{\text{redundancy}}$：程序化检测相同 (tool_name, arguments) 签名的冗余调用：

$$P_{\text{redundancy}} = \frac{N_{\text{redundant}}}{N_{\text{tool\_call}}}$$

- $P_{\text{format}}$：程序化检测 tool_call tag/JSON 格式错误：

$$P_{\text{format}} = \frac{N_{\text{malformed}}}{N_{\text{tool\_call}}}$$

- **奖励分配到步骤/token**：全部 reward 聚合为一个 **episode-level 标量 $R^{(g)}$**，通过 GRPO 的 group-relative advantage 分到组内每个 rollout 的每个 token；对于被切段的 trajectory，所有段共享同一个 $R$，通过"共享 terminal reward + segmented context"的方式实现延迟压缩决策的 credit assignment。
- **辅助奖励/正则项**：三项 penalty 全部是辅助 shaping；此外 PPO clip + KL 对参考策略 $\pi_{\text{ref}}$ 起正则作用。
- **没有使用 LLM-as-judge 或人类反馈**，这一点和 Memory-R1（用 EM/判断）或 Just in Time（LLM judge）不同。所有信号都是程序化/环境化的。

---

## 4. Inference Procedure

- **记忆初始化**：$M \leftarrow [m_0, u]$（system prompt + task instruction，永不压缩），$\mathcal{D} \leftarrow \emptyset$，$\text{answer} \leftarrow \emptyset$。$m_0$ 和 $u$ 作为两个 anchor 被保留在整个 episode。
- **每步决策流水线**（Algorithm 1）：
    1. 系统自动把 `ContextStatus(M, \tau)` 追加到 $M$（告诉 agent 当前 working tokens 与阈值）。
    2. $(z_t, c_t) \leftarrow \pi_\theta(M)$：agent 输出 thinking 和 tool call。
    3. 根据 $c_t$ 分支执行：
        - **CompressExperience**：把 MemoryBlocks 写入 $\mathcal{D}$，然后 $M \leftarrow [m_0, u, \text{IndexedSummary}]$（整个 working context 被 $\sigma$ 替换）。
        - **ReadExperience(index)**：$o_t \leftarrow \mathcal{D}[\text{index}]$，然后 $M \leftarrow M \oplus [o_t]$（取回的内容作为一条新消息追加）。
        - **Finish($y$)**：设置 $\text{answer} \leftarrow y$ 并返回。
        - **其他环境工具**：执行后 $M \leftarrow M \oplus [o_t]$。
    4. 回到步骤 1，直到 Finish 或达到 $T_{\max}$。
- **推理时额外策略**：
    - **Soft triggering**（软触发压缩）：系统通过 `ContextStatus` 把 context 使用情况暴露给 agent，当 working > threshold 时还会注入额外警告，但**不强制压缩**——由 $\pi_\theta$ 自己决定何时调用 CompressExperience。
    - **Anchor-based extraction**：CompressExperience 的 db_block 内容支持两种模式：(a) 模型直接写内容；(b) 提供 (start_anchor, mid_anchor, end_anchor) 三锚点让系统从当前对话中精确抽取 verbatim 片段（mid_anchor 作校验点防假阳性）。这是一种**减少 regeneration 成本**的设计。
    - **TopK / 温度**：论文未提及推理时调 top-k 或温度。
- **推理策略是完全由学习得到的 $\pi_\theta$ 驱动，还是仍有手工规则？**：**主要由学习到的 $\pi_\theta$ 驱动**。记忆何时写、写什么、索引怎么起名、何时读——全部由 $\pi_\theta$ 决定。仅有的手工规则是：(i) $m_0$ 和 $u$ 永不压缩；(ii) 系统自动注入 ContextStatus 与警告文本；(iii) anchor extraction 的校验逻辑。

---

## 5. RQ 分析

### RQ1 (What is memory?)

Memex 构造了一种显式记忆：in-context 的 IndexedSummary $\sigma$（固定长度压缩的 $m_s$）+ off-context 的 key-value store $\mathcal{D}$（纯文本非参数 $m_l$）。两层通过稳定索引严格绑定，检索是精确 key 查找而非相似度。

### RQ2 (Which component is optimized? Which signal is used?)

优化的是主 LLM $\pi_\theta$ 本身的参数（O2a 单 agent LLM optimization），使用 GRPO + segmented trajectory processing。信号是混合 reward：task success（环境）+ context overflow penalty（程序化）+ redundancy penalty（程序化）+ format penalty（程序化），全部聚合到 episode-level terminal reward 然后通过 group-relative advantage 分配。

### RQ3 (Target of Optimization)

追求在固定 context budget 下同时最大化 task success 并最小化 working context 峰值、冗余调用和格式错误。衡量"记忆用得好"的指标包括：(a) 任务成功率 (b) peak working context tokens (c) CompressExperience / ReadExperience 的调用模式（更少压缩、更多精确检索）。

### RQ4 (How memory evolves, operates?)

记忆按 agent 决策动态演化：每次 CompressExperience 会彻底重写 $\sigma$（全量替换而非增量追加）并把原始内容沉淀到 $\mathcal{D}$；$\mathcal{D}$ 则只增不减（paper 未提到删除操作），通过稳定索引被后续 ReadExperience 精确解引用。"何时压、压什么、读什么"三个决策全部由学习到的 $\pi_\theta$ 在每一步生成 tool call 时决定。

---

## Conclusion

这篇文章解决的是 LLM agent 在长时任务里 context 爆炸的问题，给出的答案是要求 agent 自己把历史切成两份：一份是短而结构化的索引清单（留在 context 里），另一份是原始证据（扔到外部 KV 库、用稳定 key 命名）。当以后要用的时候，agent 通过精确的索引解引用把原始证据拉回来。作者把压缩和取回都当成工具，和环境工具一起放进动作空间，再用 GRPO 训 Qwen3-30B 去学"什么时候压、怎么压、什么时候读"。为了让训练信号能穿过压缩边界，他们把 trajectory 按压缩切段、共享 terminal reward。最后在一个加难过的 ALFWorld 上，任务成功率从 24% 涨到 86%，峰值 working context 从 17K 降到 9.6K。