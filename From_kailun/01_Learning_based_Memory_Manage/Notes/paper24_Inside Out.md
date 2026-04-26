# Inside Out: Evolving User-Centric Core Memory Trees for Long-Term Personalized Dialogue Systems

**Source:** [https://arxiv.org/abs/2601.05171v2](https://arxiv.org/abs/2601.05171v2) (2026-1-25 修订)

## 符号映射表

| 论文原始符号                                              | 框架符号                             | 含义                          |
| --------------------------------------------------- | -------------------------------- | --------------------------- |
| $H = \lbrace x_1, y_1, ..., x_t, y_t \rbrace$       | $T$ (trajectory) + $C$ (context) | 用户历史对话序列                    |
| $x_t$                                               | $q$                              | 用户输入（user input）            |
| $y_t$                                               | $a$ / $A_n$                      | 系统响应（最终答案）                  |
| $D_t$                                               | 分块的 $q, o$ 序列                    | 第 $t$ 个对话块（dialogue chunk）  |
| $T_t$ (PersonaTree at step $t$)                     | $m_l$                            | 长期结构化用户记忆（树形结构）             |
| $f_{update}$                                        | $g_l$                            | 长期记忆更新函数                    |
| $f_{recall}$                                        | $v$                              | 检索函数（agentic recall mode 下） |
| $f_{gen}$                                           | $\pi_\theta$ on $M$              | 响应生成函数（main LLM）            |
| $O_t = \lbrace ADD, UPDATE, DELETE, NO\_OP \rbrace$ | $a$ (memory action)              | MemListener 输出的原子操作列表       |
| MemListener $\pi_\phi$                              | 独立可插拔模块（policy）                  | 轻量级记忆操作生成器                  |
| $R$ (rule set)                                      | 系统约束                             | Schema、写入范围、操作语法            |
| Schema (Biopsychosocial)                            | $m_l$ 的结构先验                      | 树的 trunk 字段定义               |

## 概览

这篇文章用了用户的多轮长对话历史（按 $w=3$ 分块成 $D_t$）以及一份基于"生物-心理-社会"模型预设的层次化 Schema（树的 trunk）。通过 DeepSeek-R1-0528 离线生成 28K 条 ground-truth 操作序列作为监督数据。  
最后得到了一个轻量级的模型（基于 Qwen2.5-7B / Qwen3-8B 微调），它能把零散的对话块实时编译成结构化的 `{ADD, UPDATE, DELETE, NO_OP}` 树操作，从而在系统中维护一棵不断演化的 PersonaTree（用户画像树）。主 LLM 参数始终冻结。  
这篇文章优化的目标是让下游个性化对话回答的准确率显著提升，同时把所需上下文从 32K tokens 压缩到约 2.2K–2.6K tokens，想证明小模型管记忆 + 大模型做生成是一条低成本、高可靠的路径。

## 1. Problem Setting

- **Memory 类型**：处理 **cross-chat / long-term memory**（$m_l$），即跨会话的用户长期画像。论文显式以 PersonaTree 取代传统的"全部对话拼接"方案。
- **决策建模**：记忆维护被建模为一个**序列化的状态更新过程**——给定上一时刻的树状态 $T_{t-1}$ 和当前对话块 $D_t$，MemListener 输出一组原子操作 $O_t$。这接近一个 **MDP**（state = 当前树状态 + 对话块；action = 操作序列；reward = 过程奖励），但论文未明确写出 MDP 形式。
- **状态空间 $\mathcal{S}$**：当前 PersonaTree 的全部叶节点文本 + 当前对话块 $D_t$ + 规则集 $R$。
- **动作空间 $\mathcal{A}$**：受语法约束的原子操作序列，每个操作 $\in \lbrace ADD, UPDATE, DELETE, NO_OP \rbrace$ × path × value。
- **观测空间 $\Omega$**：与状态空间近似一致（fully observable）。
- **数据结构**：**层次化树**（hierarchical tree）。trunk 由 Biopsychosocial schema 预定义（生物 / 心理 / 性格 / 人口学 / 行为 五大维度），叶节点为描述性字符串，长度受预算约束。

|框架组件|论文实例|
|---|---|
|$m_l$|PersonaTree $T_N$（trunk 固定 + 叶节点字符串）|
|$g_l$|MemListener 生成 $O_t$ + 安全 parser 执行|
|$v$|agentic recall 模式下的查询扩展 + BGE-M3 检索 + reranker|
|$M$|冻结的主 LLM（DeepSeek-V3.1 / Longcat-Flash-Chat / DeepSeek-R1-0528）|
|独立模块|MemListener（Qwen2.5-7B / Qwen3-8B）|

## 2. Training Procedure

- **优化对象**：MemListener（独立于主 LLM 的轻量级 policy 模块），主 LLM 参数完全冻结。
- **训练算法**：两阶段——
    1. **SFT warm-up**：用 HaluMem 13K 全监督微调，让模型先学会语法正确的操作序列；
    2. **RL alignment**：在 GRPO 框架下使用 **DAPO**（Decoupled clip + Dynamic sAmpling Policy Optimization），针对长序列优化设计。
- **训练数据**：HaluMem 和 PersonaMem 的子集，由 DeepSeek-R1-0528 离线生成 ground-truth 操作序列，再人工过滤掉语法错误、路径错误、语义不一致的样本。
- **是否冻结 LLM**：**冻结主 LLM**；MemListener 本身是 7B–8B 的小模型，**全参数微调**（full fine-tuning, bfloat16）。

**核心训练目标**：

SFT 阶段（论文公式）：

$$\mathcal{L}_{SFT}(\theta) = -\frac{1}{\tau} \sum_{t=1}^{\tau} \log P_\theta(o_t \mid o_{< t}, s)$$

其中 $s$ 是输入 context（对话块 + 上一棵树 + 规则约束），$o$ 是 ground-truth 操作序列。

RL 阶段（DAPO 目标）:

$$J_{RL}(\theta) = \mathbb{E}_{(s, y^\star) \sim \mathcal{D},\ \lbrace y_i \rbrace \sim \pi_{\theta_{old}}} \Big[ \frac{1}{\sum_{i=1}^G \lvert y_i \rvert} \sum_{i=1}^G \sum_{t=1}^{\lvert y_i \rvert} \min\big(r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\varepsilon_{low}, 1+\varepsilon_{high}) \hat{A}_{i,t}\big) \Big]$$

其中 importance ratio 与 advantage：

$$r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid s, y_{i,< t})}{\pi_{\theta_{old}}(y_{i,t} \mid s, y_{i,< t})}$$

$$\hat{A}_{i,t} = \frac{R_i - \text{mean}\lbrace R_j \rbrace_{j=1}^G}{\text{std}\lbrace R_j \rbrace_{j=1}^G}$$

约束条件防止全对/全错的 group degeneration：

$$0 < \lvert \{ y_i \mid \text{is\_equivalent}(y^*, y_i) \} \rvert < G$$

参数：$G = 8$，$\varepsilon_{low} = 0.2$，$\varepsilon_{high} = 0.28$（非对称 clipping，给低概率探索 token 更大上调空间），$\beta_{KL} = 0.001$。

## 3. Reward Signal

- **奖励类型**：**sequence-level dense reward**（每个 sampled output 都获得一个连续标量分数 $R_i \in [-1, 1]$），通过 within-group 标准化分配到 token 级别。
- **奖励来源**：**LLM-as-a-judge**——使用 Qwen3-32B (reasoning mode) 作为判别器，对 Pred_Ops 与 GT_Ops 的整体质量打分。
- **奖励分配机制**：sequence-level 评分先经 within-group 标准化得到 $\hat{A}_{i,t}$，再通过 token-level policy gradient 在序列内均匀分摊（DAPO 的 token-level credit assignment）。
- **奖励语义层级（论文 prompt 中给出的 anchor）**：1.0（近乎完美）→ 0.7（高质量）→ 0.5（中等）→ 0.3 → 0.0 → -0.3 → -0.5 → -0.7 → -1.0，judge 在 anchor 之间细粒度插值。
- **辅助项**：未引入显式 format reward 或 length penalty；语法约束通过 SFT warm-up 和 parser 安全门保证，不进入 reward 设计。

## 4. Inference Procedure

- **记忆初始化**：从 schema 加载初始 PersonaTree $T_0$，所有叶节点为空字符串或默认占位符。
- **每步流程（核心闭环）**：
    1. 把用户对话历史 $H$ 按 $w=3$ 分块为 $(D_1, ..., D_N)$；
    2. 对每个 $D_t$，**MemListener 接收 $(D_t, T_{t-1}, R)$，直接生成操作列表 $O_t$**（论文实验证明 direct generation 比 extract-then-transform 更优）；
    3. 安全 parser 执行 $O_t$ 得到 $T_t$；
    4. 持久化为 versioned JSON。
- **响应生成（adaptive 双模式）**：
    - **Fast mode（PersonaTree-only）**：直接把 $T_N$ 的非空叶节点拼接到 query 后，单次生成（低延迟场景）；
    - **Agentic recall mode（PersonaTree-ALL）**：基于 $T_N$ 把原始 query $q$ 扩展成 $K$ 个子查询 $\lbrace \tilde{q}^{(k)} \rbrace$，并行检索 + reranker 融合，最后用 $[q, T_N, C]$ 生成答案；
    - **Router mode**：用一个 gate 决定是否触发 agentic 模式，在性能和成本间折中。
- **是否完全 learned**：MemListener 的写入策略是 RL 学到的；但 trunk schema、操作语法、parser 安全规则、路由触发条件仍是**手工设计的**。检索器（BGE-M3）和 reranker（BGE-Reranker-Large）也是预训练即用，未联合微调。

## 5. RQ 分析

### RQ1 (What is memory?)

PersonaTree 是 cross-chat 的长期记忆 $m_l$，本质上是层次化、可 CRUD 的非参数化外部记忆（T1），但 trunk 受预定义 schema 约束、叶节点受长度预算约束，因此同时带有固定结构压缩（T2）的特征。而训练出来的小模型 MemListener 是独立于主 LLM 的可插拔模块（T5）。

### RQ2 (Which component is optimized?)

优化的是 MemListener 这一独立可插拔记忆管理模块（O1），主 LLM 完全冻结。训练信号来自 LLM-as-a-judge 对操作序列质量的打分。

### RQ3 (Target of Optimization)

主目标是写入决策质量（G4，通过 LLM judge 打分对 Pred_Ops vs GT_Ops 的语义对齐）和回答准确度（G1，通过下游 PersonaMem benchmark 上的 7 类 QA accuracy）。同时效率（G2）也是核心卖点：把 32K 上下文压缩到 2.2K–2.6K，端到端 token 节省超过 90%。

### RQ4 (Training Signal and RL Algorithm)

采用 DAPO 算法（属于 GRPO 家族 / A2），在 SFT warm-up 之上做 process-reward RL。关键设计是 token-level credit assignment + 非对称 clipping ($\varepsilon_{low}=0.2$, $\varepsilon_{high}=0.28$) + dynamic resampling 防止 group 退化。

### RQ5 (How memory evolves, operates?)

记忆通过 MemListener 实时编译对话块为原子操作 $\lbrace ADD, UPDATE, DELETE, NO\_OP \rbrace$ 演化，每步对树做版本化更新；冲突解决在 generation 阶段由 LLM 完成；推理时支持 fast / agentic / router 三种读取策略。

## Conclusion

Inside Out 提出了一个面向长期个性化对话的记忆系统：以生物-心理-社会模型预定义一棵 PersonaTree 作为用户画像的结构骨架，然后训练一个轻量级的 MemListener（7B–8B 小模型）把不断到来的对话块实时编译成 `{ADD, UPDATE, DELETE, NO_OP}` 这四种原子树操作，从而维护一棵随交互演化的长期记忆。训练采用两阶段 SFT + DAPO process-reward RL，reward 来自 LLM judge 对操作序列与 ground-truth 的语义对齐打分。实验证明，把 32K 的对话拼接压缩到约 2.5K 的结构化树记忆后，下游回答准确率不降反升（最高达 76.06，超过 ALL Dialogue 的 64.86）。