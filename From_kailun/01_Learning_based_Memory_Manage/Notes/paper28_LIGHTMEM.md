# LightMem: Lightweight and Efficient Memory-Augmented Generation

**Source:** [arXiv:2510.18866v4](https://arxiv.org/abs/2510.18866) （2026-2-28 ,已被 ICLR 2026 接收）

---

## 符号映射表

|论文原始符号 / 表达|框架符号|说明|
|---|---|---|
|原始对话数据 $D$|输入流 $\lbrace o_t \rbrace$|用户–模型交互历史，按 turn 增量进入系统|
|分段后的数据 $D^{(g)} = f_{\text{seg}}(D;g)$|分段后的 $\lbrace o_t \rbrace$|turn / session / topic 三种粒度|
|内存银行 $\mathcal{M}$|$m_l$|长期记忆库（向量索引 + 元数据）|
|单条记忆条目 Entry$_i$ = $\lbrace$topic, $e_i$, user$_i$, model$_i\rbrace$|$e_i \in m_l$|以 topic 为粒度的条目|
|感觉记忆 buffer（512 tokens）|$m_s^{\text{sens}}$|第一级 working memory，存放 pre-compressed tokens|
|短期记忆 buffer（$th$ tokens）|$m_s$|第二级 working memory，存放 topic 分组后的内容|
|摘要函数 $f_{\text{sum/extract}}$|$g_l$（写入侧）|触发条件：$\lvert m_s \rvert \geq th$|
|更新函数 $f_{\text{update}}$|$g_l$（合并侧）|离线 sleep-time 并行执行|
|检索函数 $f_{\text{retrieve}}$|$v$|余弦相似度 + 时间戳约束|
|对话生成函数 $f_{\text{chat}}$|$M$ / $\pi_\theta$|主 LLM（GPT-4o-mini / Qwen3-30B / GLM-4.6）|
|压缩模型 $\theta$（LLMLingua-2）|$M_{\text{pre}}$（**新提议符号**）|独立的预压缩模块，提供 token 保留概率与 attention 矩阵|
|嵌入模型（all-MiniLM-L6-v2）|$\phi$|提供 $e_i = \phi(\text{sum}_i)$|
|拓扑边界 $\mathcal{B} = \mathcal{B}_1 \cap \mathcal{B}_2$|—|用 attention 局部峰 + 语义相似度联合切分|
|更新队列 $Q(e_i)$|$g_l$ 的中间结构|每条 LTM 条目维护一个候选更新源队列|

---

## 概览

这篇文章用了对话历史的原始 turn 序列作为输入，借助一个轻量化的预训练压缩器和它的 attention 矩阵，把冗余 token 滤掉、把语义连续的 turn 聚成 topic，再用主 LLM 把每个 topic 摘要成长期记忆条目。更新过程被搬到"离线 sleep-time"做并行执行。
最后得到了一个完全 training-free 的三级记忆 pipeline（感觉记忆 → 短期记忆 → 长期记忆），可以即插即用地配在任何主 LLM 之上，没有任何参数被训练或微调。
这篇文章追求的目标不是训练信号意义上的优化，而是在维持甚至提升QA 准确率的前提下，把记忆构建阶段的 token 消耗、API 调用数、运行时延 大幅压低。

---

## 1. Problem Setting

- **记忆类型**：cross-chat（跨 session 的长程对话记忆），同时具备 $m_s$（双层 buffer）和 $m_l$（topic-indexed 向量库）。
- **决策过程建模**：**不建模为 RL 决策过程**。整个记忆构建是一个由阈值触发的确定性 pipeline——感觉 buffer 满 → 触发 topic 切分；STM buffer 满 → 触发摘要；更新触发器到达 → 触发离线并行更新。无 MDP / POMDP / bandit 形式化。
- **数据结构**：
    - $m_s^{\text{sens}}$：固定容量 token buffer（512 tokens）
    - $m_s$：固定容量 token buffer（$th \in \lbrace 256, 512, 768, 1024 \rbrace$）
    - $m_l$：向量库 + 元数据（topic 标签、嵌入、原始 user/model turn、timestamp）
- **核心组件与符号映射**：

| 组件              | 论文表达                       | 框架符号             | 备注                                 |
| --------------- | -------------------------- | ---------------- | ---------------------------------- |
| Pre-compressor  | $f_{\text{pre\_compress}}$ | $M_{\text{pre}}$ | LLMLingua-2，frozen                 |
| Topic segmenter | $f_{\text{topic}}$         | —                | attention $\cap$ similarity，frozen |
| Summarizer      | $f_{\text{sum/extract}}$   | $g_l$（写入分支）      | 主 LLM 直接 prompt，frozen             |
| Updater         | $f_{\text{update}}$        | $g_l$（合并分支）      | 主 LLM 直接 prompt，offline parallel   |
| Retriever       | $f_{\text{retrieve}}$      | $v$              | 余弦相似度 + $t_j \geq t_i$ 约束          |
| Chat            | $f_{\text{chat}}$          | $\pi_\theta$     | 主 LLM，frozen                       |

> 仍然清楚地区分了 $m_s / m_l$ 的角色。

---

## 2. Training Procedure

**本文不涉及任何形式的训练。**

- 优化的组件：**无**。所有模型（LLMLingua-2、all-MiniLM-L6-v2、GPT-4o-mini / Qwen3-30B / GLM-4.6）在论文中始终冻结。
- 优化算法：**无**（既非 SFT、也非 RL、也非蒸馏）。
- 训练数据：**无**。
- LLM 是否冻结：**全部冻结**。
- 目标函数：**无可训练目标**。

论文中出现的"目标"仅是 pipeline 内部的判定式：

$$\hat x = \lbrace x_i \in x \mid P(\text{retain}\ x_i \mid x;\theta) > \tau \rbrace$$

$$\mathcal{B}_1 = \lbrace k \mid M_{k,k-1} > M_{k-1,k-2},\ M_{k,k-1} > M_{k+1,k},\ 1 < k < n \rbrace$$

$$\mathcal{B}_2 = \lbrace k \mid \text{sim}(s_{k-1}, s_k) < \tau,\ 1 \leq k < n \rbrace,\ \mathcal{B} = \mathcal{B}_1 \cap \mathcal{B}_2$$

这些是**推理时的硬决策规则**，不是被优化的损失。

---

## 3. Reward Signal

**本文未涉及此问题。** 没有训练过程，自然没有 reward。

唯一与"评估信号"沾边的是 **LLM-as-judge 的 ACC 指标**（GPT-4o-mini 作为裁判判定 QA 正确与否），但它仅用于**离线评测**，没有反馈回任何参数或策略。

---

## 4. Inference Procedure

### 4.1 推理时记忆初始化

冷启动时 $m_l = \emptyset$，$m_s^{\text{sens}} = m_s = \emptyset$。随着对话 turn 的增量到来，三级 buffer 依次填充并触发各自的下游操作。

### 4.2 每步流水线

对每条 turn $o_t$（user + model 配对消息）：

1. **预压缩**：$\hat o_t = M_{\text{pre}}(o_t; r)$，按保留率 $r \in [0.4, 0.8]$ 删除低信息 token（基于 retention 概率或交叉熵）。
2. **写入感觉 buffer**：$m_s^{\text{sens}} \leftarrow m_s^{\text{sens}} \cup \lbrace \hat o_t \rbrace$。
3. **触发条件 1**：若 $\lvert m_s^{\text{sens}} \rvert \geq 512$，则计算 attention 矩阵 $M$ 与相邻 turn 相似度，得 $\mathcal{B} = \mathcal{B}_1 \cap \mathcal{B}_2$，把 buffer 切成若干 topic 段，写入 STM。
4. **触发条件 2**：若 $\lvert m_s \rvert \geq th$，则对每个 topic 段调用主 LLM 做摘要 $\text{sum}_i = f_{\text{sum}}(S_i)$，组装 $\text{Entry}_i = \lbrace \text{topic}, \phi(\text{sum}_i), \text{user}_i, \text{model}_i \rbrace$ **soft-insert** 到 $m_l$（仅追加，不修改既有条目）。
5. **离线触发**：到达 sleep-time 信号后，对每个 $e_i$ 计算更新队列：

$$Q(e_i) = \text{Topk}\lbrace (e_j, \text{sim}(v_i, v_j)) \mid t_j \geq t_i,\ j \neq i \rbrace_{:n}$$

各队列**并行**调用 $f_{\text{update}}$ 完成 add / update / merge / ignore 决策。

### 4.3 查询时

新查询 $q$ 到达 → $v(q, m_l)$ 取 top-k 条目 → 与 $q$ 拼接成 $C_N$ → $\pi_\theta$ 生成回答 $\hat a$。

### 4.4 推理策略

完全由**手工触发规则 + 阈值**驱动。无学习得到的策略 $\pi$，无温度调节，无多轮 replan。变量是手工调的 $(r, th)$ 二元组。

> 把传统 memory system 中串行、阻塞、test-time 的更新负担彻底搬到了并行、异步、offline 的位置。这个 trick 是架构层面的，不依赖训练，但收益极大（runtime 数倍下降）。值得未来 RL-based memory 工作借鉴："什么时候更新"和"怎么更新"是两个正交问题。

---

## 5. RQ 分析

### RQ1（What is memory?）

LightMem 把记忆建模为显式的多级 token 缓冲 + 显式的非参数化外部条目库。具体而言：感觉 buffer（512 tokens）+ STM buffer（$th$ tokens）共同实现 $m_s$，而 $m_l$ 是 topic 索引的向量库 $\lbrace (\text{topic}, \phi, \text{user}, \text{model}, t) \rbrace$。两层 buffer 都是固定长度（T2 风格），LTM 是 RAG style。

### RQ2（Which component is optimized? Which signal is used?）

本文未涉及此问题。 

### RQ3（Target of Optimization）

作者通过架构和超参搜索追求 (QA ACC, token cost, API call count, runtime) 的 Pareto 前沿，应可视为隐式的 G1（Accuracy）+ G2（Efficiency）目标。

### RQ4（Training Signal and RL Algorithm）

本文未涉及此问题。 属于 N/A: Non-RL Methods（甚至连 SFT / 蒸馏都没有，纯 inference-time pipeline）。

### RQ5（How memory evolves, operates?）

两条触发链：(1) online soft-insert，感觉 buffer 满触发 topic 分割；STM 满触发摘要并仅追加到 $m_l$，不动既有条目；(2) offline sleep-time consolidation，异步对每条 $e_i$ 用 timestamp-constrained 检索得到候选队列 $Q(e_i)$，并行调用主 LLM 做 add/update/merge/ignore 决策。这种"在线只追加 + 离线再整合"的双阶段演化是本文的核心创新。

---

## Conclusion

LightMem 是一篇受 Atkinson–Shiffrin 人类记忆模型启发的、面向 LLM agent 的纯架构层面记忆系统论文。它不训练任何模型，而是通过三个工程级 trick 显著降本增效：(1) 在记忆构建之前用 LLMLingua-2 做 token 级预压缩，过滤冗余；(2) 用 attention 局部峰 + 语义相似度联合切分 topic，使摘要单元更内聚；(3) 把昂贵的 LLM 更新调用从 test-time 解耦到 sleep-time 并并行执行。在 LongMemEval 和 LoCoMo 上，LightMem 用 frozen 的 GPT-4o-mini / Qwen3-30B / GLM-4.6 做主干，相对 A-MEM、Mem0、MemoryOS 等强基线，准确率最多提升 29.3%，token 用量最多减少 38×（offline 计入）或 117×（online-only），API 调用减少 30–310×。  
很多记忆系统的痛点其实是工程问题而非学习问题，把更新与推理解耦、把过滤前置、把分组对齐到语义边界，这三件事不需要训练就能换来巨大收益。