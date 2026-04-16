# Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval (RF-Mem)

**Source:** [https://arxiv.org/abs/2603.09250](https://arxiv.org/abs/2603.09250) (2026-03-10, Accepted by ICLR 2026)

---

## 符号映射表

| 论文原始符号                                           | 框架符号                        | 含义                                    |
| ------------------------------------------------ | --------------------------- | ------------------------------------- |
| $q$ / $q_t$                                      | $q$                         | 用户查询                                  |
| $M = \lbrace m_1, \ldots, m_M \rbrace$           | $m_l$                       | 用户长期记忆库（所有历史记忆条目）                     |
| $z_i = \phi(m_i)$                                | $m_l$ 的 embedding 表示        | 记忆条目的向量编码                             |
| $C_t$ = Top-K retrieved set                      | $v(q, m_l)$                 | 检索算法返回的相关记忆子集                         |
| $s_i = \langle x_t, z_i \rangle$                 | 检索相似度得分                     | cosine similarity                     |
| $\bar{s}$, $H(p)$                                | 熟悉度信号                       | 均值得分与熵，用于路径选择                         |
| $\alpha$-mix recollect query                     | 无直接对应，提议符号 $q_{\text{rec}}$ | Recollection 路径中通过 $\alpha$ 混合生成的扩展查询 |
| $R$ (max rounds), $B$ (beam width), $F$ (fanout) | 无直接对应                       | Recollection 路径的搜索预算参数                |
| LLM (answer generator)                           | $M$                         | 主模型，用于最终回答生成                          |

---

## 概览

这篇文章用了用户的历史对话记忆，以及一个预训练的 dense retriever 来编码和检索这些记忆。    
最后得到了一个即插即用的双路径检索框架，不需要训练任何模型参数，可以直接替换现有 RAG 系统中的检索模块，根据熟悉度信号自适应地在快速检索和深度检索之间切换。    
优化了个性化记忆检索的 Recall 和下游回答的 Accuracy，同时在延迟和 token 消耗上保持接近 one-shot 检索的效率。但这些仅用于事后评估，不作为 reward 反馈到任何学习循环中。  

---

## 1. Problem Setting

- **记忆类型**：Cross-chat memory（跨会话的用户长期记忆），对应 $m_l$。用户的历史对话、偏好、背景信息被存储为外部记忆条目集合。不涉及 $m_s$（无显式的 working memory / scratchpad）。
- **决策过程建模**：**未建模为 MDP / POMDP / bandit**。RF-Mem 是一个纯推理时的启发式检索算法，路径选择通过阈值规则（公式 3）完成，不涉及学习型决策过程。
- **状态空间 $\mathcal{S}$**：不适用（无 MDP 建模）
- **动作空间 $\mathcal{A}$**：不适用（无 MDP 建模）
- **观测空间 $\Omega$**：不适用
- **记忆的数据结构**：向量库。每条记忆 $m_i$ 被编码为 embedding $z_i = \phi(m_i)$，存储在向量索引中，通过 cosine similarity 检索。

|核心组件|框架符号|本文对应|
|---|---|---|
|长期记忆|$m_l$|用户记忆向量库 $M = \lbrace m_1, \ldots, m_M \rbrace$|
|短期记忆|$m_s$|不适用|
|检索算法|$v$|RF-Mem 双路径检索（Familiarity top-K / Recollection 迭代扩展）|
|主模型|$M$|LLM（GPT-4o-mini 等，冻结）|
|策略|$\pi$|不适用（无学习型策略）|

> RF-Mem 的核心创新在于检索策略 $v$ 的设计，而非记忆表征本身。记忆仍然是标准的 embedding 向量库，与 Dense Retrieval baseline 完全相同。

---

## 2. Training Procedure

**本文没有训练过程。** RF-Mem 是一个完全基于规则的推理时检索算法：

- **优化的组件**：无。LLM 参数冻结，retriever（Contriever）参数冻结，无辅助模型被训练。
- **优化算法**：无。路径选择通过手工设定的阈值 $\theta_{\text{high}}$、$\theta_{\text{low}}$、$\tau$ 完成；Recollection 路径通过 KMeans 聚类 + $\alpha$-mix 完成，均为确定性算法。
- **训练数据来源**：无。
- **是否冻结 LLM 参数**：是。
- **核心训练目标函数**：不适用。

---

## 3. Reward Signal

这篇文章没有使用任何 reward signal，因为不存在训练过程。

---

## 4. Inference Procedure

- **记忆初始化**：用户的所有历史对话记忆被预先编码为 embedding 向量并存入向量索引（如 FAISS）。
    
- **每步决策流程**：
    
    1. **Probe 检索**：给定查询 $q$，编码为 $x_t = \phi(q)$，对记忆库执行 top-K 相似度检索，得到候选集及得分 $\lbrace s_i \rbrace$
    2. **熟悉度判断**：计算均值 $\bar{s}$ 和熵 $H(p)$（公式 1-3），决定走 Familiarity 还是 Recollection 路径
    3. **Familiarity 路径**：直接返回 probe 检索的 top-K 结果
    4. **Recollection 路径**：
        - 对 probe 结果做 KMeans 聚类，得到 $B$ 个质心 $g_b^{(r)}$
        - 通过 $\alpha$-mix 将每个质心与原始查询混合，生成 recollect query：$x_b^{(r+1)} = \text{norm}(\alpha x^{(r)} + (1-\alpha) g_b^{(r)} + x_t)$
        - 用 recollect query 执行新一轮检索
        - 重复 retrieve-cluster-mix 循环，最多 $R$ 轮
        - 最终返回所有轮次候选的 top-K 并集
    5. **回答生成**：将检索到的记忆文本拼入 prompt，由 LLM 生成最终回答
- **推理时额外策略**：Beam width $B$、fanout $F$、最大轮数 $R$ 控制搜索预算；$\alpha$ 控制查询与质心的混合比例；阈值 $\theta_{\text{high}}$、$\theta_{\text{low}}$、$\tau$ 控制路径切换。
    
- **策略来源**：**完全由手工规则驱动**，无学习得到的 $\pi$。
    

> Recollection 路径的 retrieve-cluster-mix 循环是本文的核心贡献，它在 embedding 空间中模拟了人类的"链式回忆"过程，无需调用 LLM 做 query reformulation，因此延迟很低。

---

## 5. RQ 分析

### RQ1 (What is memory?)

RF-Mem 中的记忆是用户历史对话的 embedding 向量集合，属于T1。记忆条目以文本形式存储，通过 dense encoder 编码为向量后存入向量索引，支持 cosine similarity 检索。记忆本身的数据结构与标准 Dense Retrieval 完全相同，RF-Mem 的创新在于检索策略而非记忆表征。

### RQ3 (Which component is optimized?)

本文未涉及此问题。RF-Mem 没有训练/优化过程。所有组件（LLM、retriever、阈值参数）均为预设或手工调整。

### RQ4 (Target of Optimization)

本文未涉及此问题。评估时使用了 Accuracy 和 Recall@K 作为指标，但这些仅用于事后评估，不作为 reward 反馈到任何学习循环中。

### RQ4 (How memory evolves, operates?)

记忆在运行时是只读的：RF-Mem 不涉及记忆的写入、更新或删除操作。核心创新在于读取方式，即通过熟悉度信号自适应切换 Familiarity（one-shot top-K）和 Recollection（迭代 cluster-mix 扩展）两条检索路径。Recollection 路径通过在 embedding 空间中混合查询与聚类质心来"重构"相关记忆链，模拟人类的深度回忆过程。

---

## Conclusion

RF-Mem 是一个受认知科学"Recollection-Familiarity 双过程理论"启发的个性化记忆检索框架。它的核心思想是：当查询与记忆库的匹配度高（熟悉）时，直接用标准 top-K 检索；当匹配度低（不熟悉）时，通过 KMeans 聚类和 $\alpha$-mix 查询扩展在 embedding 空间中迭代地"回忆"相关记忆链。整个框架不需要任何模型训练，完全是推理时的算法设计，因此可以即插即用地替换现有 RAG 系统的检索模块。实验表明 RF-Mem 在三个个性化记忆 benchmark 上一致优于 one-shot 检索和 full-context 方法，同时保持接近 one-shot 的延迟。其主要局限在于：所有超参数（阈值、$\alpha$、beam width 等）需要手工调整，且记忆在推理过程中是只读的，不支持动态更新。