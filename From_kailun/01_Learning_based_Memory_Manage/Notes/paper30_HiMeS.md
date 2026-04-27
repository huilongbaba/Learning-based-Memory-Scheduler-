# HiMeS: Hippocampus-inspired Memory System for Personalized AI Assistants

**Source:** [arXiv:2601.06152v1](https://arxiv.org/abs/2601.06152) (2026-01-06 修订)

---

## 符号映射表

|论文原始符号 / 概念|框架统一符号|说明|
|---|---|---|
|`query_old` (current user query)|$q$|用户当前输入的原始 query|
|dialogue history|$C$ (历史对话上下文)|进入 rewriter 的最近若干轮对话|
|`query_rewrited` (rewritten query)|$m_s$|短期记忆：被 rewriter 压缩后的精炼 query|
|Partitioned long-term memory store (16 categories ATM)|$m_l$|长期记忆：用户历史 query 按主题分区存储的非参数化数据库|
|Query rewriter (trained LLM)|$\pi_\theta$（gₛ 的实现）|实际承担 short-term 压缩与生成 $m_s$ 的策略模型|
|Embedding-based similarity + ATM 分区路由|$v$|检索算法（既用于 RAG 召回文档，又用于历史 query 召回）|
|Attention-inspired rerank score|$v$ 的二阶段分量|chunk embedding 与历史 query embedding 的相似度均值，用于 rerank|
|Frozen Task LLM (DeepSeek-R1 / V3 / Qwen3 / Kimi-K2)|$M$|冻结的黑盒回答模型，参数 $\theta_M$ 不更新|
|$A_{\text{pred}}$|$\hat{a}_N$|模型最终回答|
|$A_{\text{label}}$|$a^*$|ground-truth 回答|
|$C$ (RAG-retrieved contents)|$C_N$ 中由 RAG 注入的部分|检索到的文档片段|
|HSER reward $= F1_h + \alpha,EM_h + \beta,\text{Hit}$|$r$|RL 阶段的总奖励|

> 论文中引入了 Atomic Topic Modeling (ATM)，用于把 $m_l$ 划分到 16 个一级主题与若干子主题。

---

## 概览

这篇文章用了用户最近的对话历史（短期）和用户过去所有的提问记录（长期）。最近对话被一个 rewriter（专门训练的小模型）压缩成一句精炼的 query；过去所有提问按 16 个主题分门别类地存进一个分区数据库。  
最后得到了一个被训练好的 query rewriter（短期记忆模块）+ 一套带主题分区与 attention-rerank 的检索流水线（长期记忆模块）。这两个组件像插件一样加在任意一个被冻结的回答 LLM（DeepSeek-R1、Qwen3-235B 等）前面，作者强调这是一个 plug-and-play 的记忆层。  
这篇文章优化的是 rewriter 的策略 $\pi_\theta$，但 reward 直接来自下游回答的质量。所以最终被推动的是回答的准确度与与检索内容的对齐度（CA、QA 指标）。

---

## 1. Problem Setting

- **记忆类型**：cross-chat memory（跨会话用户画像）+ in-chat memory（当前对话历史）。同时存在 $m_s$（短期 = rewriter 的压缩输出）与 $m_l$（长期 = 用户历史 query 的分区库）。
- **决策过程建模**：单步的 conditional generation（rewriter 输出一条 rewritten query），通过 GRPO 当作单步 RL 任务训练。**没有显式建模为多步 MDP / POMDP**，因为 trajectory 是 "对话历史 → 一次 rewrite → 一次回答" 的单步生成。
- **状态/观测**：$s = (q, \text{dialogue history})$；动作 $a$ = rewritten query 的 token 序列。**这是一个 contextual bandit 风格的 setting**，而非真正的多步 RL。
- **数据结构**：
    - $m_s$：自然语言条目（一条 rewritten query 字符串）
    - $m_l$：向量库 + 两级层级树（topic / sub-topic），每条记忆是 (历史 query, topic tag, sub-topic tag, embedding)
- **检索机制 $v$**：分两阶段
    1. 用 $m_s$ 在外部知识库做 RAG 召回，得文档集 $D$
    2. 把 $D$ 切 chunk，再用每个 chunk 与 top-$n$ 个相似历史 query 的 embedding 平均相似度作为分数，rerank 取 top-$k$

|组件|论文实现|框架符号|
|---|---|---|
|短期记忆载体|单条 rewritten query 字符串|$m_s$|
|长期记忆载体|16 主题分区的历史 query 向量库|$m_l$|
|短期更新|rewriter $\pi_\theta$ 端到端生成|$g_s$|
|长期写入|ATM 分类器打 tag → embed → 存入对应分区|$g_l$|
|检索|embedding 相似度 + ATM 路由 + attention rerank|$v$|
|决策主体|训练 rewriter；response LLM 冻结|$\pi_\theta$|

> 记忆调度实际上发生在 inference pipeline 里，不是被学到的。

---

## 2. Training Procedure

- **优化对象**：rewriter $\pi_\theta$（一个独立的小 LLM，与回答 LLM 解耦）；response LLM $M$、ATM 分类器、embedding 模型均**冻结**。
- **训练算法**：两阶段。
    - Stage 1：SFT，在多 agent 系统合成的高质量 rewriting 数据上对齐输出格式与分布。
    - Stage 2：GRPO（Group Relative Policy Optimization，DeepSeekMath 的 PPO lightweight 变体）。
- **训练数据**：合成的多轮对话 query-rewriting 数据（基于开源多轮对话数据 + 真实在线对话 + persona/blueprint 多 agent 模拟生成）。
- **是否冻结 LLM 参数**：response LLM 完全冻结；只更新 rewriter 的参数。
- **核心训练目标**：

GRPO 的 group-relative advantage（论文中未列原始公式，仅引用 DeepSeekMath）：

$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

reward 设计（论文 Algorithm 1，HSER）：

$$r = F1_h + \alpha \cdot EM_h + \beta \cdot \text{Hit}$$

其中

$$F1_h = \text{Rouge-L F1}(A_{\text{pred}}, A_{\text{label}})$$

$$EM_h = \mathbb{1}\lbrack A_{\text{pred}} \equiv A_{\text{label}} \rbrack$$

$$\text{Hit} = \mathbb{1}\lbrack A_{\text{pred}} \in C \rbrack \quad (C = \text{RAG-retrieved contents})$$

附录中又探索了一种 HSER + SSER 的融合 reward：

$$r = r_{\text{HSER}} + \lambda \cdot r_{\text{SSER}}$$

其中 $r_{\text{SSER}}$ 仅基于 rewriting 与人工标注 rewrite 的相似度。

---

## 3. Reward Signal

- **奖励类型**：sparse terminal reward（每条 rollout 结束后给一个标量奖励，没有 step-level dense reward，因为整个 rewriting 是单步生成）。
- **奖励来源**：
    - 部分来自 ground-truth 比较（F1、EM 与 $A_{\text{label}}$）
    - 部分来自检索环境反馈（Hit 检查 $A_{\text{pred}}$ 是否在检索内容 $C$ 中）
- **分配方式**：因为是单步 contextual bandit，奖励直接归到整条 rewrite 上，不存在跨步骤 credit assignment。
- **辅助 / 正则项**：
    - $\alpha \cdot EM_h$ 与 $\beta \cdot \text{Hit}$ 是论文设定的两个加权辅助项，权重通过消融实验在 test set 上调优。
    - 附录中的 SSER 项可视为针对 rewrite-level 的 imitation 正则。

> Hit 在鼓励 rewriter 把 query 写成"能让 RAG 召回到正确内容"的形态。

---

## 4. Inference Procedure

- **记忆初始化**：$m_l$ 是用户历史 query 的累积，初次使用为空；$m_s$ 在每个 turn 由 rewriter 重新生成，无跨 turn 的 latent state。
- **每步流程**（推理阶段，单 turn 内）：

1. 接收 $q$ 与最近 dialogue history $C$
    
2. **gₛ**: rewriter 生成 $m_s = \pi_\theta(q, C)$
    
3. **Pre-RAG**: 用 $m_s$ 检索文档，得到 $D = v(m_s, \text{KB})$
    
4. **gₗ (write side)**: ATM 分类器给 $q$ 打 (topic, sub-topic) tag，连同 embedding 写入 $m_l$ 对应分区
    
5. **Long-term recall**: 用 $q$ 的 embedding 在 $m_l$ 中召回 top-$n$ 历史 query: $H_n = v(q, m_l)$
    
6. **Attention-rerank**: 把 $D$ 切 chunk $\mathcal{C}$，对每个 chunk $c_i$ 计算
    
    $$\text{score}_i = \text{mean}_{h \in H_n} \text{sim}(\text{embed}(c_i), \text{embed}(h))$$
    
    按 score 取 top-$k$ 得到 `golden_contents`
    
7. **Reader**: 冻结的 $M$ 在 `golden_contents` 上生成 $\hat{a}_N$
    

- **额外推理策略**：top-$n$ 历史 query 召回（通常 $n$ 为小整数）、chunk top-$k$ rerank。**所有这些都是手工固定的 pipeline 规则**，不是 learned policy。
- **Learned vs. handcrafted**：仅 rewriter 是 learned；ATM 分类、embedding、rerank、reader 全是 frozen / rule-based。

---

## 5. RQ 分析

### RQ1 (What is memory?)

HiMeS 同时拥有显式 $m_s$（一条 rewritten query 字符串，可读可编辑）与显式 $m_l$（按 16 个主题分区存储的历史 query 向量库）。$m_s$ 本质上是把短期对话压缩成一段自然语言（接近 T2 的"上下文级压缩"思想，但不强制 fixed-length）；$m_l$ 是典型的非参数化外部记忆 + 层级结构（T1）。整个 rewriter 模块又可被理解为独立可插拔的记忆管理器（T5）。

### RQ2 (Which component is optimized? Which signal is used?)

被优化的是独立的 query rewriter 小模型。它与冻结的 response LLM 解耦，专门负责把对话历史压缩成 retrieval-friendly 的 query，属于 O1（独立可插拔记忆管理器）。训练信号是来自下游回答的混合 reward（Rouge-L F1 + EM + Hit），用 GRPO 优化。

### RQ3 (Target of Optimization)

最终追求的是回答准确度（G1，CA 与 QA 指标）。Hit 信号附带了一些"回答必须落在检索内容里"的 grounding 约束，可视为辅助项（G6 的弱形式）。

### RQ4 (Training Signal and RL Algorithm)

明确使用 **GRPO**（DeepSeekMath 提出的 PPO lightweight 变体），属于 A2。SFT warm-up 是预热，不是 RL 主体。

### RQ5 (How memory evolves, operates?)

推理时，$m_s$ 每 turn 由 rewriter 全新生成（无跨 turn latent 状态）；$m_l$ 通过 ATM 分类器 + embedding 在每次新 query 到来时增量写入（write-on-arrival），通过两级层级树定位分区，再通过 embedding 相似度召回 top-$n$ 历史 query 做 attention-rerank。整套读写策略全部是 handcrafted pipeline，没有 learned。

---

## Conclusion

HiMeS 是一个面向工业部署 AI 助手的双记忆 RAG 框架，灵感来自海马体—皮层的记忆机制。它的核心做法很简单：在一个被冻结的回答 LLM 前面，加两个可插拔模块：一个用 SFT+GRPO 训练出来的 query rewriter（短期记忆，把对话历史压成精炼 query），一个按 16 主题分区存储用户历史提问、并用 attention 方式 rerank RAG 文档的检索增强器（长期记忆）。reward 设计（HSER = F1 + EM + Hit）让 rewriter 直接对齐下游回答质量。在工业多轮对话数据集上，HiMeS 大幅领先 native RAG 基线，并且能 plug-and-play 到 DeepSeek-R1/V3、Qwen3-235B、Kimi-K2 等不同回答模型上。