# MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens

**Source:** [arXiv:2603.23516](https://arxiv.org/abs/2603.23516) (2026-03-06)

---

## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$D = \lbrace d_1, \ldots, d_N \rbrace$|$m_l$|文档库即长期记忆|
|选中的 Top-k 文档的压缩 KV ($\bar{K}, \bar{V}$)|$m_s$|加载到注意力窗口中的工作记忆|
|Router Q/K Projector 的 cosine 相似度打分|$v$|检索算法（稀疏路由）|
|Qwen3-4B-Instruct backbone|$M$|基础模型|
|User query hidden state $H_q$|$q$|用户输入|
|Memory Interleave 中的文档 ID 生成|$t$|推理轨迹（逐步检索推理）|
|Top-k 选择 + KV 拼接|$a$|动作（检索 + 上下文组装）|
|自回归生成的最终答案|$A_n$|最终输出|
|Router Projector 参数 $W_{QR}^h, W_{KR}^h$|新增参数（$\theta_R$）|Router 新增的可训练参数|
|$\mathcal{L}_{\text{aux}}$|—|辅助对比损失，监督路由决策|
|$\mathcal{L}_{\text{LLM}}$|—|标准语言模型生成损失|

---

## 1. Problem Setting

**记忆类型：** MSA 处理的是 **cross-chat 的长期记忆** ($m_l$)。整个文档语料库被离线编码为压缩的 KV 缓存，构成一个持久的 latent state memory bank，容量可达 1 亿 token。在线查询时，通过稀疏路由从 $m_l$ 中选取 Top-k 文档的压缩 KV 加载到注意力窗口中，形成临时的工作记忆 ($m_s$)。

**决策过程建模：** MSA **未被显式建模为 MDP 或 POMDP**。它本质上是一个端到端可微的稀疏注意力架构，通过监督学习（对比损失 + 语言模型损失）联合优化检索与生成，而非通过策略梯度或 Q-learning 来优化决策。Memory Interleave 机制中的多跳检索虽然具有序列决策的形式，但训练时被拆分为单步样本独立训练，并未使用 RL。

**状态/动作/观测空间：**

|组件|内容|对应符号|
|---|---|---|
|状态 $\mathcal{S}$|当前 query 的隐状态 $H_q$ + 已检索文档的 KV 缓存|$q + m_s$|
|动作 $\mathcal{A}$|从 $N$ 篇文档中选 Top-k 文档（或生成文档 ID）|$a$|
|观测 $\Omega$|选中文档的压缩 KV（$\bar{K}, \bar{V}$）|$o$|

**记忆数据结构：** 压缩后的 KV 缓存（latent state）。每篇文档的 $K, V, K^R$ 矩阵经 chunk-wise mean pooling（chunk size = 64 tokens）压缩为 $\bar{K}, \bar{V}, \bar{K}^R$，存储在 GPU（路由键）和 CPU（内容 KV）的分层存储系统中。

> MSA 将记忆表示为模型内部的 latent state 而非外部文本/向量，这使其与 RAG 有本质区别。但这也意味着记忆内容一旦编码便不可由用户直接编辑或理解，缺乏可解释性。此外，论文未将检索过程建模为 MDP，因此严格来说不属于 RL-based memory scheduling。

---

## 2. Training Procedure

**优化组件：**

- backbone LLM 参数 $\theta$（**未冻结**，通过持续预训练更新）
- 新增的 Router Projector 参数 $\theta_R$（$W_{QR}^h, W_{KR}^h$，随机初始化）

**优化算法：** 纯监督学习，无 RL。

**训练阶段：**

1. **持续预训练（Continual Pre-training, 158.95B tokens）：**
    
    - **Warm-up 阶段：** 以 $\mathcal{L} = 0.1 \mathcal{L}_{\text{LLM}} + \mathcal{L}_{\text{aux}}$，学习率 1e-4，优先对齐路由器。
    - **主预训练阶段：** 以 $\mathcal{L} = \mathcal{L}_{\text{LLM}} + 0.1 \mathcal{L}_{\text{aux}}$，学习率 6e-6，优先生成式检索任务。
    - 目标：训练模型执行 **Generative Retrieval**（自回归生成相关文档 ID）。
2. **后训练（Post-Training, SFT）：**
    
    - **Stage 1：** 在 8k 上下文上做 QA 的 SFT。
    - **Stage 2（课程学习）：** 清洗数据 + 将记忆上下文从 8k 扩展到 64k。

**核心目标函数：**

辅助路由损失（论文 Eq. 5，对比学习）：

$$\mathcal{L}_{\text{aux}} = -\frac{1}{\lvert P \rvert} \sum_{i=1}^{\lvert P \rvert} \log \frac{\exp(s_i^+ / \tau)}{\exp(s_i^+ / \tau) + \sum_{j=1}^{\lvert N \rvert} \exp(s_{i,j}^- / \tau)}$$

其中 $s_i^+$ 是第 $i$ 个正样本（query-正文档对）的相关性分数，$s_{i,j}^-$ 是对应的负样本分数，$\tau$ 为温度。

**框架符号标注：** 此损失直接优化检索算法 $v$（Router Projector）的参数 $\theta_R$，使其在 latent routing space 中将相关文档与不相关文档分离。

总训练损失：

$$\mathcal{L} = \mathcal{L}_{\text{LLM}} + \lambda \mathcal{L}_{\text{aux}}$$

其中 $\lambda$ 在 warm-up 阶段为主导（$\lambda = 1, \mathcal{L}_{\text{LLM}}$ 系数为 0.1），在主训练阶段为辅助（$\lambda = 0.1$）。

**训练数据来源：** 离线语料（158.95B tokens，涵盖科学文献、QA、新闻等 17 个领域）+ 人工构造的检索标注（正/负文档对）。

> 这篇文章采用的是经典的两阶段"先预训练后微调"范式。辅助对比损失 $\mathcal{L}_{\text{aux}}$ 是关键创新，它在模型内部 latent space 中直接监督路由决策，避免了 RAG 中检索器与生成器优化目标不一致的问题。但这也意味着训练需要显式的正/负文档标注，数据构造成本较高。

---

## 3. Reward Signal

**MSA 不使用 RL，因此没有传统意义上的奖励信号。**

替代机制如下：

|方面|描述|
|---|---|
|信号类型|监督信号（非 reward）|
|生成监督|标准交叉熵损失 $\mathcal{L}_{\text{LLM}}$（下一个 token 预测）|
|检索监督|对比损失 $\mathcal{L}_{\text{aux}}$（正/负文档对）|
|评估信号|LLM-as-judge（0-5 分），仅用于评估，不参与训练|
|信号分配|$\mathcal{L}_{\text{LLM}}$ 逐 token 分配；$\mathcal{L}_{\text{aux}}$ 逐层、逐 query-文档对分配|
|辅助正则|无显式正则项|

---

## 4. Inference Procedure

**记忆初始化：** 离线阶段（Stage 1: Global Memory Encoding）。对整个文档语料库执行一次前向传播，生成每篇文档的 $K, V, K^R$ 矩阵，经 chunk-wise mean pooling 压缩后缓存。路由键 $\bar{K}^R$ 存储在 GPU VRAM，内容 KV（$\bar{K}, \bar{V}$）卸载到 CPU DRAM。

**每步决策流程（单跳）：**

1. **观测：** 接收用户 query $q$，计算隐状态 $H_q$。
2. **检索（$v$）：** Router Q Projector 生成 $Q_{q}^R$，与缓存的 $\bar{K}^R$ 计算 cosine 相似度（Eq. 2），经 head mean pooling + token max pooling + document max pooling 得到文档级分数 $s_i$，选取 Top-k 文档。
3. **动作（$a$）：** 加载选中文档的 $\bar{K}, \bar{V}$ 到 GPU，与 query 的 $K_q, V_q$ 拼接形成稀疏上下文。
4. **生成（$A_n$）：** 在稀疏上下文上执行标准自回归生成。

**多跳推理（Memory Interleave）：**

对于需要多跳推理的复杂 query，MSA 迭代执行 Stage 2 和 Stage 3：

- 模型自回归生成文档 ID 序列 → 系统加载对应原文 → 将原文追加到 query 上下文 → 再次路由检索 → 重复直到模型判断证据充足 → 生成最终答案。
- 每轮检索的文档数量由模型**自适应决定**（非固定 k）。

**推理策略性质：**

- 路由决策完全由学习得到的 Router Projector 驱动（端到端学习）。
- Top-k 的 $k$ 值（默认 16）和 chunk size（64）为手工设定的超参数。
- Memory Interleave 的迭代终止由模型自主决定。

---

## 5. RQ 分析

### RQ1: What is memory?

MSA 将记忆定义为模型内部 latent state 级别的压缩 KV 缓存。文档语料库 $D$（$m_l$）被离线编码为压缩的 $\bar{K}, \bar{V}, \bar{K}^R$ 表示，存储在 GPU/CPU 分层存储系统中。这与 RAG 的文本/向量记忆和 RNN 的固定大小隐状态均不同：MSA 的记忆是可变规模的、保留在模型原生表示空间内的 latent state 集合。

### RQ2: How memory evolves, operates?

记忆在推理时不修改。$m_l$ 在离线阶段一次性编码后保持静态。运行时操作包括：

- **读取（retrieval）：** 通过 Router Projector 的 cosine 相似度选择 Top-k 文档的 KV 加载到 $m_s$。
- **无写入/更新：** 已编码的记忆不可动态修改。新增知识需要重新编码。
- **多跳演化：** Memory Interleave 通过扩展 query 上下文来隐式地"积累"信息，但记忆库本身不变。

### RQ3: Which component is optimized? Which signal is used?

优化了两个组件：(1) backbone LLM 参数 $\theta$，通过 $\mathcal{L}_{\text{LLM}}$（下一 token 预测）；(2) Router Projector 参数 $\theta_R$，通过 $\mathcal{L}_{\text{aux}}$（监督对比损失）。所有信号均为监督信号，**未使用 RL reward**。

### RQ4: Regarding online optimization

本文未涉及此问题。MSA 的记忆是离线编码的静态 KV 缓存，推理时不更新模型参数或记忆内容。

---

## Conclusion

这篇文章提出了一种端到端可训练的稀疏注意力框架，用于将 LLM 的记忆容量扩展到 1 亿 token 的人类终身记忆规模。其核心思路是：将文档语料库离线编码为压缩的 KV 缓存（latent state memory），并在推理时通过学习得到的 Router Projector 进行 Top-k 稀疏检索，仅将最相关文档的 KV 加载到注意力窗口中。配合 document-wise RoPE（实现短训练长推理的外推）、KV 缓存压缩（实现线性复杂度）和 Memory Interleave（支持多跳推理），MSA 在 QA 和 NIAH 基准上显著超越了同规模 RAG 系统和长上下文 LLM。
从 memory scheduling 的视角看，MSA 的检索策略是纯监督学习（对比损失 + LLM损失），无 RL。