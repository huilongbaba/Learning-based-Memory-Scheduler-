# Training Plug-and-Play Knowledge Modules with Deep Context Distillation

**Source:** [arXiv:2503.08727](https://arxiv.org/abs/2503.08727) (Published at COLM 2025)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$D$|—|文档（document），框架中无直接对应，可视为 $m_l$ 的一个实例来源|
|$\theta_{KM}$|$\theta_{KM}$（新增）|Knowledge Module 参数（LoRA adapter），编码单篇文档知识的连续参数|
|$\theta_{KE}$|$\theta_{KE}$（新增）|Knowledge Extractor 参数（任务适配 LoRA）|
|$p(\cdot \mid D)$|—|教师模型（以文档 $D$ 作为上下文的 base LM）|
|$p(\cdot ; \theta_{KM})$|—|学生模型（不含文档，仅依赖 KM 参数）|
|$h^l$|—|第 $l$ 层隐藏状态|
|$\tilde{D}$|—|蒸馏用数据（合成或文档片段）|
|$S_k$|—|合成数据（摘要、QA 对等）|
|$C_k, C_{k+1}$|$C$（context 的实例）|文档中连续的 token 块|
|$q_i, a_i$|$q, A_n$|问题与答案|
|$P_1^i, \dots, P_M^i$|—|RAG 检索到的段落，属于 $R$（RAG 系统）的输出|
|base LM|$M$|基座语言模型|

> 本文不涉及 RL/policy 优化。$m_s$（短期记忆）可类比为 ICL 时的上下文窗口内容；$m_l$（长期记忆）可类比为训练好的 KM 参数 $\theta_{KM}$，因为它们在推理时被"插入"模型以提供持久化的文档知识。

---

## 1. Problem Setting

- **Memory 类型：** 本文处理的是 **cross-chat memory**，更准确地说是**文档级知识的持久化编码**。每篇文档的知识被压缩进一个 LoRA adapter（$\theta_{KM}$），在推理时按需加载。这对应 $m_l$（长期记忆），因为 KM 一旦训练完成即可跨会话复用，不依赖当前对话上下文。
- **决策过程建模：** 本文 **未将记忆管理建模为 MDP 或任何序贯决策过程**。KM 的训练是一个标准的监督优化问题（蒸馏 + 合成数据），不涉及状态转移或策略学习。
- **状态空间 / 动作空间 / 观测空间：** 不适用。本文无 RL 建模，无 $\mathcal{S}$、$\mathcal{A}$、$\Omega$ 的定义。
- **记忆的数据结构：** KM 以 **LoRA 低秩矩阵对** $(A, B)$ 的形式存在于模型的每一个线性层中，是一种**连续参数化的隐式记忆**，而非向量库或 KV 对等显式结构。

|核心组件|框架符号|本文实例|
|---|---|---|
|长期记忆|$m_l$|$\theta_{KM}$：每篇文档对应一个 LoRA adapter|
|短期记忆|$m_s$|ICL 上下文窗口（RAG 检索段落 $P_1, \dots, P_M$）|
|检索算法|$v$|SFR-embedding-2 余弦相似度 Top-K 检索|
|RAG 系统|$R$|文档分块 + embedding 检索 + 段落拼接|
|模型|$M$|Phi-3 (3B) / Llama-3.1 (8B)，参数冻结|
|问题|$q$|QA 数据集中的问题 $q_i$|
|最终答案|$A_n$|模型生成的答案 $a_i$|

> KM 将 $m_l$ 实现为模型参数的一部分而非外部数据库，这使得"记忆"与"模型能力"的边界变得模糊。与传统 RAG（$m_l$ 是外部向量库）相比，KM 的记忆是"内化"的，不可直接检查或编辑单条记忆条目。

---

## 2. Training Procedure

- **优化的组件：** 仅优化 $\theta_{KM}$（文档级 LoRA 参数）和可选的 $\theta_{KE}$（任务级 LoRA 参数）。**基座模型 $M$ 的参数完全冻结**。
- **优化算法：** 标准梯度下降（Adam 优化器 + cosine learning rate scheduler），**非 RL 算法**。
- **训练数据来源：**
    - KM 训练：文档本身（DDCD）或从文档生成的合成数据（SDCD），包括摘要、QA 对、Entigraph 实体关系图
    - KE 训练：有监督的 QA 标注数据 $\lbrace (q_i, a_i, D_i) \rbrace$
    - 合成数据可由 base LM 自身或更强模型（GPT-4o）生成
- **冻结策略：** LLM 参数完全冻结，仅优化 LoRA 的 $(A, B)$ 矩阵（rank=16）

### 核心训练目标函数

**Deep Context Distillation (DCD) 损失（论文 Eq. 2）：**

$$\mathcal{L}_{DCD} = \min_{\theta_{KM}} \mathrm{KL}\big( p(\tilde{D} \mid D) \Vert p(\tilde{D}; \theta_{KM}) \big) + \sum_l \frac{1}{Z^l} \lVert h^l_{\tilde{D} \mid D} - h^l_{\tilde{D}; \theta_{KM}} \rVert_1$$

用框架符号重写：教师为 $M(C, D)$（文档在上下文中的模型），学生为 $M(\theta_{KM})$（仅依赖 KM 的模型），优化目标是让 $\theta_{KM}$（即 $m_l$）编码足够的文档知识以逼近教师的隐藏状态和输出分布。

**Synthetic DCD 损失（论文 Eq. 4）：**

$$\mathcal{L}_{SDCD} = \min_{\theta_{KM}} \mathrm{KL}\big( p(S_k \mid C_k) \Vert p(S_k; \theta_{KM}) \big) + \sum_l \frac{1}{Z^l} \lVert h^l_{S_k \mid C_k} - h^l_{S_k; \theta_{KM}} \rVert_1$$

**Knowledge Extractor 损失（论文 Eq. 5）：**

$$\mathcal{L}_{KM+KE} = \min_{\theta_{KE}, w} - \sum_i \log p(a_i \mid q_i; [\theta_{KM}^{D_i}, \theta_{KE}]_w)$$

其中 $[\theta_{KM}^{D_i}, \theta_{KE}]_w = w_M A_{D_i} B_{D_i}^T + w_E A_E B_E^T$，$w_M, w_E$ 为可学习的逐层标量权重。

> DCD 的核心洞察在于：在低数据场景下，蒸馏（匹配教师分布）比直接 NTP（匹配 one-hot 标签）提供更丰富的梯度信号。Table 6 的消融实验通过降低温度 $\tau \to 0$ 将 DCD 退化为 NTP，证实了这一点。这与 RL 中 reward shaping 的思想有类比——更密集的信号有助于更高效的学习。

---

## 3. Reward Signal

- **奖励类型：** 本文**不使用 RL 奖励信号**。训练信号来自蒸馏损失（KL 散度 + L1 隐藏状态匹配），属于监督学习范畴。
- **奖励来源：** 不适用。评估指标为 Rouge-L（NarrativeQA）和 Accuracy（QuALITY），但这些不作为训练信号。
- **奖励分配：** 不适用。蒸馏损失在每个 token 位置和每一层都提供梯度信号，可视为一种天然的"dense signal"。
- **辅助奖励 / 正则项：** 无显式正则项。隐藏状态 L1 损失可被视为一种辅助信号，为每层 LoRA 提供更直接的 credit assignment。

---

## 4. Inference Procedure

- **记忆初始化：** 推理前，根据目标文档加载对应的预训练 $\theta_{KM}$（LoRA adapter），按需插入基座模型。若有 KE，同时加载 $\theta_{KE}$ 并通过加权组合 $[\theta_{KM}, \theta_{KE}]_w$ 合并。
- **每步决策流程：**
    1. 接收问题 $q$
    2. （可选 open-book）通过 $v$（SFR-embedding-2）从文档 $D$ 中检索 Top-K 段落 $P_1, \dots, P_K$
    3. 将 $q$（及可选的检索段落）输入装载了 $\theta_{KM}$ 的模型 $M$
    4. 模型生成答案 $A_n$（NarrativeQA 使用 greedy decoding）
- **推理时额外策略：** Top-K 检索（$k = 1, 5, 8, 16$），贪婪解码。无多轮 replan 或温度调节。
- **策略驱动方式：** 推理完全由 **训练得到的参数** $\theta_{KM}$（和 $\theta_{KE}$）驱动，结合标准的 RAG 检索流水线（手工设计的 Top-K 规则）。无学习到的 $\pi$（策略）。

> 当前 KM 推理假设已知文档归属（即知道问题来自哪篇文档）。作者在结论中提到未来可结合 zero-shot routing 方法解决文档选择问题，这将引入一个显式的路由策略 $\pi_{route}$，更接近 RL 框架。

---

## 5. RQ 分析

### RQ1: What is memory?

本文将 memory 定义为文档级的连续参数化知识，具体实现为 LoRA adapter $\theta_{KM}$。每篇文档对应一个独立的 KM，编码该文档的全局信息。这是一种隐式的 $m_l$（长期记忆），知识被"蒸馏"进模型权重而非存储在外部数据库中。

### RQ2: How memory evolves, operates?

KM 在训练阶段通过 DCD 蒸馏一次性构建，推理时不再演化或更新。运行时仅执行"读取"操作，即将预训练的 LoRA 参数插入模型。无显式的写入、删除或更新机制（但可通过删除整个 KM 实现文档级遗忘）。记忆的"读取"是隐式的：模型前向传播时自动通过 LoRA 调制激活值。

### RQ3: Which component is optimized? Which signal is used?

优化的组件为 $\theta_{KM}$（文档级 LoRA）和 $\theta_{KE}$（任务级 LoRA），基座模型冻结。训练信号为蒸馏损失（KL 散度 + L1 隐藏状态匹配），不使用 RL 奖励。KE 使用标准交叉熵监督信号。

### RQ4: Regarding online optimization

**未涉及。** 本文的 KM 训练是离线的一次性过程，不支持在线持续学习。KM 训练后参数固定，不随新交互更新。若文档内容变化，需重新训练对应的 KM。这与在线适应的 RL 框架有本质区别。

---

## Conclusion

本文提出了 Knowledge Modules (KMs)，一种将文档知识编码为轻量级 LoRA adapter 的方法，很像我们想做plugable的思路，可在推理时按需插拔。核心技术贡献是 Deep Context Distillation (DCD)：通过让学生模型（仅依赖 KM 参数）逼近教师模型（以文档为上下文）的隐藏状态和输出分布，在低数据场景下远优于标准的 next-token prediction 训练。实验表明 DCD 与合成数据生成、RAG 系统均有协同增益。
从 memory 框架的视角看，KM 提供了一种将 $m_l$ 实现为模型参数子空间的方案，但其离线训练、不可在线更新的特性使其不直接适用于需要动态记忆管理的 RL 场景。