# MemReader: From Passive to Active Extraction for Long-Term Agent Memory

**Source:** [arXiv:2604.07877](https://arxiv.org/abs/2604.07877) (2026-04-10 修订, 正在投递)

## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$x_t$|$q$|当前 user utterance（每轮新输入）|
|$M_t$|$m_l$|长期记忆库（结构化 JSON 条目集合，以 Milvus 向量索引）|
|$B_t$|$m_s$|临时缓冲区（保存不完整、待补全的候选记忆条目）|
|$s_t = (x_t, M_{t-1}, B_{t-1})$|$s$|决策状态|
|$\tau_t = \lbrace (z_t^{(k)}, a_t^{(k)}, o_t^{(k)}) \rbrace_{k=1}^{K_t}$|$T$|单轮内的 ReAct 轨迹|
|$z_t^{(k)}$|$t$|第 $k$ 步推理 trace（`<think>` 标签内容）|
|$a_t^{(k)} \in \mathcal{A}$|$a$|第 $k$ 步动作（工具调用）|
|$o_t^{(k)}$|$o$|第 $k$ 步观测（Milvus 检索结果或工具反馈）|
|$\mathcal{A} = \lbrace \text{add},\ \text{buffer},\ \text{search},\ \text{ignore} \rbrace$|动作空间|四个记忆操作工具|
|$\pi_\theta(\cdot \mid s_t)$|$\pi_\theta$|MemReader-4B 的策略|
|$\mathcal{T}(M_{t-1}, B_{t-1}, \tau_t)$|$g_l, g_s$|状态转移算子（同时更新 $m_l$ 与 $m_s$）|
|Milvus vector search|$v$|检索机制|
|GPT-4.1-mini（response model）|$M$|主回答模型，**冻结**|

---

## 概览

这篇文章用了多轮对话训练数据 + 由强 teacher 模型（Gemini-3-Flash-Preview）合成的 ReAct 轨迹（包含完整的 Think–Action–Observation 链）。在 search 动作处接入真实的 Milvus 向量数据库提供观测反馈，最终得到 7k SFT + 3k GRPO 样本。  
最后得到了一个独立可插拔的记忆管理器 MemReader-4B（基于 Qwen3-4B，SFT+GRPO 训练），能在每轮对话中显式判断信息的"价值/模糊性/完整性"，并选择四种动作之一（add / buffer / search / ignore）。同时附带一个轻量蒸馏版 MemReader-0.6B（Qwen3-0.6B 的 SFT 蒸馏版）用于纯结构化抽取场景。  
这篇文章直接优化目标是记忆写入决策的全过程质量：通过 4 项 reward 同时塑造（1）输出格式合规、（2）动作选择对齐 teacher 轨迹、（3）写入内容的 correctness/completeness/hallucination 由 LLM-as-judge 评分、（4）输出长度 efficiency。最终 downstream 在 LOCOMO / LongMemEval / HaluMem 三个 benchmark 上提升 QA 准确率，但训练信号本身不直接使用 QA accuracy。

---
## 1. Problem Setting

- **记忆类型**：处理 cross-session、long-term memory（$m_l$），并显式引入了一个临时短期缓冲 $B$（≈ $m_s$）用于存放不完整待补全的候选记忆。这与一般工作只建模 $m_l$ 不同——MemReader 同时维护两层显式记忆。
- **决策建模**：被显式建模为 **MDP**。
    - 状态：$s_t = (x_t, M_{t-1}, B_{t-1})$
    - 动作：$\mathcal{A} = \{ \text{add\_memory}, \ \text{buffer\_memory}, \ \text{search\_memory}, \ \text{ignore\_memory} \}$
    - 转移：$(M_t, B_t) = \mathcal{T}(M_{t-1}, B_{t-1}, \tau_t)$
    - 目标：$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=1}^T \gamma^{t-1} R(s_t, \tau_t) \right]$
- **数据结构**：
    - $m_l$ 是结构化 JSON 条目集合，每条 = `{key, memory_type, value, tags}`，以 embedding 形式存于 Milvus 向量库。
    - $m_s$（buffer）保存自然语言形式的待定信息及其触发原因。
- **观测空间**：search_memory 调用真实 Milvus 后端返回相关历史条目作为 $o$，注入下一步推理。

|组件|论文表述|框架符号|
|---|---|---|
|长期记忆|Long-term memory store $M_t$|$m_l$|
|短期缓冲|Buffer $B_t$|$m_s$|
|决策状态|$s_t = (x_t, M_{t-1}, B_{t-1})$|$s$|
|策略|$\pi_\theta(\cdot \mid s_t)$|$\pi$|
|检索|Milvus vector search|$v$|
|写入/更新|$\mathcal{T}(\cdot, \cdot, \tau_t)$|$g_l, g_s$|

---
## 2. Training Procedure

- **优化对象**：MemReader-4B 自身的全部参数 $\theta$（基于 Qwen3-4B 的全参数微调），即一个**独立可插拔的 memory manager**。**主回答 LLM（GPT-4.1-mini）保持冻结**——它只在 evaluation 阶段被调用来生成最终答案。
- **训练算法**：两阶段 **SFT + GRPO**。
    - **Stage 1（SFT warm-start）**：在 LlamaFactory 上对 7k 条 ReAct 轨迹做全参数 SFT（lr=1e-5，cosine，3 epochs，cutoff_len=4096，8×A800 + ZeRO-3），教模型 think–act 协议格式与四种动作的基本语义。
    - **Stage 2（DPO 尝试，失败）**：作者承认 DPO 训练时 chosen/rejected reward 同时下降、gradient 早早消失。归因：同一状态下两条候选输出往往都"部分合理"，pairwise preference 信号过弱。
    - **Stage 3（GRPO）**：用 verl 框架做 3k 样本的 GRPO 训练，多轮对话设置（train_batch_size=8, rollout.n=8, lr=1e-6, max_assistant_turns=16, max_response_length=768）。
- **训练数据来源**：以 Gemini-3-Flash-Preview 作为 teacher 合成 ReAct 轨迹；search 动作处接入真实 Milvus 后端获得真实 observation；ShareGPT 格式，max_chain_len=10 支持 buffer→add 的跨轮链式样本。
- **冻结情况**：主回答 LLM（GPT-4.1-mini）完全冻结；MemReader-4B 全参数微调；MemReader-0.6B 仅做蒸馏 SFT。

**SFT 目标函数**（原文 Eq. 6）：

$$ \mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(s,y) \sim \mathcal{D}_{\text{SFT}}} \left[ \sum_{\ell=1}^{\lvert y \rvert} \log \pi_\theta(y_\ell \mid s, y_{< \ell}) \right] $$

**GRPO 目标函数**（原文 Eq. 18）：

$$ \mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{s, \lbrace y_i \rbrace} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\lvert y_i \rvert} \sum_{\ell=1}^{\lvert y_i \rvert} \min\Big( \rho_{i,\ell}(\theta) \hat{A}_i,\ \text{clip}(\rho_{i,\ell}(\theta), 1-\epsilon_c, 1+\epsilon_c) \hat{A}_i \Big) \right] - \beta, D_{KL}(\pi_\theta \Vert \pi_{\text{ref}}) $$

其中 group-relative advantage $\hat{A}_i = (R_i - \bar{R}) / \sigma_R$，importance ratio $\rho_{i,\ell}(\theta) = \pi_\theta(y_{i,\ell} \mid s, y_{i,< \ell}) / \pi_{\theta_{\text{old}}}(y_{i,\ell} \mid s, y_{i,< \ell})$。

---
## 3. Reward Signal

- **奖励类型**：trajectory-level **dense** reward，由 4 个分量加权求和，作用于整条 ReAct 轨迹（而非仅最终回答）。
- **奖励分解**（原文 Eq. 7）：

$$ R_t = \lambda_{\text{fmt}} r_t^{\text{fmt}} + \lambda_{\text{align}} r_t^{\text{align}} + \lambda_{\text{judge}} r_t^{\text{judge}} + \lambda_{\text{eff}} r_t^{\text{eff}} $$

- **$r^{\text{fmt}}$（格式 reward）**：规则判定，检查 `<think>` `<tool_call>` 标签闭合 + JSON 可解析。
    
- **$r^{\text{align}}$（动作对齐 reward）**：拆为三部分（turn-level prefix 比对 + 终局动作正确性 + 动作分布一致性），其中 `wfinal` 占 ~50% 权重（终局错误代价最大）。 $$r_t^{\text{align}} = w_{\text{turn}} \bar{r}_{\text{turn}} + w_{\text{final}} r_{\text{final}} + w_{\text{dist}} r_{\text{dist}}$$
    
- **$r^{\text{judge}}$（内容质量 reward）**：LLM-as-judge 给出 correctness/completeness/hallucination 三项 $[0,1]$ 分数加权，**仅在 add_memory 触发时启用**。 $$r_t^{\text{judge}} = \alpha_{\text{cor}} s_t^{\text{cor}} + \alpha_{\text{comp}} s_t^{\text{comp}} + \alpha_{\text{hall}} s_t^{\text{hall}}$$
    
- **$r^{\text{eff}}$（效率 reward）**：基于输出字符长度的线性惩罚，超出 $L_{\max}$ 时给固定负值 $-\delta$。
    
- **Credit assignment**：是论文重点讨论的问题。除了把 reward 拆到 trajectory 各部分，还做了 **argument-level credit sinking**——reward 不仅看动作是否调用，还看其参数（写入内容）是否正确、完整、无幻觉。这种内容级 credit 是密集过程监督。
    
- **辅助/正则**：$r^{\text{fmt}}$ 与 $r^{\text{eff}}$ 都属于 shaping；KL 项 $\beta D_{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ 防止偏离参考策略。
    

> $r^{\text{judge}}$ 评估的是记忆内容质量。质量好不好由 LLM-judge 单独评估。

---
## 4. Inference Procedure

- **记忆初始化**：$M_0 = \emptyset$，$B_0 = \emptyset$；随交互逐步累积。
- **每轮决策流程**：
    1. 观测当前 utterance $x_t$、$M_{t-1}$、$B_{t-1}$。
    2. 进入 ReAct 内循环，多次 Think→Action→Observation：
        - **Think ($z$)**：内部回答三问——Q1 价值是否值得记忆？Q2 是否含模糊指代？Q3 是否完整？
        - **Action ($a$)**：调用 add / buffer / search / ignore 之一。
        - **Observation ($o$)**：若 search 则注入真实 Milvus 检索结果；若其他则进入终态。
    3. 终态后调用状态转移 $\mathcal{T}$ 更新 $(M_t, B_t)$。
    4. 下游 QA 时由独立的 GPT-4.1-mini 基于 $M_t$ 检索结果生成回答。
- **额外策略**：
    - 硬上限 max_assistant_turns=16, max_response_length=768，避免 trace 失控。
    - search 使用真实 embedding + Milvus，不是模拟。
    - 推理温度 0.7。
- **学习 vs. 规则**：动作选择完全由学习得到的 $\pi_\theta$ 驱动；ReAct 协议格式、Milvus 检索后端、buffer 的链式触发等是手工脚手架。

---
## 5. RQ 分析

### RQ1（What is memory?）

本文的 $m_l$ 是非参数化外部结构化条目库（T1）：JSON 形式的离散条目 + Milvus 向量索引，可显式查看与 CRUD。同时引入临时 buffer $B$ 作为 $m_s$，专门保存不完整待补全的候选记忆。MemReader-4B 整体作为一个独立可插拔的辅助模块（T5），拥有独立参数，独立于主回答 LLM 之外。

### RQ2（Which component is optimized?）

被优化的是 MemReader-4B 这一独立可插拔记忆管理器（O1）：主回答 LLM（GPT-4.1-mini）冻结，MemReader-4B 学习 add/buffer/search/ignore 的决策。从"学习何时调用什么记忆操作"角度也涉及 O4（工具触发概率）。

### RQ3（Target of Optimization）

多目标组合：

- G2 效率：efficiency reward 限制输出长度。
- G3 行动奖励：action-alignment reward 对每步及终局动作打分。
- G4 写入决策质量：LLM-judge 评估 add 内容的 correctness/completeness/hallucination；动作分布一致性也属此。
- G6 辅助 reward：format reward。
- G1 不直接用作训练目标，QA accuracy 仅在 evaluation 报告。

### RQ4（Training Signal and RL Algorithm）

A2 GRPO：用 group-relative advantage 替代 critic，PPO-style clipped surrogate + KL 正则。论文显式尝试过 DPO（A5）但失败，最终采用 GRPO。

### RQ5（How memory evolves, operates?）

逐轮演化：每轮 ReAct 内循环结束后通过 $\mathcal{T}$ 更新 $(M, B)$。写入仅支持新增条目（无显式合并 / 编辑机制。作者在 Conclusion 承认 memory editing / conflict detection 是未来工作）。检索是非梯度的 Milvus 向量召回。Buffer 在后续轮被消费（链式触发）后转写到 $m_l$ 或被丢弃。整体上记忆通过"显式动作 + 真实检索后端"演化，不通过 latent 状态或参数变化。

---
## Conclusion

这篇论文的核心主张是：长期记忆的瓶颈不在"如何写更好的 JSON"，而在"什么时候写、是否要先检索消歧、是否要暂缓"。记忆抽取应该是主动的状态管理，而不是被动的转写。为此作者把记忆抽取建模为一个 MDP，动作空间是四种记忆操作（add / buffer / search / ignore），用 ReAct 范式让模型显式 think-act-observe，再用 SFT+GRPO 训练一个独立于主回答 LLM 之外的记忆管理器 MemReader-4B（4B 参数）。配合一个轻量蒸馏版 MemReader-0.6B 用于纯结构化抽取场景。在 LOCOMO / LongMemEval / HaluMem 三个 benchmark 上均取得领先，尤其在知识更新和时间推理任务上提升显著。其训练信号完全在"记忆侧"闭环（不直接打通到下游 QA）。