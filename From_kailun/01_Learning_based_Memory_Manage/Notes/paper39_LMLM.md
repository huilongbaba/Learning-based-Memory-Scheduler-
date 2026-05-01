# Pre-training Limited Memory Language Models with Internal and External Knowledge (LMLM)

**Source:** [https://openreview.net/pdf?id=cvztBvlglK](https://openreview.net/pdf?id=cvztBvlglK) (2026-04-11 修订，Accepted at ICLR 2026 poster，8864)

---

## 符号映射表

|论文原始符号 / 概念|框架符号|说明|
|---|---|---|
|`pθ` (autoregressive LM)|$M, \pi_\theta$|主语言模型，参数为 $\theta$|
|External Knowledge Base (54.6M triplets `(entity, relation) → value`)|$m_l$|长期记忆：非参数化外部数据库|
|Context window (1024 tokens)|$m_s$ / $C_N$|短期记忆/上下文，无显式压缩机制|
|Fuzzy match w/ ALL-MINILM-L6-V2 (cosine sim, threshold 0.6)|$v$|检索算法|
|Lookup call: `<\|db_start\|> entity <\|sep\|> relation <\|db_retrieve\|>`|$a$ (查询动作)|触发检索的"动作"|
|Returned value (`<\|db_retrieve\|> value <\|db_end\|>`)|$o$ (观测)|数据库返回的事实值|
|Annotator model (LLAMA-3.1-8B-INSTRUCT, LoRA-tuned)|(offline 模块)|独立训练的标注器，仅用于训练数据准备阶段，**不参与推理**|
|Question / prompt|$q$|用户输入|
|Generated text|$\hat{a}_N$|最终输出|
|Loss mask $m_t$|(训练信号)|选择性 mask 掉 retrieved values 的 token-level mask|

---

## 概览

这篇文章用了预训练语料 + GPT-4o 标注的 (entity, relation, value) 三元组。具体地，作者用 GPT-4o 在 1k 段文本上做高质量种子标注，然后蒸馏出一个轻量级的 Annotator（LLaMA-3.1-8B + LoRA），用它给整个预训练语料注入 lookup 调用，并把抽出的事实三元组存进一个外部数据库。  
这篇文章最后得到了一个从头预训练的“有限”记忆语言模型 LMLM，配上一个外挂数据库。这个模型在预训练时被显式地训练用查表代替记忆，即遇到事实问题就生成 lookup 调用、把检索到的值用回去，所以模型本身参数里的事实知识被大幅减少。  
优化的目标是把语言能力和事实知识解耦。优化的指标包括：(1) 验证集 perplexity 更低；(2) 事实精确度指标提升（G1）；(3) 同等参数下匹敌大几十倍的 off-the-shelf 模型（G2）；(4) 通过删数据库条目实现瞬时遗忘。

---

## 1. Problem Setting

- **记忆类型**：本文处理的是 **cross-chat、长期、跨 episode 的事实知识**，对应 $m_l$；不涉及 in-chat 或 working memory 类型的 $m_s$。
- **决策过程建模**：本文**未将记忆管理建模为 RL 决策过程**（无 MDP/POMDP/bandit 框架）。它把"何时查询、查什么"作为一个**自回归语言建模子任务**，通过对训练数据中的 lookup 调用做 next-token prediction 来隐式学到。
- **状态空间**：$\mathcal{S}$ = 当前已生成 token 序列 $x_{< t}$；
- **动作空间**：$\mathcal{A} = V \cup V_{\text{db}}$，其中 $V$ 是原始词表，$V_{\text{db}}$ 是四个新增的特殊 token：`<|db_start|>`、`<|sep|>`、`<|db_retrieve|>`、`<|db_end|>`；
- **观测空间**：$\Omega$ = 数据库返回值（一段自然语言 value，或 `unknown`）。
- **记忆数据结构**：**结构化三元组数据库**，每条形如 `(entity, relation) → value`，全部以自然语言文本存储；不带向量索引（向量索引仅用于 fuzzy 检索时即时计算）。

|框架组件|论文实现|
|---|---|
|$M$|LLaMA2-176M / LLaMA2-382M / GPT2-124M / GPT2-355M（从头预训练）|
|$m_l$|54.6M 三元组的外部数据库（9.5M 实体、8.5M 关系、16.2M 值）|
|$m_s$|标准 1024-token context window，无额外压缩|
|$v$|Fuzzy match：sentence embedding (MiniLM) + cosine sim，阈值 0.6|
|$a$|由 LM 自回归生成的 `<\|db_start\|> entity <\|sep\|> relation <\|db_retrieve\|>`|
|$o$|数据库返回的 value 字符串|

> 将记忆操作放在 pre-training 阶段。

---

## 2. Training Procedure

- **优化对象**：主 LLM 的全部参数 $\theta$（**从头预训练**），即对应框架中的 $M$ 本身。Annotator 模型在前置阶段单独 LoRA 微调，但它仅参与训练数据准备，不在主 LMLM 的优化回路里。
- **优化算法**：标准 **next-token prediction with masked loss**（**纯监督学习/SFT-style，不是 RL**）。
- **训练数据来源**：**离线静态语料**——OLMo2 项目的 Wikipedia 子集（~3B token），由 Annotator 注入 lookup 调用。
- **是否冻结 LLM 参数**：否，主 LLM 完全从头训练；$\theta$ 全部参数都参与优化。

**核心目标函数**（论文 Eq. 1）：

$$ \mathcal{L}(\theta) = -\sum_{t=1}^{T} m_t \log p_\theta(x_t \mid x_{< t}) $$

其中 mask $m_t$ 定义为：

$$ m_t = \begin{cases} 0, & x_t \in \mathcal{T}_v \cup \lbrace \tau_{\text{end}} \rbrace \ 1, & \text{otherwise} \end{cases} $$

这里 $\mathcal{T}_v$ 是 retrieved value 的 token 集合，$\tau_{\text{end}}$ 表示 `<|db_end|>` 特殊 token。

用框架符号表达：

$$ \mathcal{L}_{\text{LMLM}}(\theta) = -\mathbb{E}_{x \sim D_{\text{annot}}} \sum_{t} m_t \log \pi_\theta(x_t \mid x_{< t}) $$

其中 $D_{\text{annot}}$ 是注入了 lookup 调用的预训练语料，$m_t$ 是显式选择性监督 mask。

> 不教模型记答案，只教模型问问题。

---

## 3. Reward Signal

- **奖励类型**：本文**没有显式的 reward signal**——使用的是负对数似然 (NLL) 监督信号。如果非要类比 RL 视角，可以把"对正确 lookup token 的 NLL"看作 dense step-level supervision。
- **奖励来源**：来自 Annotator 注入的 ground-truth lookup 序列（间接来自 GPT-4o 蒸馏 + Corrector 过滤）。
- **奖励分配**：token-level，由 mask $m_t$ 控制——$m_t=1$ 的 token 参与 NLL，$m_t=0$ 的 token（retrieved values 和 `<\|db_end\|>`）被完全排除。
- **辅助奖励**：无 explicit auxiliary reward，但有一个隐式的"selective offloading"机制——通过 LMLM/STANDARD 训练后 loss 差异挑选最值得 offload 的事实（见 § 5 中的 ranking criterion）。

---

## 4. Inference Procedure

- **记忆初始化**：数据库在预训练后已经构建完毕（54.6M 三元组），推理时**不变**——它是"读多写少"的静态长期记忆。
    
- **每步决策流程**：
    
    1. **生成**：模型自回归生成 token；
    2. **触发**：当生成到 `<\|db_start\|>` 时，自动进入 lookup 模式；
    3. **构造查询**：模型继续生成 `entity <\|sep\|> relation`，直到生成 `<\|db_retrieve\|>`；
    4. **检索 (v)**：用生成的 `(entity, relation)` 做 sentence embedding，与数据库做 cosine 相似度匹配，相似度 > 0.6 则返回对应 value，否则返回 `unknown`；
    5. **注入观测**：把 retrieved value 直接 append 到生成序列中，紧跟一个 `<\|db_end\|>`；
    6. **继续生成**：模型基于新的上下文继续自回归生成。
- **额外推理策略**：
    
    - **Logit bias**：在 FactScore 评估中给四个特殊 token 加 logit bias（5.0, 2.0, 2.0, 2.0）以鼓励 lookup；
    - **Prefix-tree decoding**（消融）：可选的 constrained decoding，确保查询合法，但实测不如 fuzzy match；
    - **Fallback**：检索失败时输出 `unknown`，模型继续 plain decoding（未经训练的 fallback，是一个 known limitation）。
- **推理策略来源**：**完全由学习得到**——预训练后模型自然知道何时插入 `<\|db_start\|>` 触发 lookup；无手工规则强制触发（评估时仅有可选的 logit bias 微调，不影响整体逻辑）。
    

---

## 5. RQ 分析

### RQ1 (What is memory?)

LMLM 的记忆是一个显式、非参数化、外部、可寻址的事实三元组数据库（$m_l$）。结构是 flat 的 `(entity, relation) → value`，靠 sentence embedding fuzzy match 检索，条目可被自由增删（这是其 unlearning 能力的基础）。完全对应 T1。

### RQ2 (Which component is optimized?)

优化的是主 LLM 本身的全参数 $\theta$（O2a：单 LLM 优化）。Annotator 是独立的 offline 工具，不参与主 LLM 的训练回路。retrieval 算法是 fixed 的 fuzzy match，未被优化。训练信号来自精心设计的 SFT 数据 + token-level loss mask，不依赖 reward model 或 RL。

### RQ3 (Target of Optimization)

主要追求两个目标：(1) G1 回答准确度，FactScore / T-REx EM / PopQA Acc 都是直接的事实回答正确性指标；(2) G2 效率，参数效率（382M 模型 ≈ LLaMA2-7B 的 FactScore）和 perplexity 降低。次要目标包括 unlearning 质量（通过删数据库条目实现）。

### RQ4 (Training Signal and RL Algorithm)

N/A: Non-RL Methods。LMLM 完全用 next-token prediction 做监督学习，损失函数就是带 mask 的标准 NLL，不存在 RL。

### RQ5 (How memory evolves, operates?)

数据库在预训练阶段一次性构建（5460 万条三元组），推理时是只读的静态记忆。不在 episode 内部演化、不写入新内容。读操作通过模型自动生成 lookup 调用 + fuzzy match 检索完成。"演化"只发生在离线维护期：要更新或遗忘事实时，直接对数据库做 CRUD 操作即可（这是 LMLM 在 unlearning 上 outperform NPO 等方法的根本原因）。runtime 记忆是静态的，所有动态性都被推到了离线维护阶段。

---

## Conclusion

LMLM 提出了一个想法：与其在训练完之后再给 LLM 接 RAG，不如在预训练阶段就让模型学会查表代替记忆。具体做法是用一个小型 Annotator 把预训练语料中的 (entity, relation, value) 三元组抽出来做成一个外部数据库，然后在原文里插入 lookup 调用 token；预训练时通过 mask 掉 retrieved value 的 loss，逼着模型只学何时查、查什么，而不学答案是什么。结果是一个非常小的 LMLM（382M 参数）在事实精确度上能逼近 LLaMA2-7B，并且因为事实存在外部数据库里，删掉条目就等于让模型遗忘。TOFU benchmark 上做到了 ideal unlearning（p > 0.05）而不损失 utility。