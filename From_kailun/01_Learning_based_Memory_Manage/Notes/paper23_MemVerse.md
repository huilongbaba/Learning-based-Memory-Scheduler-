# MemVerse: Multimodal Memory for Lifelong Learning Agents
**Source:** [arXiv:2512.03627v1](https://arxiv.org/abs/2512.03627) (2025-12-3 修订)

## 符号映射表

| 论文原始符号 | 框架符号 | 说明 |
| --- | --- | --- |
| $M_{STM}$ (short-term memory, 最近 K 条 query 的滑动窗口) | $m_s$ | 短期记忆（recent conversations） |
| $\mathcal{M} = (\{\mathcal{G}_k\}, \mathcal{C})$（知识图 + 原始 chunk） | $m_l$ | 长期记忆（hierarchical MMKG + raw chunks） |
| $\mathcal{G}_k = (\mathcal{V}_k, \mathcal{R}_k)$, $k \in \lbrace \text{core, episodic, semantic} \rbrace$ | $m_l$ 的内部结构 | 三类知识图：core / episodic / semantic |
| $\mathcal{M}^t_{\text{parametric}}$ (参数化 memory model, 7B LLM) | 新增符号 $m_p$ | 参数化记忆（periodic distillation 得到的 fast pathway） |
| RAG 检索 + KG 遍历 | $v$ | 检索算法（embedding + graph multi-hop） |
| $\Phi_{LLM}(\mathcal{C})$（从 chunks 抽实体 / 关系） | $g_l$ | 长期记忆更新函数（LLM-based KG construction） |
| $\mathcal{L}_{\text{update}}$ (token-level CE) | SFT loss | 参数化记忆的蒸馏目标 |
| Orchestrator（rule-based, 无可训练参数） | 手工规则的 $g_s / g_l / v$ 调度器 | 不属于被优化组件 |
| $q, R, \hat{R}$（question, retrieved context, answer） | $q, C_N, A_N$ | 查询 / 检索上下文 / 最终回答 |

---

## 概览

这篇文章用了长期多模态对话历史（图像 / 视频 / 音频 / 文本经过 MLLM 转成文字 chunks）和用户交互轨迹，把它们通过 LLM 抽取实体和关系，组织成三类分层知识图（core / episodic / semantic）作为长期记忆；同时保留最近若干轮对话作为短期记忆。  
这篇文章最后得到了一个插件式的记忆框架 MemVerse，它包含三部分：(1) 短期记忆（滑动窗口）、(2) 基于分层多模态知识图的长期记忆、(3) 一个通过周期性 SFT 从长期记忆蒸馏出来的轻量参数化记忆模型（7B LLM），主 LLM 通过规则引擎统一访问这三种记忆。
这篇文章它只优化了参数化记忆模型的 SFT 目标（让小模型学会"给定 query 生成 RAG 检索到的内容"），最终追求的是下游多模态 QA / 检索任务的准确度以及检索速度（ScienceQA、LoCoMo、MSR-VTT 基准，推理延迟从 RAG 的 20s 降到 2.28s）。

---

## 1. Problem Setting

- **Memory 类型**：同时覆盖 in-chat 与 cross-chat memory。
  - $m_s$：最近 K 条 query 的滑动窗口（in-chat working memory）
  - $m_l$：跨 session 持久的多模态知识图 + raw chunks（cross-chat long-term memory）
  - $m_p$（新增）：从 $m_l$ 蒸馏出来的参数化记忆模型
- **决策过程建模**：**不是 MDP / POMDP / bandit**。MemVerse 把记忆管理建模为**确定性的 rule-based pipeline**——orchestrator 用固定规则决定何时写入、何时检索，没有显式的 agent 决策 / 策略学习。唯一被优化的部分（参数化记忆的 SFT）是纯监督学习问题，目标是模仿 RAG 的检索结果。
- **状态 / 动作 / 观测**：论文没有显式定义。若强行套用，状态 $s = (m_s, m_l, m_p, C)$，动作由 orchestrator 按规则触发（无策略学习）。
- **记忆数据结构**：
  - $m_s$：自然语言 query 列表
  - $m_l$：**多模态知识图（MMKG）** + 原始文本 chunks，图节点 / 边反向索引到 chunks 和多模态数据
  - $m_p$：一个 7B transformer 的权重

**核心组件与符号映射**：

| 组件 | 论文形式 | 框架符号 |
| --- | --- | --- |
| 短期记忆 | $M_{STM} = \lbrace q_{t-K+1}, \ldots, q_t \rbrace$ | $m_s$ |
| 长期记忆 | $\mathcal{M} = (\lbrace \mathcal{G}_k \rbrace, \mathcal{C})$ | $m_l$ |
| 参数化记忆 | $\mathcal{M}_{\text{parametric}}^t$ (7B LLM weights) | $m_p$ |
| 检索 | RAG (embedding) + KG multi-hop | $v$ |
| 长期记忆更新 | $\mathcal{G} = \Phi_{LLM}(\mathcal{C})$ | $g_l$ |
| 短期记忆更新 | sliding window | $g_s$ |
| 调度 | Orchestrator（rule-based） | 手工规则 |

---

## 2. Training Procedure

- **优化的组件**：**仅有参数化记忆模型 $m_p$ 被训练**。主 LLM（Qwen / GPT-4o-mini）作为下游 consumer，不被优化；orchestrator、KG 构建过程、检索过程均无可训练参数。
- **优化算法**：**标准监督微调（SFT）**，token-level cross-entropy。无 RL，无 preference learning。
- **训练数据来源**：从 $m_l$ 的 RAG 链路离线构造 $(q, R, \hat{R})$ 三元组，$R$ 是 RAG 检索结果，让 $m_p$ 学会直接从 $q$ 生成 $R$。数据源自 ScienceQA 训练集和 LoCoMo 对话数据。
- **是否冻结 LLM**：主 LLM 完全冻结。被更新的只有参数化记忆模型 $m_p$（Qwen2.5-7B 作为底座）的全部权重。
- **核心目标函数**：

$$\mathcal{L}_{\text{update}} = -\sum_{t=1}^{T} \log P_\Theta(r_t \mid q, r_{< t})$$

（统一符号）：

$$\mathcal{L}_{\text{SFT}}(m_p) = -\sum_{t=1}^{T} \log \pi_{m_p}(r_t \mid q, r_{< t})$$

其中 $r_t$ 是 RAG 检索结果序列的第 $t$ 个 token。动态扩展机制：

$$\mathcal{M}_{\text{parametric}}^{t+1} = \mathcal{M}_{\text{parametric}}^{t} + \Delta\Theta_t$$

即随着 $m_l$ 的扩张，定期用新的 $(q_t, R_t)$ 对 $m_p$ 做增量 SFT。

> 把非参数外部记忆蒸馏进参数化模块，与 Knowledge Modules DCD 的思路类似，但 MemVerse 的 $m_p$ 是独立可插拔的小模型，而非主 LLM 的 adapter。

---

## 3. Reward Signal

- **奖励类型**：**本文不使用 RL，没有传统意义的 reward**。训练信号是 SFT 的 token-level 监督（目标序列来自 RAG 检索结果）。
- **奖励来源**：若把监督目标视作"reward proxy"，其来源是**离线 RAG 系统的输出**——即用 $m_l$ 上的检索结果作为"老师"。
- **credit assignment**：token-level CE，均匀分配到每个 token，无显式 credit assignment 机制。
- **辅助奖励 / 正则项**：无（仅有训练层面的 gradient clipping、cosine LR schedule 等工程 trick）。

---

## 4. Inference Procedure

- **记忆初始化**：
  - $m_s$：空的滑动窗口，随对话填充
  - $m_l$：从训练阶段的 MMKG 构建产物加载
  - $m_p$：已 SFT 好的 7B 模型权重
- **每步决策流水线**（由 orchestrator 按规则执行）：
  1. 多模态输入 $M$ 经 MLLM 转文本 $S = D_{\text{text}}(\mathcal{A}(E_{\text{mod}}(M)))$
  2. 新 query $q_t$ 进入 $m_s$，淘汰最老条目
  3. Orchestrator 按规则决定是否访问 $m_l$ 或 $m_p$：
     - 若需要快速响应 → 走 $m_p$（2.28s）
     - 若需要多跳 / 最新信息 → 走 $m_l$（KG 遍历 + RAG）
  4. 检索到的 $R$ 与 $q$ 拼接送入主 LLM 生成 $\hat{A}_N$
  5. 对话结束或积累足够新知识时，触发 $g_l$：$\mathcal{G} = \Phi_{LLM}(\mathcal{C})$ 更新 KG；定期触发 $m_p$ 的 SFT 更新
- **额外推理策略**：Top-K embedding 检索 + KG 邻居扩展（multi-hop），query rewrite（把检索内容拼回 query 重查）。
- **决策来源**：**完全手工规则**，$\pi$ 不存在（唯一学习得到的 $m_p$ 只是一个"快速替身检索器"，不参与调度决策）。

---

## 5. RQ 分析

### RQ1 (What is memory?)

MemVerse 同时具备三种记忆表征：(1) 固定长度 token 窗口的 $m_s$（T2 固定长度压缩）；(2) 非参数化多层级知识图 + raw chunks 的 $m_l$（T1 非参数外部记忆，同时具有 flat + 图结构）；(3) 从 $m_l$ 蒸馏而来的独立参数化小模型 $m_p$（T4 参数化 + T5 独立可插拔辅助模块的混合）。

### RQ2 (Which component is optimized?)

只有参数化记忆小模型 $m_p$ 被 SFT 训练，它既不是主 LLM，也不是 RAG 式的管理器（它不做记忆管理决策，而是模仿 RAG 检索结果）。主 LLM、orchestrator、检索算法均未被训练。最接近 O2（主 LLM 参数优化）中的 "LLM 被训练" 形态，但被训练的不是主 LLM 而是独立的蒸馏模型；也可以视为 O5 之外的一种新子类——"被训练的参数化记忆替身"。

### RQ3 (Target of Optimization)

最终追求 (1) 多模态 QA 的回答准确度（G1，ScienceQA 85.48%、MSR-VTT R@1 90.4%）；(2) 推理效率（G2，RAG 20s → 参数化记忆 2.28s，加速 89%）。不直接优化写入决策或行动奖励。

### RQ4 (Training Signal and RL Algorithm)

无 RL，纯 SFT / 蒸馏。训练信号 = RAG 检索结果的 token-level CE。归入 N/A。

### RQ5 (How memory evolves, operates?)

- $g_s$：sliding window，每步丢掉最老 query
- $g_l$：定期或知识量达到阈值时，LLM 从新 chunks 抽实体 / 关系并并入 KG
- $m_p$ 更新：$\mathcal{M}_{\text{parametric}}^{t+1} = \mathcal{M}_{\text{parametric}}^{t} + \Delta\Theta_t$，定期增量 SFT
- 读：orchestrator rule-based 路由到 $m_s$ / $m_l$ / $m_p$ 之一或其组合
- 调度策略完全手工，无遗忘策略（除窗口淘汰），但 KG 可做语义去重

---

## Conclusion

MemVerse 是一个即插即用的多模态记忆框架，它的核心想法是模仿 Kahneman 的"快慢系统"：慢路径是一个分层的多模态知识图（core / episodic / semantic 三类），用来结构化地沉淀长期经验；快路径是一个 7B 小模型，定期把知识图里的检索结果蒸馏进自己的权重，从而在推理时不用再走 RAG 就能找到相关知识。短期记忆则用一个简单的滑动窗口处理当前对话。所有的读写调度都是规则驱动的，唯一的学习是把 RAG 结果蒸馏到一个小模型里，因此它更偏 system / 架构贡献。