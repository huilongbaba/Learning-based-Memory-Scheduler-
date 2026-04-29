# HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents

**Source:** [arXiv:2604.16839](https://arxiv.org/abs/2604.16839) （2026-04-18 修订，Accepted to ACL 2026）

## 符号映射表

|论文原始符号 / 概念|框架符号|说明|
|---|---|---|
|Conversation turn (node $v_i$)|$m_l$ 中的条目 $e_i$|每个 turn 是 episodic graph 的一个节点，含原文、embedding、时间戳、关键词、speaker|
|Episodic Memory Graph $G=(V,E,W)$|$m_l^{\text{epi}}$（图结构子库）|外部、可寻址、CRUD 的图状记忆，节点为 turn，边为 Hebbian 关联|
|Semantic Memory Store|$m_l^{\text{sem}}$（结构化子库）|由 Reflective Agent 蒸馏出的 user model / factual memory / agent knowledge|
|Edge weight $w_{ij}$|$m_l$ 中的关联结构 $\text{struct}_{ij}$|由 Hebbian 规则在线更新；论文中扮演"突触强度"|
|Hebbian update rule (Eq. 1)|$g_l$|长期记忆更新函数；本文是**无梯度的启发式** $w_{ij}^{(t+1)} = (1-\lambda) w_{ij}^{(t)} + \eta \cdot \mathbb{I}(\cdot)$|
|Reflective Agent|独立模块（接近 T5 但**未训练**）|监控图结构、触发 Hebbian Distillation、Adaptive Forgetting|
|Spreading Activation (Eq. 4)|$v$（检索算法）|双路径检索：base path + flip path|
|Final retrieval set $\mathcal{R}_{\text{final}}$|$C_N$|注入主 LLM 的上下文|
|Response generation|$\hat{a}_N$|由主 LLM 基于 $C_N$ 直接生成|

---

## 概览

这篇文章用了多 session 的对话历史（每个 turn 作为一个节点）作为原始数据，通过手工设计的 Hebbian 在线更新规则和一个prompt 驱动的 Reflective Agent，把对话历史增量地组织成一个动态关联图。  
最后得到了一个完全 training-free 的双层记忆架构：底层是带可演化边权的 episodic graph，顶层是从图中 hub 节点簇蒸馏出来的 semantic store；附带一个 dual-path 检索器（语义相似度 + spreading activation）。系统中没有任何被训练的参数，主 LLM 冻结，Reflective Agent 也只是 prompt-LLM。  
在 LoCoMo 和 LongMemEval-S 上以远低于 baseline 的 token 预算（~1k tokens vs. 16k）取得更高的 F1/BLEU/Accuracy，特别在 multi-hop 与 temporal 类问题上提升显著。其优化完全是架构层面而非参数层面的。

---

## 1. Problem Setting

- **记忆类型**：处理的是 **cross-chat / 跨 session 的长期记忆 $m_l$**，目标是在长达 ~300 turns、~9k tokens 的多 session 对话中保持人物一致性、事实正确性、时间感知。$m_s$ 在本文中没有显式建模。
- **决策过程的形式化**：本文**没有把记忆管理建模为 MDP/POMDP/bandit**。Hebbian 更新是**确定性的、无 reward 的在线规则**；hub 检测和 forgetting 是基于阈值的硬决策。整个系统更接近一个 _rule-based dynamical system_。
- **状态 / 动作 / 观测空间**：不存在严格意义上的 RL 状态空间。若强行映射：状态 = 当前图 $G^{(t)}$；"动作" = Hebbian 更新 / Distillation / Forgetting / Retrieval；这些"动作"是规则触发的，不是 policy 输出。
- **记忆数据结构**：
    - **Episodic Memory Graph**：节点 = `(text, embedding, timestamp, keywords, speaker_role)`；边 = Hebbian weight $w_{ij} \in \mathbb{R}$。
    - **Semantic Memory Store**：结构化记录三类（User Model / Factual Memory / Agent Knowledge），每条带 confidence + evidence link 回 episodic 节点。

|组件|论文符号|框架符号|
|---|---|---|
|对话 turn|node $v_i$|$e_i \in m_l$|
|边权矩阵|$W = \lbrace w_{ij} \rbrace$|$\text{struct}(m_l)$|
|蒸馏出的语义条目|semantic record|$e_i \in m_l^{\text{sem}}$|
|检索得分|$S(v_j)$|$\text{score}(q, e_j)$|
|上下文|$\mathcal{R}_{\text{final}}$|$C_N$|

---

## 2. Training Procedure

**本文不进行任何参数训练。** 整个系统是 training-free 的：

- 主 LLM（GPT-4o-mini / GPT-4o / Qwen2.5-{3b,14b}）权重 $\theta$ **完全冻结**，仅做 prompted inference。
- Reflective Agent 也是同一类 LLM，只通过 prompt（见 Appendix A）触发，**没有独立参数被优化**。
- Hebbian 更新规则中的所有量（$\eta=0.02, \lambda=0.995, \beta=0.1$ 等）是**手工选定的固定超参**，不通过梯度学习。
- 没有梯度、没有 SFT、没有 RL、没有偏好对齐——属于 RQ4 中的 **N/A: Non-RL Methods**。

**唯一带"学习"字样的是 Hebbian 边权更新**（Eq. 1）：

$$ w_{ij}^{(t+1)} = \underbrace{(1-\lambda), w_{ij}^{(t)}}_{\text{synaptic decay}} + \underbrace{\eta \cdot \mathbb{I}(v_i, v_j \in \mathcal{K}_t)}_{\text{active reinforcement}} $$

但这是**无 reward、无梯度的在线启发式**——它只对应 $g_l$ 的实现，不构成"训练"。

---

## 3. Reward Signal

**本文没有 reward signal**。原因同上，系统不做任何参数优化，因此也不需要 reward。

只有 evaluation metrics（F1, BLEU-1, accuracy）用于离线测评，但这些指标**只用于报告结果，不参与任何训练循环**。

---

## 4. Inference Procedure

推理时记忆完全在线增量构建，无离线预处理。

**记忆初始化**：空图 $G^{(0)} = (\emptyset, \emptyset, \emptyset)$。

**每一步流程**（每来一条新 turn）：

1. **Online Encoding**：将新 turn 编码为节点 $v_t$，加入 $G$；与最近若干 turns 用小初始权重连边。
2. **Co-activation update**：每次检索后，对当前 retrieval set $\mathcal{K}_t$ 内的所有节点对应用 Hebbian 规则（Eq. 1）增强权重。
3. **Reflective Consolidation**（条件触发）：当某节点累积权重 $D(v_i) = \sum_{j} w_{ij} > \delta_{\text{hub}}$ 时，触发 Hebbian Distillation：
    - Reflective Agent 抽取该 hub 及其强连邻居
    - LLM 生成 declarative semantic entry，写入 $m_l^{\text{sem}}$
4. **Adaptive Forgetting**（条件触发）：节点同时满足 (a) 总权重 $< \delta_{\text{prune}}$、(b) inactive 时长 $> \delta_{\text{age}}$、(c) 零最近访问 → 删除。

**查询时检索流程**（dual-path）：

$$S_{\text{base}}(v_i) = \big(\text{sim}(\mathbf{q}, \mathbf{e}_i) + \alpha \cdot \text{kw\_match}\big) \cdot \gamma(v_i)$$

$$S(v_j) = S_{\text{base}}(v_j) + \beta \sum_{i \in \mathcal{N}(j)} S_{\text{base}}(v_i) \cdot w_{ij}$$

$$\mathcal{R}_{\text{final}} = \text{Top-}k(S_{\text{base}}) \cup \text{Top-}m(S \mid v \notin \text{Top-}k)$$

最后将 $\mathcal{R}_{\text{final}}$ 与若干 semantic records 拼接成 $C_N$，由主 LLM 直接生成 $\hat{a}_N$。

**手工规则比例**：极高。所有阈值 $(\delta_{\text{hub}}, \delta_{\text{prune}}, \delta_{\text{age}}, \theta_{\text{spread}})$ 与超参 $(\eta, \lambda, \alpha, \beta, \tau, k, m)$ 均人工设定。

---

## 5. RQ 分析

### RQ1 (What is memory?)

记忆是一个外部、显式、graph 结构的双层 store：底层 episodic graph 用 Hebbian 边权刻画关联；顶层 semantic store 由 hub 蒸馏产生的结构化记录。两层都属于 T1 的"非参数化、外部、可寻址"条件。Reflective Agent 在架构上是独立可插拔的记忆管理模块（T5），但其本身未被训练，仅以 prompt 形式驱动。

### RQ2 (Which component is optimized? Which signal is used?)

本文未涉及此问题。没有任何参数被优化。Hebbian 更新规则是手工启发式，主 LLM 与 Reflective Agent 都冻结。属于 training-free 工作。

### RQ3 (Target of Optimization)

本文未涉及此问题，没有训练就没有优化目标。

### RQ4 (Training Signal and RL Algorithm)

N/A: Non-RL Methods。无任何 RL 算法。

### RQ5 (How memory evolves, operates?)

记忆随时间双时间尺度演化：

- 快尺度（每次 retrieval 后）：Hebbian 边权按 Eq. 1 增强 + 衰减 → 关联结构涌现；
- 慢尺度（hub 累积到阈值时）：Reflective Agent 触发 Distillation，把高度互联的 episodic 簇压缩成 semantic 记录；同时 Adaptive Forgetting 剪枝无关节点。

读取时通过 dual-path 检索（base similarity + spreading activation 沿 Hebbian 边传播），实现"语义相似"与"历史共激活"的互补召回。

---

## Conclusion

HeLa-Mem 是一篇用神经科学比喻尝试取代 RL 的 memory 工作：它把对话历史建模成一个动态图，用经典 Hebbian 规则（"fire together, wire together"）让经常被一起检索的 turns 之间长出强边；同时用一个 prompt-only 的 Reflective Agent 周期性地把 hub 节点簇压缩成结构化语义条目，并把没人理的孤立节点删掉。检索时既走语义相似度，又沿 Hebbian 边做 spreading activation，从而能召回"语义不相似但历史上一起出现过"的记忆，对 multi-hop 类问题尤其有效。整套系统不训练任何参数，但靠精心设计的规则在 LoCoMo 上以 ~1k token 预算超过了用 16k token 的 baseline。  
所有阈值/超参是手调的，跨数据集迁移性存疑。HeLa-Mem 与 A-MEM 共享 graph-based 思路但更强调 edge dynamics，与 HyperRAG 都涉及图检索但 HyperRAG 是 hypergraph 结构。