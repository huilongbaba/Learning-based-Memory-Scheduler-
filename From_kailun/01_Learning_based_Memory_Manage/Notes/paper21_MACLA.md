# MACLA: Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement

**Source:** [arXiv:2512.18950v1](https://arxiv.org/abs/2512.18950) (2025-12-22 修订，accepted by AAMAS 2026)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$\tau = \lbrace (o_t, a_t, r_t) \rbrace_{t=0}^{T}$|$T$|一条 episodic trajectory（observation-action-reward 序列）|
|$o_t$|$o$|观测|
|$a_t$|$a$|动作|
|$\mathcal{L}_\theta$（frozen LLM）|$M$, $\pi_\theta$|冻结的主 LLM（负责 segmentation/abstraction/action formatting）|
|$\text{Proc}_k = \langle \mathcal{G}_k, \Psi_k, \pi_k, \Phi_k \rangle$|$e_i \in m_l$|单条结构化程序记忆条目（goal / preconditions / action schema / postconditions）|
|$\mathcal{M}_\text{proc} = \lbrace (\text{Proc}_i, \mathbf{e}_i, \alpha_i, \beta_i) \rbrace$|$m_l$|程序记忆库（长期外部记忆）|
|$\text{MP}_j = \langle \mathcal{G}_j^\text{meta}, \Psi_j^\text{meta}, \lbrace \text{Proc}_{i_k} \rbrace, \Theta_j \rangle$|$m_l$（层级化扩展）|元程序（playbook），包含组合控制策略 $\Theta_j$|
|episode buffer（$N_b = 1000$）|$m_s$|短期工作记忆（最近观测/动作）|
|$\phi$（SentenceTransformer 编码器）|—（嵌入函数）|将文本映射为向量|
|ANN search + cosine similarity|$v$|检索算法|
|$\text{Beta}(\alpha_i, \beta_i)$|—（新增：可靠性后验）|程序 $i$ 成功率的 Beta 后验|
|$\text{EU}(\text{Proc}_i \mid o_t)$|—（新增：期望效用）|用于排序的期望效用打分|
|ContrastiveExtract / UpdateBeta|$g_l$|长期记忆更新函数（refinement + Bayesian 更新）|
|ZeroShotStep|$\pi_\theta$ 的 fallback 路径|当最优 EU 低于阈值时退回到纯 LLM 推理|

---

## 概览

这篇文章用了历史交互轨迹作为原始素材：先用一个冻结的 LLM 对轨迹做语义分割和抽象，把每段子任务抽取成结构化的程序（goal / preconditions / action schema / postconditions），存入一个外部的层级化程序记忆库。  
这篇文章最后得到了一个外部层级化程序记忆系统，里面存着可复用的 human-readable 程序。LLM 本身完全不被训练，所有学习都发生在外部记忆的更新（Bayesian 更新 + 对比式精化 + 元程序组合）。  
这篇文章优化/追求的目标是：在冻结 LLM 的前提下，通过外部记忆的演化提升 agent 在下游任务（ALFWorld / WebShop / TravelPlanner / InterCodeSQL）上的最终成功率，同时大幅降低训练成本（比 SOTA 基线快 2800 倍）。

---

## 1. Problem Setting

- **处理的 memory 类型**：cross-episode 长期记忆 $m_l$（程序库 $\mathcal{M}_\text{proc}$ + 元程序库 $\mathcal{M}_\text{meta}$）为主，辅以 in-episode 的 episode buffer（$m_s$）提供短期上下文。
- **决策过程建模**：严格来说不是标准 RL 形式化；论文使用**贝叶斯决策过程**框架——每一步在候选程序集合上计算期望效用 $\text{EU}$ 并贪婪选择，没有 policy gradient，没有 value network。可视为一种 **contextual Bayesian bandit**（每条程序 $i$ 是一个 arm，上下文为当前观测 $o_t$，奖励通过后验 $\text{Beta}(\alpha_i, \beta_i)$ 估计）。
- **状态/动作/观测空间**：
    - $\mathcal{S}$：环境状态（由 $o_t$ 部分观测）
    - $\mathcal{A}$：既包含环境原语动作，也包含"选择哪条程序/元程序执行"的选择
    - $\Omega$：文本观测
- **记忆的数据结构**：**层级化结构化程序库**——既不是纯向量库也不是纯自然语言条目，每条程序是一个四元组 $(\mathcal{G}, \Psi, \pi, \Phi)$（目标 / 前置条件 / 动作序列模板 / 后置条件），附带 Beta 后验 $(\alpha, \beta)$、embedding $\mathbf{e}$、success/failure context sets $S_i, F_i$。元程序再在其上叠加组合控制策略 $\Theta_j \in \lbrace \text{continue, skip, repeat, abort} \rbrace$。

核心组件映射：

|组件|论文实现|框架符号|
|---|---|---|
|短期工作记忆|Episode buffer（$N_b = 1000$ 步）|$m_s$|
|长期外部记忆|$\mathcal{M}_\text{proc}$ + $\mathcal{M}_\text{meta}$|$m_l$|
|检索算法|ANN + cosine similarity on $\mathbf{e}$|$v$|
|短期更新|每步追加观测|$g_s$|
|长期更新|Beta 更新 + Contrastive Refinement + Meta 抽取 + 剪枝|$g_l$|
|生成模型|冻结 Llama-2-7B|$M$（$\theta$ 不变）|

> MACLA 的 memory schema 像 RAG 但不只是 RAG，显式建模 preconditions 和 postconditions，使得记忆具有"可执行语义"而非仅仅是文本片段。

---

## 2. Training Procedure

- **优化的组件**：**$m_l$（外部程序记忆库）本身**——既不训练 $\theta$，也不训练独立的 memory manager 神经网络。所有"学习"是对 $m_l$ 中条目属性（$\alpha_i, \beta_i, \Psi_i, \pi_i, \Phi_i$）的**符号化/统计式编辑**。
- **优化算法**：
    1. **Bayesian conjugate update**（Beta-Bernoulli 共轭先验）：$(\alpha_i, \beta_i) \leftarrow (\alpha_i + y, \beta_i + (1-y))$，其中 $y \in \lbrace 0, 1 \rbrace$ 是执行成功信号。
    2. **Contrastive Refinement**：LLM 对比 $S_i$ 与 $F_i$ 提取差异，编辑 $\Psi_i, \pi_i, \Phi_i$。
    3. **Meta-Procedural Composition**：发现频繁共现的程序序列，抽象为元程序。
    4. **Utility-based Pruning**：定期用综合打分剪枝低质量条目。
- **训练数据来源**：离线交互轨迹（ALFWorld 2851 条训练 trajectory 等），但支持在线增量更新（inference-time learning）。
- **LLM 参数是否冻结**：**是**。LLM 参数完全不更新；"被优化"的对象是外部记忆数据结构。
- **核心训练目标函数**：

本文**没有一个可微的 loss 函数**。最接近 objective 的是 Bayesian 期望效用（用于决策，不是训练目标）：

$$ \text{EU}(\text{Proc}_i \mid o_t) = \text{Rel}_i(o_t) \cdot \frac{\alpha_i}{\alpha_i + \beta_i} \cdot R_\max - \text{Risk}_i(o_t) \cdot \frac{\beta_i}{\alpha_i + \beta_i} \cdot C_\text{fail} + \lambda_\text{info} \cdot H[\text{Beta}(\alpha_i, \beta_i)] $$

其中三项分别对应：期望收益 / 失败代价 / 信息增益（探索项）。用框架符号重写：

$$ \text{EU}(e_i \mid o_t) = \text{Rel}(v(o_t, e_i)) \cdot \mathbb{E}[\rho_i] \cdot R_\max - \text{Risk}(o_t, e_i) \cdot (1 - \mathbb{E}[\rho_i]) \cdot C_\text{fail} + \lambda_\text{info} \cdot H[\mathbf{b}_i] $$

剪枝 utility：

$$ U(\text{Proc}_i) = \lambda_r \cdot \frac{\alpha_i}{\alpha_i + \beta_i} + \lambda_f \cdot \frac{n_i}{N_\text{total}} + \lambda_t \cdot e^{-(t_\text{current} - t_i^\text{last})/\tau} $$

其中 $\lambda_r = 0.5, \lambda_f = 0.3, \lambda_t = 0.2$（由 grid search 确定）。

---

## 3. Reward Signal

- **奖励类型**：**step-level binary success signal** $y \in \lbrace 0, 1 \rbrace$，来自**环境反馈**（precondition check + postcondition check）。与典型 sparse terminal reward 不同，MACLA 对**每次程序执行**独立打分（precondition 满足 + postcondition 满足 → $y = 1$）。
- **奖励来源**：
    - **环境验证**：`CheckPre(Ψ, o)` 和 `CheckPost(Φ, o')` 由环境或规则检查
    - **外部任务 reward**：ALFWorld 等 benchmark 本身提供的 terminal task reward 用于报告性能，但系统内部用于 Bayesian 更新的是每步 $y$
- **奖励如何分配**：**step-level credit assignment 由记忆结构天然提供**——每条程序独立维护 $(\alpha_i, \beta_i)$，只有被调用并执行的程序才更新其后验。这避免了"整条轨迹用 terminal reward 折现"的问题。元程序也有独立的 $(\alpha_j, \beta_j)$。
- **辅助奖励/正则项**：
    - **信息增益项** $\lambda_\text{info} \cdot H[\text{Beta}(\alpha_i, \beta_i)]$：鼓励探索低证据程序（$\lambda_\text{info} = 0.1$）
    - **失败代价项** $C_\text{fail} \cdot (1-\rho)$：风险规避项（$C_\text{fail} = 0.5$）
    - **Relevance / Risk 上下文项**：用当前观测与历史成功/失败上下文的相似度调制效用

---

## 4. Inference Procedure

- **记忆初始化**：推理时加载离线构建好的 $\mathcal{M}_\text{proc}$（187 条）和 $\mathcal{M}_\text{meta}$（43 条）；episode buffer 清空。
- **每步决策流程**（Algorithm 1）：
    1. 观测 $o_t$，编码为 $\mathbf{h} = \phi(o_t)$
    2. ANN 检索 top-$k$ 候选 $\mathcal{C}_t = \lbrace \text{Proc}_i, \text{MP}_j \rbrace$
    3. 对每个候选计算 $\text{EU}(c \mid o_t)$
    4. 选 $c^* = \arg\max \text{EU}$
    5. 如果 $\text{EU}(c^*) < \theta_\text{conf}$：fallback 到 zero-shot LLM
    6. 否则：验证前置条件 → 执行 action schema → 验证后置条件 → 得到 $y$
    7. 更新 $(\alpha, \beta)$，记录 context 到 $S_i$ 或 $F_i$
    8. 若 $\min(\lvert S_i \rvert, \lvert F_i \rvert) \geq 3$：触发 Contrastive Refinement
    9. 循环直到任务完成或达到 horizon
- **推理时额外策略**：
    - **Top-$k$ ANN 检索**（sublinear $O(\log N_p)$）
    - **Confidence threshold fallback**：$\theta_\text{conf} = 0.4$
    - **Ontological semantic grounding**：跨词汇等价映射（mug↔cup↔glass）
    - **Meta-procedure hierarchical execution**：若选中元程序，按组合策略 $\Theta_j$ 递归执行
    - **Inference-time learning**：每步都更新 Beta 后验，每个 episode 结束后抽取新元程序并剪枝
- **学习 vs. 手工规则**：**混合**——Bayesian 选择逻辑、剪枝公式、阈值（$\theta_\text{dup}, \theta_\text{conf}, \theta_\text{meta}$）都是手工设定；但程序内容、preconditions、action schema 均由 LLM 在构建期生成，并被对比式精化迭代改进。

> MACLA 的 inference loop 与传统 RL agent 最大的差别是：策略 $\pi$ 是一个在记忆条目上的闭式贝叶斯决策函数，而非神经网络。这使得整个系统高度可解释。

---

## 5. RQ 分析

### RQ1 (What is memory?)

MACLA 主要使用 T1（非参数化外部记忆） 的层级化变体：$m_l$ 是一个外部的、可 CRUD 的、离散条目集合 $\mathcal{M}_\text{proc} = \lbrace \text{Proc}_i \rbrace$，每条是结构化的四元组（$\mathcal{G}, \Psi, \pi, \Phi$）而非自由文本。另有 $\mathcal{M}_\text{meta}$ 作为图/层级结构的扩展。$m_s$ 是固定 1000 步的 episode buffer，提供局部上下文，但不被压缩（与 T2 不同）。

### RQ2 (Which component is optimized? Which signal is used?)

被优化对象：$m_l$ 本身（程序记忆库的内容 + 每条程序的 Beta 后验）。主 LLM 完全冻结，既不训练 $\theta$ 也不训练一个独立的 neural memory manager。从分类角度看最接近 O1（可插拔管理器），但更精确地说应该是是"记忆数据结构本身作为可更新对象"。

### RQ3 (Target of Optimization)

主要追求 G1（回答/任务准确度），ALFWorld / WebShop / InterCodeSQL 的 task completion reward，TravelPlanner 的 CS score。同时显著优化 G2（效率）：2800× 更少训练时间、更少 LLM 调用次数（每 episode 6.2 次 vs. ReAct 16-20 次）。间接涉及 G4（写入决策质量）：utility-based pruning 确保只保留高质量程序（retained 程序平均 $\rho = 0.79$ vs. pruned 的 0.42）。

### RQ4 (Training Signal and RL Algorithm)

本文不使用任何 RL 算法。无 policy gradient、无 value network、无 Bellman 更新。训练信号通过 Beta-Bernoulli 共轭贝叶斯更新转化为 arm 可靠性估计，通过 expected-utility maximization 转化为选择决策。归入 N/A。

### RQ5 (How memory evolves, operates?)

记忆在推理时以四种方式演化：

1. **Bayesian 后验更新**：每次程序执行后 $(\alpha_i, \beta_i) \leftarrow (\alpha_i + y, \beta_i + (1-y))$
2. **Contrastive Refinement**：当 $\min(\lvert S_i \rvert, \lvert F_i \rvert) \geq 3$ 时，LLM 对比成功/失败上下文，编辑 $\Psi_i, \pi_i, \Phi_i$
3. **Meta-procedure 抽取**：频繁共现序列被抽象为组合型元程序 MP$_j$
4. **Utility-based Pruning**：定期按综合 utility 剪枝低价值条目（73% pruned 程序成功率 < 0.5）

操作上：读取通过 ANN + cosine similarity，写入通过 LLM-guided segmentation + 重复检测合并。

---

## Conclusion

MACLA 是一个"去训练化"的 agent 记忆框架：它不更新 LLM 参数，把所有学习搬到一个外部的、层级化的、结构化程序 RAG 里。每条程序是一个带有前置/后置条件的可执行骨架，维护一个 Beta 后验刻画其可靠性；agent 在推理时用贝叶斯期望效用挑选最佳程序，通过对比成功/失败样本精化程序内容，并自动抽取出高层组合型 playbook。它在四个 benchmark 上用 7B 小模型达到 SOTA（78.1% 平均），同时训练速度比 SOTA 基线快 2800 倍，并产出完全可解释的人类可读知识库。