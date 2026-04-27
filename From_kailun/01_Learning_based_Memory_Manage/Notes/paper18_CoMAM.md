# Collaborative Multi-Agent Optimization for Personalized Memory System (CoMAM)

**Source:** [arXiv:2603.12631v1](https://arxiv.org/abs/2603.12631) (2026-03-13)

---
## 符号映射表

|论文原始符号|含义|框架符号|
|---|---|---|
|$H = \lbrace h_1, \ldots, h_N \rbrace$|长期对话历史|原始 context，是 $m_l$ 的构建来源|
|$q$|用户后续查询|$q$|
|$p$|生成的个性化回答|$\hat{a}_N$（最终 answer）|
|$p^*$|标准答案|$a^*$|
|$\mathcal{M}_f$|细粒度记忆（事实/事件）|$m_l^{\text{fine}}$（$m_l$ 的子存储）|
|$\mathcal{M}_c$|粗粒度记忆（用户偏好/画像）|$m_l^{\text{coarse}}$（$m_l$ 的子存储）|
|$\mathcal{M} = \lbrace \mathcal{M}_f, \mathcal{M}_c \rbrace$|多粒度记忆集合|$m_l$|
|$\mathcal{E}_q \subseteq H$|与 $q$ 相关的证据集合|检索 ground truth|
|$\mathcal{E}'_q = \lbrace m_k \rbrace_{k=1}^{K}$|检索到的 top-$K$ 记忆|$v(q, m_l)$ 的输出|
|$\pi_{\theta}^{\text{cons}_f}$|细粒度构建 agent 策略|$\pi^{(0)}$（写入 $m_l^{\text{fine}}$ 的策略）|
|$\pi_{\theta}^{\text{cons}_c}$|粗粒度构建 agent 策略|$\pi^{(1)}$（写入 $m_l^{\text{coarse}}$ 的策略）|
|$\pi_{\theta}^{\text{ret}}$|检索 + 回答 agent 策略|$\pi^{(2)}$（合并 $v$ 与生成）|
|$\mathcal{V}$|冻结的打分 LLM|LLM-as-judge reward model|
|$r_{\text{cons}_f}, r_{\text{cons}_c}, r_{\text{ret}}$|局部任务奖励|各 agent 的 local reward|
|$r_{\text{ans}} = \mathbb{I}(p^* = p)$|全局回答奖励|terminal reward|
|$w_n$|基于 NDCG 的 credit 权重|自适应 credit assignment 权重|

> 本文引入了一个新概念：跨 agent 的 credit assignment 机制（通过 NDCG 衡量 local reward 与 global reward 的排序一致性）。

---

## 概览

这篇文章用了用户长期对话历史 $H$ 和后续查询 $q$ 及标准答案 $p^*$。训练过程中从三个 agent 组成的 pipeline 采样 $G$ 条 MDP 轨迹 $\lbrace \tau_i \rbrace$，每条轨迹记录"细粒度抽取 - 粗粒度画像 - 检索+回答"三步的 state-action 序列，并收集每步的 local reward 和最终的 global accuracy reward。  
最后得到了一个由三个协同优化过的 LLM agent 组成的个性化记忆系统：一个细粒度抽取 agent、一个粗粒度画像抽象 agent、一个检索+回答 agent。三个 agent 参数均被更新。
优化的核心目标是个性化问答准确率（$\mathbb{I}(p^*=p)$），但同时通过 local reward（信息覆盖度、抽象合理性、检索精度）和 adaptive credit assignment 机制，在保证每个 agent 局部任务能力的同时，让三个 agent 的改进方向与全局准确率对齐，解决局部最优不等于全局最优的问题。

---

## 1. Problem Setting

- **记忆类型**：cross-chat memory（$m_l$），以多组件/多层级形式存在。$m_l$ 被显式拆分为两个子存储：$\mathcal{M}_f$（细粒度事实性记忆）和 $\mathcal{M}_c$（粗粒度用户画像）。不涉及 $m_s$（短期工作记忆）。
- **决策过程建模**：被显式建模为**序列 MDP**（sequential MDP），关键设计是将三个异构 agent 的异步执行"串起来"成一条 3 步的轨迹，把 agent 间的依赖关系编码到 state transition 里。
- **状态空间 $\mathcal{S}$**：
    - $s_0 = H$（Extraction 的初始状态，原始对话历史）
    - $s_1 = \mathcal{M}_f$（Profile 的状态，是上一步的产出）
    - $s_2 = (\mathcal{M}, q)$（Retrieval 的状态，包含多粒度记忆和用户查询）
    - $s_3 = (\mathcal{E}'_q, p)$（终止状态，检索结果和回答）
- **动作空间 $\mathcal{A}$**：$a_0 = \mathcal{M}_f$（写入细粒度记忆），$a_1 = \mathcal{M}_c$（写入粗粒度记忆），$a_2 = (\mathcal{E}'_q, p)$（检索 top-$K$ 并生成回答）。每个 action 本质是 LLM 生成的一段文本。
- **观测空间 $\Omega$**：在本文中 $\Omega = \mathcal{S}$，环境完全可观测（no partial observability）。
- **记忆数据结构**：自然语言文本条目的多层级集合，使用多粒度（fine / coarse）的划分方式。

**核心组件映射表：**

|组件|论文实现|框架符号|
|---|---|---|
|Long-term memory|$\mathcal{M} = \lbrace \mathcal{M}_f, \mathcal{M}_c \rbrace$|$m_l$（多组件）|
|Short-term memory|未使用|$m_s$ = ∅|
|Retriever|$\pi_{\theta}^{\text{ret}}$ 的 top-$K$ 选择|$v$|
|Writer (fine)|$\pi_{\theta}^{\text{cons}_f}$|$g_l^{\text{fine}}$|
|Writer (coarse)|$\pi_{\theta}^{\text{cons}_c}$|$g_l^{\text{coarse}}$|
|Main policy|$\pi_{\theta}^{\text{ret}}$（融合检索与回答）|$\pi$|

> 三个 agent 都是被训练的 LLM policy，它们共同承担记忆的写入、检索和使用

---

## 2. Training Procedure

- **优化的组件**：三个 agent 的策略参数 $\theta$ 全部被同时更新。主 LLM 参数不冻结；但也没有独立的检索器 $v$ 或外部 Q-network——检索行为是 Retrieval Agent LLM policy 的一部分。
- **优化算法**：GRPO（Group Relative Policy Optimization），每个 agent 各自使用 GRPO 目标，但共享同一批采样的 $G$ 条 MDP 轨迹。
- **训练数据来源**：在线交互——每个 $(H, q)$ 输入对会采样 $G$ 条轨迹（默认 $G=8$），收集局部和全局奖励后更新所有三个 agent。
- **是否冻结 LLM 参数**：不冻结。Extraction 和 Profile agent 用 Qwen2.5-3B-Instruct / Llama-3.2-3B-Instruct，Retrieval agent 用 Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct，全部参与训练。
- **核心训练目标函数**：

对每个 agent $n \in \lbrace 0, 1, 2 \rbrace$，GRPO 目标为：

$$ \mathcal{J}_{\text{GRPO}}(\pi_\theta^n) = \frac{1}{G} \sum_{i=1}^{G} \min\Big[ \rho_n^{(i)} A_n^{(i)},\ \text{clip}(\rho_n^{(i)}, 1 \pm \epsilon) A_n^{(i)} \Big] $$

其中 importance ratio $\rho_n^{(i)} = \dfrac{\pi_\theta^n(a_n^{(i)} \mid s_n^{(i)})}{\pi_{\theta_{\text{old}}}^n(a_n^{(i)} \mid s_n^{(i)})}$。

advantage $A_n^{(i)}$ 由 integrated reward 归一化得到：

$$ r_{\text{final}, n}^{(i)} = r_n^{(i)} + w_n \cdot r_3^{(i)} $$

其中 $r_n$ 是 agent $n$ 的 local reward，$r_3 = r_{\text{ans}}$ 是 global answer accuracy reward，$w_n$ 是通过 NDCG 计算的自适应权重。

> 每个 agent 的 advantage 虽然是各自计算的，但 global reward $r_3$ 同时进入三个 agent 的 advantage，实现 credit 的共享。

---

## 3. Reward Signal

- **奖励类型**：**混合 reward**——既有 dense 的 step-level local reward（每个 agent 在自己那一步后就能拿到 local reward），又有 sparse 的 terminal global reward（整个 MDP 结束后才有 answer accuracy）。
- **奖励来源**：
    - Extraction agent 的 $r_{\text{cons}_f}$：基于规则的 IoU+Recall 混合分数（使用相对于 $\mathcal{E}_q$ 的 F-like 指标）
    - Profile agent 的 $r_{\text{cons}_c}$：**LLM-as-judge**，由一个冻结的 LLM $\mathcal{V}$ 基于 rubric（场景、偏好、用户画像）打 $[0, 1]$ 分
    - Retrieval agent 的 $r_{\text{ret}}$：基于规则的 IoU+Recall 混合分数
    - Global 的 $r_{\text{ans}}$：EM-like 的 0/1 指标 $\mathbb{I}(p^* = p)$（多选题的选项匹配）
- **奖励如何分配到各步骤**：这是本文的核心贡献。不是简单均分，而是通过 NDCG-based adaptive credit assignment：

对每个 agent $n$，计算其 local reward 序列 $\mathcal{R}_{\text{local}, n} = \lbrace r_n^{(i)} \rbrace_{i=1}^{G}$ 与 global reward 序列 $\mathcal{R}_{\text{global}} = \lbrace r_3^{(i)} \rbrace_{i=1}^{G}$ 的排序一致性：

$$ v_n = \text{NDCG}(\sigma(\mathcal{R}_{\text{local}, n}), \sigma(\mathcal{R}_{\text{global}})) $$

然后通过 softmax 归一化为权重：

$$ w_n = \frac{\exp(v_n)}{\sum_{n'} \exp(v_{n'})} $$

最终 reward：$r_{\text{final}, n}^{(i)} = r_n^{(i)} + w_n \cdot r_3^{(i)}$

**直觉**：如果某 agent 的 local reward 排序与 global reward 排序高度一致（NDCG 高），说明它的局部改进确实能带动全局改进，那它应该分到更多的 global credit。

- **局部奖励具体形式**：

$$ r_{\text{cons}_f} = \alpha \cdot \frac{\lvert \mathcal{M}_f \cap \mathcal{E}_q \rvert}{\lvert \mathcal{E}_q \rvert} + (1 - \alpha) \cdot \frac{\lvert \mathcal{M}_f \cap \mathcal{E}_q \rvert}{\lvert \mathcal{M}_f \cup \mathcal{E}_q \rvert}, \quad \alpha = 0.8 $$

$$ r_{\text{ret}} = \beta \cdot \frac{\lvert \mathcal{E}'_q \cap \mathcal{E}_q \rvert}{\lvert \mathcal{E}_q \rvert} + (1 - \beta) \cdot \frac{\lvert \mathcal{E}'_q \cap \mathcal{E}_q \rvert}{\lvert \mathcal{E}'_q \cup \mathcal{E}_q \rvert}, \quad \beta = 0.2 $$

前者偏向 recall（$\alpha$ 大），后者偏向 precision。

---

## 4. Inference Procedure

- **记忆初始化**：对每个用户，Extraction agent 先一次性处理完整的对话历史 $H$，生成 $\mathcal{M}_f$；然后 Profile agent 从 $\mathcal{M}_f$ 抽象出 $\mathcal{M}_c$。$\mathcal{M} = \lbrace \mathcal{M}_f, \mathcal{M}_c \rbrace$ 在查询到来前就已构建完毕。
- **每步决策流程**：对于每个到来的查询 $q$：
    1. Retrieval agent 接收 $(\mathcal{M}, q)$
    2. 从 $\mathcal{M}$ 中选出 top-$K$ 相关记忆 $\mathcal{E}'_q$
    3. 基于 $\mathcal{E}'_q$ 和 $q$ 生成回答 $p$

Retrieval 和回答生成被合并为**同一个 LLM 的一次 rollout**（policy $\pi_\theta^{\text{ret}}$ 在 `<information>...</information>` 中输出检索内容，在 `<final_answer>...</final_answer>` 中输出选项），所以不需要单独的向量检索器。

- **额外策略**：在 prompt 中使用了结构化输出标记（`<information>`、`<final_answer>`），帮助区分检索和回答两个子步骤。评估时 temperature = 0.8；训练时 = 1.0。
- **策略驱动 vs. 手工规则**：推理流程本身是固定的（构建 → 检索 → 回答三步），但每一步的具体行为（什么算"重要信息"、如何抽象画像、如何选 top-$K$）完全由学习得到的 $\pi_\theta$ 决定。记忆的 CRUD 操作没有显式的规则层。

---

## 5. RQ 分析

### RQ1 (What is memory?)

本文的 memory 是 T2，$m_l$ 被拆成 $\mathcal{M}_f$（细粒度事实）和 $\mathcal{M}_c$（粗粒度画像）两个子存储，用自然语言文本条目存储。

### RQ2 (Which component is optimized? Which signal is used?)

优化的是主 LLM 本身（O2），三个 agent 都基于 LLM backbone 并被更新参数。信号是局部规则/模型奖励 + 全局 EM 奖励的自适应加权组合，通过 NDCG-based credit assignment 平衡。

### RQ3 (Target of Optimization)

主要目标是回答准确度（G1）（query-answer accuracy），次要关注训练效率（G2）（收敛步数）。局部任务能力（覆盖率、抽象合理性、检索精度）是辅助目标，服务于全局准确率。

### RQ4 (How memory evolves, operates?)

记忆在训练时演化（通过 RL 调整三个 agent 的 policy 来改进写入和检索行为），但在推理时是静态的（构建一次后不再更新）。没有 CRUD 操作；写入由两个 construction agent 完成，读取由 retrieval agent 完成。

---

## Conclusion

CoMAM 针对多 agent 记忆系统提出了一个协同强化学习框架，解决了"各 agent 独立优化导致全局次优"的问题。它把记忆系统的三个异步 agent（细粒度抽取、粗粒度画像、检索+回答）串成一条 3 步 MDP 轨迹，用 GRPO 同步训练三个 agent。关键创新是基于 NDCG 排序一致性的 adaptive credit assignment，不是简单地把全局 reward 等分给每个 agent，而是根据每个 agent 的 local reward 排序与 global reward 排序的吻合度，自适应分配全局 credit。本质上这是一个joint training 的多层级记忆写入+检索系统，通过联合优化让构建 agent 更好的与检索 agent 合作沟通。