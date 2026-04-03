# Memento: Fine-tuning LLM Agents without Fine-tuning LLMs

**Source:** [arXiv:2508.16153v2](https://arxiv.org/abs/2508.16153) (cs.LG, 25 Aug 2025)

---

## 符号映射表

| 论文原始符号                            | 统一符号        | 含义                                           |
| --------------------------------- | ----------- | -------------------------------------------- |
| $s_t$ (state)                     | $q$ / $C$   | 当前任务指令或上下文（state 包含 question 和已有 context）    |
| $a_t$ (action)                    | $a$         | LLM 生成的计划或工具调用                               |
| $c_t$ (case)                      | — (新符号，见下)  | 从 case bank 检索到的历史案例三元组 $(s_i, a_i, r_i)$    |
| $M_t$ (case bank / memory)        | $m_l$       | 长期记忆：不断增长的 episodic case bank                |
| Subtask Memory                    | $m_s$       | 短期记忆：当前任务的子任务及结果                             |
| Tool Memory                       | $m_s$       | 短期记忆：当前子任务的工具调用记录                            |
| $\mu(\cdot \mid s, M)$            | $v$         | 检索策略（case retrieval policy）                  |
| $p_{\text{LLM}}(\cdot \mid s, c)$ | $M$ (model) | 冻结的 LLM，条件于 state 和 retrieved case 生成 action |
| $\pi(a \mid s, M)$                | $\pi$       | CBR agent 的整体策略                              |
| $\theta$                          | $\theta$    | 检索模块参数（kernel 网络或 Q 网络的参数）                   |
| $Q(s, M, c)$                      | — (新符号)     | Q 函数，衡量在 state $s$ 下检索 case $c$ 的长期价值        |
| $\tau$                            | $T$         | 轨迹                                           |
| $\mathcal{R}(s, a)$               | —           | 奖励函数                                         |
| enc(·)                            | —           | 冻结的文本编码器（SimCSE）                             |
| $r_t$                             | —           | 即时奖励（二值: 0/1）                                |

**新增符号：**

- $c$：case（历史案例），三元组 $(s, a, r)$，代表一次完整的历史经验
- $\mathcal{B}$：case bank，即 $m_l$ 的具体数据结构
- $Q_\theta(s, c)$：参数化 Q 函数，用于评估 state-case 匹配质量
- $\mathcal{D}$：episodic memory，存储 $(s, c, Q)$ 元组，用于 kernel-based Q 估计

---

## 1. Problem Setting

**记忆类型：** 论文处理的是 **cross-chat memory**（$m_l$）。Case bank 在不同任务之间持续增长和复用。同时，Subtask Memory 和 Tool Memory 属于 **in-chat memory**（$m_s$），仅在当前任务内有效。（§3, §4.1）

**决策过程建模：** 论文将 CBR agent 的决策建模为 **Memory-Based Markov Decision Process (M-MDP)**（Definition 3.1, §3），定义为元组 $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, \mathcal{M} \rangle$，其中 $\mathcal{M} = (\mathcal{S} \times \mathcal{A} \times \mathbb{R})^*$ 是记忆空间。在实际实现中（§4.2），CBR 仅用于规划阶段，被简化为 **单步 MDP**（single-step setting）。

**状态空间 $\mathcal{S}$：** 所有有限长度词汇序列的集合，具体为当前任务指令（自然语言）。

**动作空间 $\mathcal{A}$：** 同为有限长度词汇序列的集合，具体为 LLM 生成的计划（子任务分解）。

**观测空间 $\Omega$：** 论文采用完全可观测的 MDP 设定，未显式定义观测空间。状态即可观测量。

**记忆数据结构：**

- Case Memory（$m_l$）：向量化存储，每个 case 为三元组 $(s_i, a_i, r_i)$，其中 $s_i$ 用 SimCSE 编码为向量，$a_i$（计划）和 $r_i$（二值奖励）保留原始形式
- Subtask Memory（$m_s$）：文本形式，存储子任务及其执行结果
- Tool Memory（$m_s$）：文本形式，存储工具调用及结果

> **评注：** M-MDP 中的 memory space $\mathcal{M}$ 定义非常通用，但实际实现中被简化为单步设定，这使得理论框架（多步 soft Q-learning）与实际实现之间存在一定 gap。论文在 §4.2 明确指出这一简化并给出了理由（规划层的 CBR 只需单步决策），但多步 M-MDP 的潜力尚未被完全验证。

---

## 2. Training Procedure

**优化的组件：** 检索策略 $v$（即 $\mu$）。LLM 参数 $\theta_{\text{LLM}}$ 完全冻结。具体优化的是 Q 网络的参数 $\theta$（两层 MLP），该 Q 网络用于驱动 case 检索分布。（§3, §4.2）

**优化算法：**

- **理论框架：** Maximum entropy RL / Soft Q-learning（§3, Eq. 3–8）
- **实际实现（参数化记忆）：** 单步 Q-learning，由于单步设定 TD target 退化为即时奖励，进一步简化为监督学习（二分类）。使用 **交叉熵损失**（CE loss）替代 MSE（§4.2, Eq. 15）
- **实际实现（非参数化记忆）：** 无需优化，直接使用 SimCSE 余弦相似度做 TopK 检索

**训练数据来源：** 在线交互产生的轨迹。每个任务完成后，将 $(s_t, a_t, r_t)$ 存入 case bank 和 replay buffer $\mathcal{B}$。（§4.2, Algorithm 1）

**LLM 参数是否冻结：** 是。LLM 参数完全冻结。优化的参数仅为 Q 网络 $\theta$（两层 MLP），输入为 SimCSE 编码的 state 和 case 表示。（§4.2, §5.3）

**核心训练目标函数：**

**理论层面的 soft Q-learning 目标（Eq. 3, §3）：**

$$J(\pi) = E_{\tau \sim p}\left[\sum_{t=0}^{T-1}\left[\mathcal{R}(s_t, a_t) + \alpha \mathcal{H}(\mu(\cdot \mid s_t, M_t))\right]\right]$$

统一符号：最大化轨迹 $T$ 上的累计奖励加上检索策略 $v$ 的熵正则项（温度系数 $\alpha$）。

**实际实现的单步 Q-learning 目标（Eq. 14→15, §4.2）：**

$$\mathcal{L}(\theta) = E_{(s,c,r)}\left[-r \log Q(s, c; \theta) - (1-r) \log(1 - Q(s, c; \theta))\right]$$

统一符号：对检索策略 $v$ 的参数 $\theta$ 进行优化，$Q_\theta(s, c)$ 表示在状态 $q$（或 $C$）下检索到 case $c$ 导致成功的概率。$r \in {0, 1}$ 为二值奖励。

**Kernel-based Q 估计（理论层面，Eq. 9, §3）：**

$$Q^{EC}(s, M, c; \theta) = \frac{\sum_{(s', c', Q') \in \mathcal{D}_c} k_\theta(s, s') Q'}{\sum_{(\hat{s}, \hat{c}, \hat{Q}) \in \mathcal{D}_c} k_\theta(s, \hat{s})}$$

统一符号：通过可学习核函数 $k_\theta$ 对 episodic memory $\mathcal{D}$ 中相同 case 的历史 Q 值进行加权平均。

> **评注：** 理论框架（多步 soft Q-learning + kernel-based EC）和实际实现（单步 CE loss + MLP Q 网络）之间的差异值得关注。实际实现更简洁高效，但牺牲了多步推理的理论优势。

---

## 3. Reward Signal

**奖励类型：** **Sparse terminal reward**（稀疏终端奖励）。仅在任务完成后给出一个二值奖励 $r \in {0, 1}$。（§4.2）

**奖励来源：** **Exact Match (EM)** 或 **LLM-as-judge (Partial Match)**。在 GAIA 上使用 EM 精确匹配（标准化后与 ground-truth 对比），在 DeepResearcher、SimpleQA、HLE 上使用 GPT-4o-mini 作为答案评估器给出 PM 分数。（§5.2）

**奖励分配：** 奖励直接关联到整个 episode 的最终步骤。由于实际实现为单步 MDP（CBR 仅用于规划阶段），不存在跨步骤的 credit assignment 问题——每个 case bank entry 的 $r$ 即为该任务的终端奖励。（§4.2）

**辅助奖励或正则项：**

- **熵正则项**（Eq. 3, §3）：$\alpha \mathcal{H}(\mu(\cdot \mid s_t, M_t))$，鼓励检索策略的多样性，避免 collapse 到单一 case。理论框架中明确存在，但实际实现中使用 TopK 确定性选择（Eq. 16），不再显式使用熵正则。
- 无其他辅助奖励。

> **评注：** 二值奖励在 deep research 任务中是一个较粗糙的信号。论文提到 "reward function and memory update can also be probabilistic in some specific cases, which we leave as future work"（§3），其实作者也意识到更细粒度的奖励设计可能带来进一步提升。

---

## 4. Inference Procedure

**记忆初始化：** Case bank $m_l$ 初始化为空集 $M_0 = \emptyset$（Algorithm 1），或使用先前迭代积累的 case bank（如 GAIA test set 使用 validation 阶段积累的 case bank，§5.4）。Subtask Memory 和 Tool Memory 每个新任务初始化为空。

**每步决策流程（§4.1, Figure 3）：**

1. **Case-Based Planning（Stage 1）：**
    - 输入：用户查询 $q$
    - 检索：Planner（GPT-4.1）从 case bank $m_l$ 中检索 $K$ 个最相关 case
        - 非参数化：$\text{TopK}_{(s_i,a_i,r_i) \in M_t} \text{sim}(\text{enc}(s_t), \text{enc}(s_i))$
        - 参数化：$\text{TopK}_{c_i \in M_t} Q(s_t, c_i; \theta)$
    - 生成：将 retrieved cases 拼接到 prompt 中，LLM 生成子任务分解计划
2. **Tool-Based Execution（Stage 2）：**
    - Executor（o3 或 o4-mini）逐个执行子任务
    - 每个子任务中，Executor 通过 MCP 协议调用外部工具（search, crawl, code, image, video, math, document 等）
    - 结果存入 Subtask Memory 和 Tool Memory
3. **Replan Loop：**
    - Planner 检查 Subtask Memory，若任务未完成则基于更新的上下文重新规划
    - 若完成，返回最终答案 $A_n$，并将 $(s_t, a_t, r_t)$ 写入 case bank $m_l$

**推理时额外策略：**

- **TopK 检索**：默认 $K=4$（基于消融实验，§5.5.1）
- **Pass@3**：在 GAIA validation 上使用多次采样取最优（§5.4）
- **多轮 replan**：Planner 和 Executor 交替迭代直到任务完成
- **工具选择**：Executor 通过 MCP 协议动态选择工具

**推理策略的来源：** 混合驱动。Case 检索由学习得到的 $Q_\theta$（参数化版本）或固定的相似度函数（非参数化版本）驱动；计划生成和工具调用由冻结的 LLM 的 $p_{\text{LLM}}$ 驱动；planner-executor 交替循环和 MCP 工具调用协议为手工设计的规则。（§4.1, §4.2）

> **评注：** Memento 的推理流程本质上是 "learned retrieval + handcrafted orchestration"。核心学习信号仅体现在 case 检索上，而 planner-executor 架构、MCP 工具调用、replan 策略等均为工程设计。这与 end-to-end RL 训练的 agent（如 DeepResearcher）形成鲜明对比：Memento 以更低的训练成本换取了更多的工程设计负担。

---

## 5. RQ 分析

### RQ1: What is memory?

Memento 定义了三层记忆结构：
（1）Case Memory（$m_l$）是核心的长期记忆，以向量化 episodic case bank 的形式存储历史经验三元组 $(s, a, r)$，支持跨任务复用；（2）Subtask Memory 和 Tool Memory（$m_s$）是短期工作记忆，以文本形式存储当前任务的中间状态。记忆的最小单元是 case，即一个完整的任务-计划-结果三元组。

### RQ2: How memory evolves, operates?

记忆通过 Write 和 Read 两个操作演化。Write 在每个任务完成后将新 case 追加到 case bank，参数化版本同时在线更新 Q 函数。Read 在规划时检索 TopK 个最相关 case：非参数化版本基于 SimCSE 余弦相似度，参数化版本基于学习的 Q 值。记忆只增不删，论文未实现遗忘或淘汰机制。

### RQ3: Which component is optimized? Which signal is used?

优化的是 case 检索策略 $v$（$\mu$），具体为 Q 网络参数 $\theta$（两层 MLP）。LLM 参数完全冻结。使用的信号是任务完成后的二值终端奖励 $r \in {0,1}$（EM 或 LLM-as-judge），通过交叉熵损失优化 Q 网络。理论框架中还包含熵正则项以鼓励检索多样性。

### RQ4: Regarding online optimization

Memento 明确支持在线持续学习。Case bank 在推理过程中不断增长（online-growing），新任务的经验被即时写入，后续任务可立即利用。这是 cross-chat 的在线适应，即case bank 跨不同任务/对话持续积累。论文展示了 5 轮迭代的持续学习曲线，参数化 CBR 从约 80% 提升到约 85%。但论文也指出，约 3k 数据后 case bank 快速饱和，边际收益递减。

---

## Conclusion

这篇论文提出了一种不需要微调 LLM 参数的 agent 持续学习框架。其核心思想是将 LLM agent 的规划过程建模为 Memory-Based MDP，并通过一个不断增长的 episodic case bank（长期记忆）来存储历史任务的经验。在面对新任务时，agent 从 case bank 中检索相似的历史经验来辅助规划，类似于人类的类比推理。检索策略可以是简单的相似度匹配（非参数化），也可以通过一个轻量 Q 网络来学习（参数化），后者使用任务成功与否的二值反馈在线更新。整个系统采用 planner-executor 架构：planner 负责任务分解和案例检索，executor 通过 MCP 协议调用各种外部工具执行子任务。其理论上推导了完整的多步 soft Q-learning 框架（含 kernel-based episodic control），但实际实现简化为单步二分类损失，我认为其实本质上是监督学习而非真正的 RL。