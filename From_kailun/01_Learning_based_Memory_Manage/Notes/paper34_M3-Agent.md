# M3-Agent: Seeing, Listening, Remembering, and Reasoning — A Multimodal Agent with Long-Term Memory

**Source:** [arXiv / ICLR 2026 paper](https://openreview.net/pdf?id=PMz29A7Muq) (2026-4-11 修订，Accepted at ICLR 2026 poster，842)

## 符号映射表

| 论文原始符号 / 概念                                                     | 框架符号                   | 说明                                  |
| --------------------------------------------------------------- | ---------------------- | ----------------------------------- |
| Input question                                                  | $q$                    | 用户提问                                |
| Memorization input (video stream + audio)                       | $o$                    | 观测（每 30s 一个 clip）                   |
| Long-term memory $\mathcal{M}$（entity-centric multimodal graph） | $m_l$                  | 节点为 text / image / audio，边为关系       |
| Trajectory $\tau$（control 中的多轮交互）                               | $T$                    | 多轮 reasoning + action + observation |
| Control policy $\pi_\theta$                                     | $\pi$                  | 决定 [Search] 或 [Answer] 的策略          |
| Memorization policy（memory-7b-sft）                              | $g_l$                  | 长期记忆写入函数（由 LLM 实现）                  |
| `search_node`, `search_clip` 工具                                 | $v$                    | 检索算法（embedding-based MIPS）          |
| Reasoning trace（`<think>...</think>`）                           | $t$                    | 思考中间态                               |
| `[Search]` / `[Answer]` action with argument                    | $a$                    | 动作                                  |
| Retrieved memory（注入下一轮 context）                                 | 进入 $C_N$               | 检索结果作为上下文一部分                        |
| Final submitted answer $y_i$                                    | $\hat{A}_N$            | 最终答案                                |
| Memorization model parameters                                   | $\theta_{\text{mem}}$  | 由 SFT 训练                            |
| Control model parameters                                        | $\theta_{\text{ctrl}}$ | 由 DAPO（RL）训练                        |

---

## 概览

 这篇文章用了一段不停流入的视频 + 音频流（模拟机器人/智能体的感官输入），还有针对每段视频的 QA 对（用于训练 control 阶段）。它没有专门的trajectory 数据集，而是让一个 multimodal LLM 自己在 30 秒一段的视频片段上边看边写，把所看到的事件和推断出的世界知识写进一个外部记忆图。  
最后得到了两个被分别训练好的 LLM 外加一个外部多模态记忆图：(1) 一个会写记忆的多模态模型 `memory-7b-sft`（SFT 训出来的，负责把视频流编码成 episodic + semantic memory）；(2) 一个会做多轮检索-推理-回答的 control 模型 `control-32b-rl`（DAPO RL 训出来的）；(3) 一个 entity-centric 的图结构 long-term memory，节点是文本/图像/音频。  
最终目标就是 QA 准确率，给定一段长视频和一个问题，最终答案是否和 ground-truth 一致（由 GPT-4o 当 judge 给 0/1 reward）。

---

## 1. Problem Setting

- **记忆类型**：cross-chat（更准确说是 cross-clip / cross-session）的 $m_l$。M3-Agent 处理的是无限流式视频，需要跨越 30s 一个的 clip 维护一致的实体身份和世界知识，所以是典型的长期记忆。论文未显式建模 $m_s$，因为短期上下文就是当前 control trajectory $\tau$ 本身。
- **决策过程建模**：control 阶段被建模为**多轮 token-level POMDP**：每一轮 $\pi_\theta$ 观察 trajectory，输出 reasoning + action（`[Search]` 或 `[Answer]`）+ argument；若是 `[Search]` 则环境（即 $m_l$）返回检索结果作为下一轮观察；最多 $H=5$ 轮。memorization 阶段则被建模为 **per-clip 监督学习问题**（不是 RL）。
- **状态 / 动作 / 观测空间**：
    - $\mathcal{S}$：当前 trajectory $\tau$（system prompt + 历史 reasoning + 历史检索结果）
    - $\mathcal{A}$：`[Search](query)` 或 `[Answer](text)`
    - $\Omega$：检索函数返回的 top-k 节点 / clip 内容（自然语言 + 多模态引用）
- **记忆的数据结构**：entity-centric 多模态**图**。节点属性见下表，边表示节点间的逻辑关系（如同一实体跨模态绑定）。
- **核心组件与符号映射表**：

|组件|论文中的名字 / 实现|框架符号|
|---|---|---|
|长期记忆|entity-centric multimodal graph (text/image/audio nodes)|$m_l$|
|检索算法|`search_node` / `search_clip`，基于 MIPS + 阈值|$v$|
|写入函数|`memory-7b-sft` 生成 episodic + semantic 列表，按规则插入图|$g_l$|
|控制策略|`control-32b-rl` (Qwen3-32B + DAPO)|$\pi_\theta$|
|工具|facial recognition (InsightFace), speaker ID (ERes2NetV2), embedding (text-embedding-3-large)|$v$ 的子模块|

---

## 2. Training Procedure

- **优化的组件**：两个独立的 LLM，分别承担 memorization 和 control。
    - `memory-7b-sft`：从 Qwen2.5-Omni-7b 出发，SFT 优化 $\theta_{\text{mem}}$
    - `control-32b-rl`：从 Qwen3-32B 出发，DAPO 优化 $\theta_{\text{ctrl}}$
- **算法**：memorization 用纯 SFT（imitation learning，非 RL）；control 用 DAPO（GRPO 家族变体）。memorization 训练后**冻结**，作为 control RL 的环境组件之一。
- **训练数据来源**：
    - memorization：从内部视频库切出 26,943 个 30 秒 clip，由 GPT-4o + Gemini-1.5-Pro 协同生成 episodic memory；用 progressive meta-clip 算法生成跨模态 identity 等价标注；最终合成 10,752 训练样本。
    - control：在内部 500 个长视频（含 2,736 QA pair）上进行在线 rollout，记忆环境由 `memory-7b-sft` 离线预先生成。
- **冻结情况**：训练 control 时，memorization 模型完全冻结；它生成的 $m_l$ 作为静态环境。loss 仅在 control LLM 的生成 token 上计算（搜索结果 token 不计 loss）。
- **核心训练目标函数**：

DAPO 目标（论文式 (2)，按框架符号统一）：

$$ \mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}_{(q,a)\sim D,\ \lbrace\tau_i\rbrace_{i=1}^{G}\sim \pi_{\theta_{\text{old}}}(\cdot\mid q)}\Bigg[\frac{1}{Z}\sum_{i=1}^G \sum_{t=1}^{\lvert\tau_i\rvert}\mathbb{I}(\tau_{i,t})\cdot $$

$$ \min\Big( r_{i,t}(\theta),\hat{A}_{i,t},\ \text{clip}\big(r_{i,t}(\theta),\ 1-\epsilon_{\text{low}},\ 1+\epsilon_{\text{high}}\big),\hat{A}_{i,t} \Big)\Bigg] $$

其中 $r_{i,t}(\theta)=\pi_\theta(\tau_{i,t}\mid \tau_{i,< t})/\pi_{\theta_{\text{old}}}(\tau_{i,t}\mid \tau_{i,< t})$，$Z=\sum_{i=1}^{G}\sum_{t=1}^{\lvert\tau_i\rvert}\mathbb{I}(\tau_{i,t})$，约束 $0<\sum_i R_i<G$（DAPO 的 dynamic sampling，剔除全对 / 全错的 group），advantage 用 group-relative 估计：

$$\hat{A}_{i,t} = \frac{R_i - \mathrm{mean}(\lbrace R_j\rbrace_{j=1}^G)}{\mathrm{std}(\lbrace R_j\rbrace_{j=1}^G)}$$

---

## 3. Reward Signal

- **奖励类型**：**sparse terminal reward**。整条 trajectory $\tau_i$ 只在终止时由 GPT-4o judge 打一个 0/1 分。
- **奖励来源**：LLM-as-judge（GPT-4o），输入是 (question, ground-truth answer, agent answer)，输出 Yes / No。论文用 100 个样本验证 judge 与三人多数投票的一致率为 96%。
- **奖励分配**：均匀广播到 trajectory 的所有生成 token。即同一条 $\tau_i$ 内，所有 LLM 生成的 token 共享同一个 $\hat{A}_{i,t}$（仅 group 内归一化），不做更细粒度的 step-level credit assignment。检索动作、reasoning token、最终 answer token 共享同一份信号。
- **辅助奖励 / 正则**：DAPO 没有显式 KL，只有 dynamic sampling（去掉全对全错 group）和 decoupled clip 起到稳定化作用。论文未引入 format reward / length penalty / process reward。

奖励函数（论文式 (1)，统一符号）：

$$R_i = \mathbb{1} \big[ \text{GPT4o\_eval}(q, \ a^*, \ \hat{A}^{(i)}_N) = \text{True} \big]$$

> 训练信号只能从最终答案对错中反推回来。

---

## 4. Inference Procedure

- **记忆初始化**：推理前，先用 `memory-7b-sft` 离线把整段视频按 30s clip 流式处理一遍，把所有 episodic + semantic 条目写进 entity-centric graph $m_l$。问题被严格限制为只能在该视频对应的 $m_l$ 内检索。
    
- **每步决策流程**（Algorithm 1）：
    
    1. 初始 trajectory $\tau$ = system prompt + question $q$ + instruction prompt
    2. 进入循环（最多 $H=5$ 轮）：
        - $\tau_i \leftarrow \pi_\theta(\cdot\mid\tau)$，模型生成 reasoning + action + argument
        - 若 action = `[Search]`：调 $v(\text{argument}, m_l)$ 取回 top-k 节点 / clip，把结果 + instruction prompt 拼到 $\tau$
        - 若 action = `[Answer]`：返回内容并终止
    3. 最后一轮强制 `[Answer]`（last-round prompt 提示"必须回答，可以基于已有信息合理猜测"）
- **额外推理策略**：
    
    - 检索固定 top-$k=2$，相似度阈值 $t=0.5$（严格设置，降低噪声）
    - 节点 voting：多次激活的（face, voice）对应关系权重累积，conflict 时取权重高者
    - search 输入支持多模态（文本 / 图像 / 音频），可只检索特定模态
    - 论文显式建议在没有 character ID 时先搜 ID 再用 ID 检索（写在 instruction prompt 里）
- **学习 vs 手工规则边界**：决策（搜什么 / 何时停 / 如何 reasoning）完全由 $\pi_\theta$ 学到；但**工具实现本身**是手工的（人脸聚类阈值、voting 权重规则、top-k 阈值都是固定超参）。也就是说，agent 学的是"什么时候调哪个工具、查什么 query"，而不是工具内部的检索算法。
    

---

## 5. RQ 分析

### RQ1 (What is memory?)

非参数化外部记忆 + 显式（T1）。具体形式是 entity-centric multimodal graph，节点同时承载 text / image / audio 三种模态，边表示跨模态身份等价或同实体关联。该图独立于 LLM，可单独 inspect / 替换 / 清空。

### RQ2 (Which component is optimized? Which signal is used?)

主 LLM 优化（O2）：训了两个主 LLM。memorization 走 SFT，control 走 DAPO RL。两者串行训练（先 SFT 出 memorization 并冻结，再训 control），不是 joint 多 agent 优化。所以更接近 O2a 的"单 LLM 优化"在两个独立组件上各做一次，而不是 O2b 的 joint multi-agent。训练信号见下。

### RQ3 (Target of Optimization)

只有 G1 回答准确度。reward 只来自 GPT-4o 对 final answer 的 0/1 判定，没有 token efficiency / loop efficiency / 写入决策质量 / format reward 任何辅助项。memorization 走 SFT 的话不在 RL reward 范畴（监督的是 demonstration 输出本身）。

### RQ4 (Training Signal and RL Algorithm)

A2 (GRPO 家族)：control 训练用 DAPO，含 group-relative advantage、decoupled clipping ($\epsilon_{\text{low}}=0.2$, $\epsilon_{\text{high}}=0.28$)、dynamic sampling（剔除 $\sum R_i \in \lbrace 0, G\rbrace$ 的 group）。论文还在 ablation 跑了 GRPO，结论是 DAPO 更优。memorization 走 SFT，对应 N/A: Non-RL Methods（imitation learning）。

### RQ5 (How memory evolves, operates?)

- 写入：memorization 模型按 30s clip 流式处理，输出 episodic + semantic 文本条目，连同 face / voice 特征插入图。已存在节点会"激活并加权"，不存在则新增。
- 读取：control 阶段 $\pi_\theta$ 自主决策何时调 `search_node` / `search_clip`，最多 $H=5$ 轮。
- 冲突解决：weighted voting，激活频率高的（face, voice）映射胜出。
- 跨模态绑定：同 character 的 face / voice / 文本知识共享一个 `<character_id>`，使得查任一模态都能召回其余模态的关联知识。

---

## Conclusion

M3-Agent 是一个面向长视频 / 流式多模态输入的智能体框架。记忆的写入和组织规则（voting 权重、连边规则）是手工的、不可学。它把记忆做成一张 entity-centric 多模态图：人脸、声音、文本知识围绕同一个角色连结，跨片段保持身份一致性；记忆同时包含具体事件（episodic）和高层世界知识（semantic）。系统训了两个 LLM：一个负责把视频流写成结构化记忆条目（SFT），一个负责在多轮 reasoning 里自主调用搜索工具，最后给出答案（DAPO RL）。RL 的奖励非常简单：最终回答对了得 1 分，错了得 0 分。在自建的 M3-Bench（含 100 段机器人视角视频 + 920 段网络视频）和 VideoMME-long 上，M3-Agent 比 prompting 商业大模型组合的 baseline 高 5–8 个点。    
与 RAG 类工作相比，M3-Agent 把实体一致性做成图结构里的一等公民，靠 voting 处理 cross-clip 冲突。与 online video 模型相比，它把压缩 visual token 换成用 LLM 写自然语言记忆，因此能存语义知识。