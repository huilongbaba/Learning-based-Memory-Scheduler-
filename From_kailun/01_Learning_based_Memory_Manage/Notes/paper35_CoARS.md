# CoARS: Self-Distilled Reinforcement Learning for Co-Evolving Agentic Recommender Systems

**Source:** [arXiv:2604.10029v2](https://arxiv.org/abs/2604.10029) (2026-04-18 修订)

---

## 符号映射表

|论文原始符号|框架符号|含义|
|---|---|---|
|$h_u$ + $C_t$|$q$|RecAgent 的输入：用户历史 + 候选物品集|
|$r_t$, $j_t$|$t$（thought）|RecAgent 推荐理由 / UserAgent 反馈理由|
|$i_t$|$a$（推荐侧 action）|RecAgent 在第 $t$ 轮选出的物品|
|$a_t \in \lbrace \text{click, star, skip, dislike} \rbrace$|$a$（用户侧 action）|UserAgent 的反馈动作|
|$m^{rec}_t$, $m^{user}_t$|$a$ + $t$（消息载体）|一轮交互内 Rec / User 输出的完整消息|
|$m^{user}_t$（对 RecAgent 而言）|$o$|RecAgent 的下一轮观测|
|$\mathcal{M}_t$|$m_s$ + $m_l$|单用户、跨轮但同一 session 的自然语言记忆|
|$\tau_t$|单步 trajectory 元组|一轮交互的完整记录|
|$\mathcal{E}_u$|$T$（trajectory）|用户 $u$ 的完整交互轨迹|
|$x_t$|$C$（context）|当前 agent 的输入上下文|
|$\pi^{rec}$, $\pi^{user}$|$\pi$|两个 agent 的策略|
|LoRA params|$\theta$|被优化的可训练参数（主 LLM 冻结，仅训练 LoRA）|
|$d_t$|（新概念）参考轨迹|由历史 trajectory 改写而来的 reference，用于 teacher mode|
|$\pi_T$ / $\pi_S$|（新概念）teacher / student 策略|同一 agent 两种 conditioning：有/无 reference|
|$A_{t,n}$|（新概念）token-level diagnostic advantage|teacher-student log-prob 差|
|$R^{rec}_t$, $R^{user}_t$|$r$（turn-level reward）|双向交互奖励|
|最终选中物品|$A_N$|多轮交互后给出的最终推荐|

> 论文中的 $\mathcal{M}_t$ 同时承担框架中 $m_s$（当前 episode 上下文）与 $m_l$（跨轮持久化）的角色，因为 ARS 中长期记忆按 user-episode 边界划分。

---

## 概览

这篇文章使用了已经发生过的 ARS 多轮交互轨迹（包括两边 agent 的推荐内容、推荐理由、用户反应、反馈理由），并把这些 trajectory 在事后用 ground-truth 物品和回顾性解释改写成"参考轨迹"。  
最后得到了两个被联合训练（LoRA 微调）的 LLM agent：一个 RecAgent 和一个 UserAgent，二者在同一交互循环里被同时更新。  
优化的是推荐准确度（Hit@1）和用户模拟保真度（F1），手段是设计 (1) 双向 turn-level interaction reward（同时奖励两个 agent） (2) 基于 self-distillation 的 token-level credit assignment（把改写后的历史 trajectory 当 teacher 来给 student 提供细粒度信号）。

---

## 1. Problem Setting

- **Memory 类型**：in-episode、跨轮（每个用户 session 独立）的自然语言记忆 $\mathcal{M}_t$。它既是 $m_s$（被注入下一轮 prompt）也是 $m_l$（在该用户的整个交互过程中持久存在），但**不跨用户、不跨 episode**。
- **决策过程建模**：多轮 sequential decision，没有显式标记为 MDP / POMDP。每一轮 RecAgent 与 UserAgent 各自从对方上一轮输出和共享 memory 出发产生当前轮输出，可以理解为一个**双 agent 交替决策的 turn-based partially observable game**。
- **状态 / 动作 / 观测**：
    - $\mathcal{S}$：当前 $(h_u, C_t, \mathcal{M}_{t-1})$
    - $\mathcal{A}^{rec}$：候选集 $C_t$ 中的物品 + 自然语言 rationale
    - $\mathcal{A}^{user}$：四类动作 $\lbrace \text{click, star, skip, dislike} \rbrace$ + acceptance score $s_t \in [0,1]$ + rationale
    - $\Omega$：对方 agent 上一轮的输出消息
- **记忆数据结构**：自然语言条目列表，按轮次 append（Reflexion-style）：

$$\mathcal{M}_t = \mathcal{M}_{t-1} \cup \lbrace (m^{rec}_t, m^{user}_t) \rbrace$$

- **核心组件映射**：

|组件|论文符号|框架符号|
|---|---|---|
|推荐策略|$\pi^{rec}$|$\pi$|
|用户策略|$\pi^{user}$|$\pi$|
|单轮交互记录|$\tau_t$|trajectory step|
|用户级完整轨迹|$\mathcal{E}_u$|$T$|
|记忆库|$\mathcal{M}_t$|$m_s$ / $m_l$|
|参考轨迹|$d_t$|（论文新增，无对应）|

---

## 2. Training Procedure

- **优化对象**：主 LLM 自身的参数（通过 LoRA adapter）。**RecAgent 和 UserAgent 的 LoRA 同时被训练**，并通过共享交互轨迹和耦合的 reward 设计进行 co-evolution。属于多 LLM 联合优化（**O2b**）。
- **优化算法**：论文没有显式调用 PPO/GRPO/DPO 等命名算法，而是采用**带 token-level reward shaping 的 REINFORCE 风格策略梯度**。loss 形式是 $(R_t + \lambda A_{t,n}) \cdot \log \pi_S$，其中 $R_t$ 是 turn-level interaction reward，$A_{t,n}$ 是来自 self-distillation 的 token-level advantage。
- **训练数据来源**：在线交互（RecAgent 与 UserAgent 实际 rollout 出的轨迹）+ 由 ground-truth 物品和回顾性诊断改写得到的 reference trajectory $d_t$。
- **是否冻结 LLM 参数**：主 LLM **冻结**，只训练 LoRA。但 LoRA 直接修改主 LLM 的 forward 行为，因此本质上仍是对主 LLM 的参数化扩展（属 O2 而非 O5）。
- **核心目标函数**：

RecAgent 目标（Eq. 11，已统一符号）：

$$ \mathcal{J}_{rec} = \mathbb{E}_{(u, \mathcal{E}_u) \sim \mathcal{D}} \Bigg[ \sum_{t=1}^{T_u} \frac{1}{\lvert y^{rec}_t \rvert} \sum_{n=1}^{\lvert y^{rec}_t \rvert} \big( R^{rec}_t + \lambda^{rec}_{SD} \cdot A^{rec}_{t,n} \big) \cdot \log \pi^{rec}_S(y^{rec}_{t,n} \mid x^{rec}_t, y^{rec}_{t, < n}) \Bigg] $$

UserAgent 目标（Eq. 12）：

$$ \mathcal{J}_{user} = \mathbb{E}_{(u, \mathcal{E}_u) \sim \mathcal{D}} \Bigg[ \sum_{t=1}^{T_u} \frac{1}{\lvert y^{user}_t \rvert} \sum_{n=1}^{\lvert y^{user}_t \rvert} \big( R^{user}_t + \lambda^{user}_{SD} \cdot A^{user}_{t,n} \big) \cdot \log \pi^{user}_S(y^{user}_{t,n} \mid x^{user}_t, y^{user}_{t, < n}) \Bigg] $$

token-level diagnostic advantage（Eq. 10）：

$$ A_{t,n} = \mathrm{clip}\Big( \log \pi_T(\hat{y}_{t,n} \mid x, d, \hat{y}_{t, < n}) - \log \pi_S(\hat{y}_{t,n} \mid x, \hat{y}_{t, < n}),\ -1,\ 1 \Big) $$

---

## 3. Reward Signal

- **奖励类型**：dense **turn-level** reward（每轮一个 $R_t$）+ dense **token-level** advantage（每个采样 token 一个 $A_{t,n}$）。**不是** sparse terminal-only reward。
- **奖励来源**：
    - **RecAgent reward** (Eq. 7)： $$R^{rec}_t = (2,\mathrm{hit}_t - 1)(0.5 + 0.5 s_t) D_t$$
        - $\mathrm{hit}_t \in \lbrace 0,1 \rbrace$ 来自 ground-truth 比对（**EM-style**）
        - $s_t$ 来自 UserAgent 的 acceptance strength（**agent 反馈**）
        - $D_t$ 是 stage-sensitivity（**手工设计**）
    - **UserAgent reward** (Eq. 8)： $$R^{user}_t = (2,\mathrm{hit}_t - 1)(2 s_t - 1)\big(1 - \alpha, q_t (2 s_t - 1)\big)$$
        - $q_t \in [-1, 1]$ 是 peer similarity（来自外部传统模型 SASRec 的 user embedding 相似度，**外部环境信号**）
- **奖励分配机制**：
    - $R_t$ 在该轮所有 token 上**均匀分摊**（每 token 加同样的 $R_t$）
    - $A_{t,n}$ 提供**显式的 token-level credit assignment**（每个 token 的 advantage 由 teacher / student log-prob 比较得出）
- **辅助奖励 / 正则**：
    - 论文显式排除"推荐错误但 UserAgent 仍给正反馈"的 case，避免 dataset incompleteness 污染监督信号
    - $\alpha = 0.1$（peer similarity 系数）、$\lambda^{rec}_{SD} = \lambda^{user}_{SD} = 0.1$（self-distillation 权重）作为 shaping 超参

---

## 4. Inference Procedure

- **记忆初始化**：$\mathcal{M}_0 = \emptyset$（每个用户开始时记忆为空）。
- **每步决策流水线**（一轮交互内）：
    1. RecAgent 接收 $(h_u, C_t, \mathcal{M}_{t-1})$，输出 $(i_t, r_t)$
    2. UserAgent 接收 $(h_u, m^{rec}_t, \mathcal{M}_{t-1}, p_t)$，输出 $(a_t, s_t, j_t)$；其中 $p_t$ 是来自其他用户的"peer recommendation opinions"（外部辅助证据）
    3. memory 更新：$\mathcal{M}_t = \mathcal{M}_{t-1} \cup \lbrace (m^{rec}_t, m^{user}_t) \rbrace$
    4. 终止条件：UserAgent 输出 click 或达到最大轮数 $T_u$
- **额外推理策略**：
    - **不使用** top-$k$ 检索（memory 全量注入，无 $v$ 检索器）
    - **不使用** 温度调节、replan 等技巧
    - **不使用** reference trajectory $d_t$（$d_t$ 仅用于训练阶段的 teacher mode，推理时只有 student 行为）
    - 候选集由 SASRec 预先生成 top-20，因此 ARS 实际是在做"agentic reranker"
- **手工规则 vs 学得策略**：策略本身完全由学得的 $\pi^{rec}$ / $\pi^{user}$ 驱动；但**外部框架**（最大轮数、终止条件、候选集生成、acceptance interval 与 action 的映射 click/star/skip/dislike → score interval）仍是手工规则。

> CoARS 的 co-evolution 实际上是 parameter-level（LoRA 更新）的 co-evolution。

---

## 5. RQ 分析

### RQ1（What is memory?）

CoARS 的 memory 是 in-episode 的自然语言交互记录 $\mathcal{M}_t$，按轮次 append、被 serialize 后注入 prompt，属于显式非参数化外部记忆（T1）。同时，由于通过 LoRA 把交互经验内化为参数，也具有 T4 的成分。reference trajectory $d_t$ 是训练期的 privileged context。

### RQ2（Which component is optimized?）

优化的是主 LLM（O2）；具体是 RecAgent 和 UserAgent 的 LoRA adapter 同时被训练（O2b）。两个 agent 通过共享轨迹与耦合 reward 显式地实现 cross-agent credit assignment。检索器、记忆管理器、KV cache 均未被优化。

### RQ3（Target of Optimization）

主目标是 G1（推荐准确度 Hit@1）；同时 UserAgent 的训练目标是用户模拟保真度 F1，本质仍是"判别正确响应"的准确度，归 G1。

### RQ4（Training Signal and RL Algorithm）

CoARS 使用的是 vanilla policy gradient (REINFORCE) + token-level reward shaping。

### RQ5（How memory evolves, operates?）

memory 演化采用最朴素的 append-only 规则，无遗忘、无重写、无压缩、无检索；只在每一轮把上一轮的两侧消息加进去并整体作为 prompt 注入下一轮。memory 的"读"是全量注入而非 selective retrieval。CoARS 的创新点完全集中在用 memory 作训练信号（生成 $d_t$）这一侧，而非 memory 自身的 read/write/forget 机制。

---

## Conclusion

CoARS 把 agentic recommender 中的两个 agent 多轮对话看成一个可以双向训练的耦合系统：每一轮交互天然产生两边都能用的监督信号（推荐对不对、用户接不接受、反应强不强），论文用一个手工设计的 turn-level interaction reward 同时奖励 RecAgent 和 UserAgent；同时把已经发生过的历史 trajectory 用 ground-truth 物品改写成"reference trajectory"，让同一个 agent 在 conditioning 上 reference 时（teacher）和不 conditioning 时（student）的 token 概率差异充当 token-level advantage。最终通过 LoRA + REINFORCE-style 策略梯度联合训练两个 agent，在 LastFM / MovieLens / Instruments 上同时提升了推荐准确度（Hit@1）和用户模拟保真度（F1）。与 Memory-R1 / A-MEM 这类强调 memory 管理的工作相比，CoARS 几乎没有 touch memory 本身，而是把 memory 当训练数据源。
