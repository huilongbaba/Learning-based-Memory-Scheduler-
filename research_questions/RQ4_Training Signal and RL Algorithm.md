# RQ4: Training Signal and RL Algorithm

> Research Question: 对于真正使用 RL 训练记忆相关策略的 memory-augmented LLM agent，采用什么 RL 算法将训练信号转化为策略梯度？

不使用 RL 的工作（纯 SFT、蒸馏、prompt-engineering、training-free heuristic、监督对比学习等）统一归入 **N/A: Non-RL Methods**，不再细分。

---

## 分类总览

以下是当前观察到的 RL 算法：

| 代号      | 算法                                              | 核心特征                                                                                               | 代表论文                                        |
| ------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **A1**  | PPO                                             | actor-critic + clipped surrogate，需要 value network                                                  | MIRA, Memory-Based Advantage Shaping        |
| **A2**  | GRPO                                            | group-relative advantage，无需 value network；包含 DAPO、MemexRL、Task-Stratified GRPO、Multi-Conv DAPO 等变体 | EMPO, MemPO, MemSifter, CoMAM, Memex        |
| **A3**  | Q-Learning / Soft Q-Learning                    | 价值型，TD / Bellman 回归                                                                                | Memento                                     |
| **A4**  | Training-Free / Closed-Form Policy Optimization | 无梯度更新，KL-约束下的闭式最优策略                                                                                | JitRL                                       |
| A5      | DPO                                             | 基于成对偏好数据 $(O^+, O^-)$ 的策略优化，避免显式 reward function                                                   | MemoBrain                                   |
| A6      | REINFORCE / REINFORCE++                         | 不学习 value network，且 advantage 估计不依赖 same-prompt group                                              | CoARS, Frehnewss-Aware PER                  |
| **N/A** | Non-RL Methods                                  | 纯 SFT、蒸馏、监督、prompt-engineering、training-free                                                       | A-Mem, A-MAC, HyperRAG, MSA, Mnemis, RF-Mem |

---

## 类别定义

### A1: PPO

**代表论文**：MIRA, Memory-Based Advantage Shaping

**定义**：经典 actor-critic + clipped surrogate objective，

$$
\mathcal{L}^{\text{CLIP}}(\theta)=\mathbb{E}_t\!\left[\min\!\big(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\big)\right],
$$

其中 $r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{\theta_\text{old}}(a_t|s_t)$。需要显式 value network $V_\phi$ 估计 advantage。

---

### A2: GRPO

**代表论文**：EMPO, MemPO, MemSifter (DAPO), CoMAM, Memex (MemexRL)

**定义**：用同一 prompt 下 group 内样本的相对排名替代 value network 估计 advantage：

$$
\hat{A}_i=\frac{r_i-\text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}.
$$

家族内各变体仅在采样、clipping、归一化维度上有所调整：

- **DAPO**（MemSifter）：decoupled clip + dynamic sampling，对长序列更稳定；
- **Multi-Conv DAPO**（GRU-Mem）：在多轮对话上扩展 group；
- **Task-Stratified GRPO**（UMA）：按任务类型分层归一化 advantage；
- **EMPO**：混合 on-policy / off-policy 采样。

> 选择数量上有压倒性优势。

---

### A3: Q-Learning / Soft Q-Learning

**代表论文**：Memento

**定义**：价值型方法，学习 $Q_\theta(s,a)$ 满足 Bellman 方程；Soft Q-learning 加入熵正则：

$$
Q^*(s,a)=r(s,a)+\gamma\,\mathbb{E}_{s'}\!\left[\tau\log\sum_{a'}\exp(Q^*(s',a')/\tau)\right].
$$

---

### A4: Training-Free

**代表论文**：JitRL

**定义**：不做梯度更新，直接利用 KL-约束 RL 目标的闭式解。对目标

$$
\max_\pi\ \mathbb{E}_{a\sim\pi}[r(s,a)] - \beta\,\text{KL}(\pi\|\pi_\text{ref}),
$$

最优策略为

$$
\pi^*(a\mid s)\propto \pi_\text{ref}(a\mid s)\exp\!\big(r(s,a)/\beta\big).
$$

---

### A5: DPO

**代表论文**：MemoBrain

**定义**：基于成对偏好数据 $(O^+, O^-)$ 的策略优化，通过 Bradley-Terry preference model 推导出闭式 loss，避免显式 reward function。包含 DPO、IPO、KTO 等家族变体。

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \log \sigma \left( \beta \log \frac{\pi_\theta(O^+ \mid s)}{\pi_{\text{ref}}(O^+ \mid s)} - \beta \log \frac{\pi_\theta(O^- \mid s)}{\pi_{\text{ref}}(O^- \mid s)} \right)$$

---

### A6: REINFORCE / REINFORCE++

**代表论文**：Freshness-Aware PER

定义：critic-free 的 policy gradient 算法家族。核心特征是不学习 value network，且 advantage 估计不依赖 same-prompt group——可以直接使用原始 reward（vanilla REINFORCE），也可以在 batch / 全局层面做归一化（REINFORCE++）。

**典型形式**：

vanilla REINFORCE + 可选 reward shaping：

$$\mathcal{L} = -\mathbb{E}_t \big[ (R_t + \lambda A_t^{\text{aux}}) \cdot \log \pi_\theta(a_t \mid s_t) \big]$$

- **$R_t$**: 累积回报 (Return)。
    
- **$A_t^{\text{aux}}$**: 辅助奖励/优势项，用于 **Reward Shaping**。
    
- **$\lambda$**: 辅助项的权重超参数。

REINFORCE++（global normalization + ratio clip）：

$$\hat{A}_t = \frac{R_t - \mu_{\text{batch}}}{\sigma_{\text{batch}} + \epsilon},  \mathcal{L} = -\mathbb{E}_t \big[ \min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \big]$$

- **$\rho_t$**: 策略比例因子，为重要性比率，即 $\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$。
    
- **$\text{clip}$**: 裁剪操作，防止策略更新幅度过大。
    
- **$\mu_{\text{batch}}, \sigma_{\text{batch}}$**: 当前 Batch 的均值与标准差。

---
## N/A: Non-RL Methods

对于不使用 RL 的论文（如纯 SFT、蒸馏、监督、prompt-engineering、training-free等），统一归入 N/A。

---

## 判定标准

为减少类别归属的模糊性，采用以下优先级判定：

1. 无梯度更新 → **A4**
2. 学习 $Q$ 函数 → **A3**
3. 使用 preference pair $(O^+,O^−)$ → **A5**
4. 使用 group-relative advantage（same-prompt group 内做 mean/std 归一化）→ **A2**
5. actor-critic + ratio clip + value network → **A1**
6. REINFORCE 或 REINFORCE++ 风格的 batch-global normalization → **A6**
7. 完全不使用 RL → **N/A**

---
## 论文分类表

| Paper                          | A1 PPO | A2 GRPO | A3 Q-learning | A4 Training-Free | A5 DPO | A6 REINFORCE | N/A |
| ------------------------------ | ------ | ------- | ------------- | ---------------- | ------ | ------------ | --- |
| Memory-R1                      | +      | +       |               |                  |        |              |     |
| MemSearcher                    |        | +       |               |                  |        |              |     |
| General Agentic Memory         |        | +       |               |                  |        |              |     |
| Memento                        |        |         | +             |                  |        |              |     |
| JitRL                          |        |         |               | +                |        |              |     |
| Fine-tuning with RAG           |        |         |               |                  |        |              | +   |
| Knowledge Modules (DCD)        |        |         |               |                  |        |              | +   |
| A-MEM                          |        |         |               |                  |        |              | +   |
| MSA                            |        |         |               |                  |        |              | +   |
| RLIF                           |        | +       |               |                  |        |              |     |
| RF-Mem                         |        |         |               |                  |        |              | +   |
| GRU-Mem                        |        | +       |               |                  |        |              |     |
| UMA                            |        | +       |               |                  |        |              |     |
| HyperRAG                       |        |         |               |                  |        |              | +   |
| Mnemis                         |        |         |               |                  |        |              | +   |
| MIRA                           | +      |         |               |                  |        |              |     |
| Memory-Based Advantage Shaping | +      |         |               |                  |        |              |     |
| EMPO                           |        | +       |               |                  |        |              |     |
| MemPO                          |        | +       |               |                  |        |              |     |
| MemSifter                      |        | +       |               |                  |        |              |     |
| A-MAC                          |        |         |               |                  |        |              | +   |
| CoMAM                          |        | +       |               |                  |        |              |     |
| Memex (RL)                     |        | +       |               |                  |        |              |     |
| Titans                         |        |         |               |                  |        |              | +   |
| Mem-$\alpha$                   |        | +       |               |                  |        |              |     |
| Memory as Action               |        | +       |               |                  |        |              |     |
| Scaling Context Folding        |        | +       |               |                  |        |              |     |
| MAGMA                          |        |         |               |                  |        |              | +   |
| MACLA                          |        |         |               |                  |        |              | +   |
| LightSearcher                  |        | +       |               |                  |        |              |     |
| MemVerse                       |        |         |               |                  |        |              | +   |
| Inside Out                     |        | +       |               |                  |        |              |     |
| MemBuilder                     |        | +       |               |                  |        |              |     |
| MemoBrain                      |        |         |               |                  | +      |              | +   |
| Fine-Mem                       |        | +       |               |                  |        |              |     |
| LIGHTMEM                       |        |         |               |                  |        |              | +   |
| AtomMem                        |        | +       |               |                  |        |              |     |
| HiMeS                          |        | +       |               |                  |        |              |     |
| Hela-Mem                       |        |         |               |                  |        |              | +   |
| Freshness-Aware PER            |        |         |               |                  |        | +            |     |
| MemReader                      |        | +       |               |                  |        |              |     |
| M3-Agent                       |        | +       |               |                  |        |              |     |
| CoARS                          |        |         |               |                  |        | +            |     |
| HiExp                          |        | +       |               |                  |        |              |     |
| MEMORY-T1                      |        | +       |               |                  |        |              |     |
| MEM1                           | +      |         |               |                  |        |              |     |
| LMLM                           |        |         |               |                  |        |              | +   |
| PTE                            |        | +       |               |                  |        |              |     |

> `+` = 适用  
> `—` = 此维度不适用（无记忆系统 / 无训练过程）  
> `!` = 无法归入现有类别，需扩展分类

