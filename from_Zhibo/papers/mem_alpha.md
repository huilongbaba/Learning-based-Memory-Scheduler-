# $Mem_{\alpha}$
https://openreview.net/forum?id=dm42omwep1
## Problem Setting
给定输入序列：

\[
C = \{c_1, c_2, \dots, c_n\}
\]

目标是学习一个 **plug-in model**，逐 chunk 构建 memory：

\[
a_t \sim \pi_\theta(a_t \mid C_{\le t}, m_s^t, m_l^t)
\]

\[
M_t = a_t(M_{t-1})
\]

最终回答通过：

\[
a_n \sim \pi_{\text{llm}}(a_n \mid q, r(q, M_T))
\]

优化目标：**最大化 QA accuracy**

---

## Training

将 **memory construction** 建模为一个 MDP：
### State

\[
s_t = (c_t, m_l^{t-1})
\]

---

### Action

\[
a_t = \{a_t^{(1)}, \dots, a_t^{(K_t)}\}, \quad a_t^{(k)} \in \mathcal{A}_{write}
\]

其中：

\[
\mathcal{A}_{write} = \{\text{insert}, \text{update}, \text{delete}\}
\]

---

### Memory Update

\[
m_l^t = g_l(m_l^{t-1}, a_t)
\]

论文表示为：

\[
M_t = T(M_{t-1}, a_t)
\]

---

### Objective (PPO)

\[
J(\theta) = \mathbb{E} \left[
\sum_{t=1}^{n} \sum_{j=1}^{|a_t|}
\min \left(
\frac{\pi_\theta}{\pi_{\text{old}}} A_t,
\text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}\right) A_t
\right)
\right]
\]

---

##  Reward Design

总 reward：

\[
r_t = r_1 + r_{2,t} + \beta r_3 + \gamma r_{4,t}
\]

---

### (1) QA Correctness（核心）

\[
r_1 = \frac{1}{m} \sum_{j=1}^{m} \mathbf{1}[\hat{a}_j = a_j]
\]

其中：

\[
\hat{a}_j = R(q_j, v(m_l^n, q_j))
\]

👉 对应 **RAG pipeline**

---

### (2) Tool Valid（结构正确）

\[
r_{2,t} = \frac{1}{K_t} \sum_{k=1}^{K_t} \mathbf{1}[a_t^{(k)} \text{ valid}]
\]

---

### (3) Compression（压缩能力）

\[
r_3 = 1 - \frac{|m_l^n|}{\sum_{i=1}^{n} |c_i|}
\]

---

### (4) Memory Quality（语义质量）

\[
r_{4,t} = \frac{1}{K_t} \sum_{k=1}^{K_t} \mathbf{1}[a_t^{(k)} \text{ semantically valid}]
\]

---

## Inference

### Memory Construction

对每个 chunk：

\[
m_l^t = g_l(m_l^{t-1}, a_t), \quad a_t \sim \pi_\theta
\]

---

### Retrieval + Answer

#### Retrieval

\[
z = v(m_l^n, q)
\]

使用 **BM25 top-k**

---

#### Generation

\[
a_n \sim \pi_{\text{llm}}(a_n \mid q, r(q, M_T))
\]

---


## Conclusion

This paper gets 4464 points on ICLR 2026 (although rejected) :) give us confidence


