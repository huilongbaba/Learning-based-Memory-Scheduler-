# 🧠 Memory under the Context of RL

*A Formal Framework and Literature Review Structure*

---

# 1. Notation & Definitions

## Core Interaction

* **q**: question / human input
* **t**: thought (reasoning trace)
* **a**: action
* **o**: observation
* **C**: context

---

## Memory

* **mₛ**: short-term memory (working memory)
* **mₗ**: long-term memory (external memory)

---

## System Components

* **v**: retrieval algorithm
* **R**: RAG system
* **M**: model

---

## Learning & Policy

* **T**: trajectory
* **θ**: model parameters
* **π**: policy of the model

---

## Output

* **Aₙ**: final answer

---

# 2. Problem Formulation

We model a memory-augmented agent as:

> **POMDP (Partially Observable Markov Decision Process) + External Memory System**

$$
\mathcal{P} = (\mathcal{S}, \mathcal{O}, \mathcal{A}, \mathcal{M}, P, R, \gamma)
$$

Where:

* $s_t \in \mathcal{S}$: environment state (not fully observable)
* $o_t \in \mathcal{O}$: observation (corresponds to **o**)
* $a_t \in \mathcal{A}$: action (corresponds to **a**)
* $m_t \in \mathcal{M}$: memory state (includes $m_s, m_l$)
* $R$: reward function
* $\gamma$: discount factor

---

# 3. Memory Architecture

## 3.1 Short-Term Memory (Working Memory)

$$
m_s^t = f(m_s^{t-1}, o_t, a_{t-1})
$$

Characteristics:

* Short-lived
* Task-dependent
* Analogous to context window / KV cache

---

## 3.2 Long-Term Memory (External Memory)

$$
m_l = \{e_1, e_2, \ldots, e_n\}
$$

Each memory entry:

$$
e_i = (c_i, \text{meta}_i)
$$

* $c_i$: content (text / embedding / structured info)
* $\text{meta}_i$: timestamp / importance / source

---

## 3.3 Memory Retrieval

$$
\tilde{m}_l^t = v(q_t, m_l)
$$

* $v$: retrieval algorithm
* Corresponds to: **RAG system (R)**, i.e. dense search/graph retrieval/hybrid retrieval.

---

## 3.4 Context Construction

$$
C_t = g(o_t, m_s^t, \tilde{m}_l^t)
$$

Corresponds to:

* **C: context**: prompt-level abstraction.

---

# 4. Policy & Reasoning

$$
(t_t, a_t) \sim \pi_\theta(\cdot \mid C_t)
$$

* $t_t$: thought (chain-of-thought / reasoning trace)
* $\pi_\theta$: model policy
* $\theta$: model parameters

---

# 5. Trajectory

$$
T = \{(q_t, t_t, a_t, o_t)\}_{t=1}^T
$$

---

# 6. Memory Update

## 6.1 Write Mechanism

$$
m_l \leftarrow m_l \cup \{\phi(o_t, t_t, a_t)\}
$$

---

## 6.2 Selective Memory Writing

$$
\text{write if } \text{importance}(o_t, t_t) > \tau
$$

---

# 7. Training Procedure

## 7.1 Policy Optimization

$$
\max_\theta \mathbb{E}[R(T)]
$$

---

## 7.2 Memory System Optimization

### (1) Retrieval Optimization

$$
\max_v \mathbb{E}[R]
$$

Optimizes:

* Retrieval accuracy
* Ranking quality
* Latency

---

### (2) Memory Compression / Selection

$$
\max \; \text{utility}(m_l) - \lambda |m_l|
$$

---

### (3) Memory Writing Policy

$$
\pi_{\text{write}}(e_t \mid T)
$$

---

# 8. Reward Signal

Reward signals may come from:

* Task success (accuracy, completion rate)
* Human feedback (RLHF)
* Memory efficiency (retrieval hit rate)
* Reasoning quality

---

# 9. Inference Procedure

**For each step** $t$:

1. $o_t \leftarrow \text{observe}$
2. $q_t \leftarrow \text{construct query}$
3. $\tilde{m}_l^t \leftarrow v(q_t, m_l)$
4. $C_t \leftarrow g(o_t, m_s^t, \tilde{m}_l^t)$
5. $(t_t, a_t) \sim \pi_\theta(\cdot \mid C_t)$
6. $o_{t+1} \leftarrow \text{environment feedback}$
7. Update $m_s$
8. Optionally update $m_l$

---

# 10. Expected Outputs (Research Questions)

## RQ1: What is memory?

→ Memory is a system consisting of:

* storage
* retrieval
* update
* policy interaction

---

## RQ2: How does memory evolve and operate?

→ A dynamic loop:

```
write → store → retrieve → update
```

---

## RQ3:

### Which component is optimized?

* Model parameters $\theta$
* Retrieval function $v$
* Memory writing policy
* Context construction

---

### Which signal is used?

* Reward signal (RL)
* Supervised signal (SFT)
* Implicit signals (usage frequency, attention)

---

## RQ4: Online Optimization

Key questions:

* Is memory updated online?
* Is retrieval updated online?
* Does policy adapt online?
