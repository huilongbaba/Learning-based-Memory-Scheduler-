# 🧠 Memory under the Context of RL  
*A Formal Framework and Literature Review Structure*

---

# 1. Notation & Definitions

## Core Interaction

- **q**: question / human input  
- **t**: thought (reasoning trace)  
- **a**: action  
- **o**: observation  
- **C**: context  

---

## Memory

- **mₛ**: short-term memory (working memory)  
- **mₗ**: long-term memory (external memory)  

---

## System Components

- **v**: retrieval algorithm  
- **R**: RAG system  
- **M**: model  

---

## Learning & Policy

- **T**: trajectory  
- **θ**: model parameters  
- **π**: policy of the model  

---

## Output

- **Aₙ**: final answer  

---

# 2. Problem Formulation

We model a memory-augmented agent as:

POMDP + External Memory System

---

# 3. Memory Architecture

## Short-Term Memory

m_s^t = f(m_s^{t-1}, o_t, a_{t-1})

## Long-Term Memory

m_l = {e_1, e_2, ..., e_n}

## Retrieval

m_l^t = v(q_t, m_l)

## Context

C_t = g(o_t, m_s^t, m_l^t)

---

# 4. Policy

(t_t, a_t) ~ π_θ(C_t)

---

# 5. Trajectory

T = {(q_t, t_t, a_t, o_t)}

---

# 6. Memory Update

m_l ← m_l ∪ φ(o_t, t_t, a_t)

---

# 7. Training

Maximize expected reward over trajectory

---

# 8. Reward

- Task success  
- Human feedback  
- Memory efficiency  
- Reasoning quality  

---

# 9. Inference Loop

observe → retrieve → build context → infer → update memory

---

# 10. Research Questions

- What is memory?  
- How does it evolve?  
- What is optimized?  
- What signals are used?  
- How does online optimization work?  

---

# Summary

Memory in RL agents is a learnable interface between past experience and current decision-making.
