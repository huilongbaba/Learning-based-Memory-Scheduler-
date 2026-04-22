# MEMAGENT FORMALIZATION
https://openreview.net/forum?id=k5nIOvYGCL
## 1 Problem Setting
given the trunks, which is fed to the agent one by one

$$
C_k = \{c_1, c_2, \dots, c_K\}
$$

the whole trajectory $\tau$ on token level is described as

$$
p(x_{1:N}) = \prod_{n=1}^{N} p(x_n \mid x_{1:n-1})
$$

with the short memory

$$
m_s^k \in \mathcal{V}^M
$$

then the trajectory with the memory could be defined as

$$
p(x_{1:N}) = \sum_{m_s^{1:K}} \prod_{k=1}^{K}
p(c_k \mid m_s^{k-1}) \cdot p(m_s^k \mid c_k, m_s^{k-1})
$$

in the end, the answer is received throguh

$$
A_n = \pi_\theta(q, m_s^K)
$$


---

## Training

use Dr. GRPO 

每个context window包括

$$
s_k = (m_s^{k-1}, c_k)
$$

写出value function

$$
J(\theta) =
\mathbb{E}
\left[
\sum_{i,j,t}
\min
\left(
r_{i,j,t} \hat{A}_i,
\text{clip}(r_{i,j,t}) \hat{A}_i
\right)
- \beta D_{KL}(\pi_\theta || \pi_{ref})
\right]
$$

---

## Reward

reward 就只用一种， 即 回答正确与否

$$
R(\hat{y}, y) =
\mathbf{1}[\hat{y} = y]
$$

---

## Inference

$$
m_s^0 = \varnothing
$$

$$
m_s^k = \pi_\theta(m_s^{k-1}, c_k)
$$

$$
k = 1 \dots K
$$

$$
A_n = \pi_\theta(q, m_s^K)
$$

---

## conclusion

ICLR 2026
