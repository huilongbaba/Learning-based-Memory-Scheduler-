# Problem Setting
given the definition of state

$$s_t = (q,\; m_s^t,\; \{a_{t-1}, o_{t-1}\})
$$

get the trajectory

$$
T = \{(s_0,a_0,o_0),\dots,(s_T,a_T)\}
$$

which is sampled from llm based on

$$
(t_t,\; m_s^{t+1},\; a_t) \sim \pi_\theta(\cdot \mid s_t)
$$

the process of getting observation is modeled as follow

$$
o_t \sim E(\cdot \mid a_t)
$$

the target of this paper is to get $\pi_{\theta}$ such that the quality and also the efficiency. 

---

# Training

the reward is given in 

$$
R_T \in \{0,1\}
$$

it is divided into different steps

$$
r_t = \gamma^{T - t} \cdot R_T
$$

then, the process for calculating GSPO

$$
\hat{A}_{i,t} = \frac{r_{i,t} - \mu_r}{\sigma_r}
$$

$$
\rho_{i,t}(\theta) =
\frac{\pi_\theta(d_{i,t} \mid s_{i,t})}
{\pi_{\theta_{old}}(d_{i,t} \mid s_{i,t})}
$$

the value function can be formulated as:
  
$$
J(\theta) =
\mathbb{E}
\left[
\frac{1}{|C|}
\sum_{i=1}^{G}
\sum_{t=1}^{T_i}
\min\Big(
\rho_{i,t}(\theta)\hat{A}_{i,t},
\text{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat{A}_{i,t}
\Big)
\right]
$$

where

$$
C = \{(s_{i,t}, d_{i,t}, r_{i,t})\}
$$

and

$$
|C_{train}| =
\left\lfloor \frac{|C|}{DP} \right\rfloor \cdot DP
$$

---

# Reward

the reward is calculated first based on accuracy

$$
R(s_t,a_t) =
\begin{cases}
1 \\
0
\end{cases}
$$

then involve efficiency

$$
r_t = \gamma^{T-t} R_T
$$

$$
\max \mathbb{E}\left[\sum_t \gamma^{T-t} R_T \right]
$$

---

# Inference

the inference process is as normal

$$
s_0 = (q,\; m_s^0,\; \emptyset)
$$

each step, the model gets the updated memory and the action

$$
(t_t,\; m_s^{t+1},\; a_t) \sim \pi_\theta(\cdot \mid s_t)
$$

get the observation

$$
o_t \sim E(\cdot \mid a_t)
$$

the context is updated based on

$$
s_{t+1} = (q,\; m_s^{t+1},\; \{a_t,o_t\})
$$

final unser is get through

$$
a_T = A_n
$$

---

# conclusion
ICLR 2026

