https://openreview.net/forum?id=JaLXQnA2wi

# Problem
given
$$q
$$
,trajectory
$$\tau = (a_1,o_1,a_2,o_2,\dots,a_T,o_T)
$$
the action is given trhough
$$a_t \sim \pi_\theta(\cdot \mid q, C_t)
$$
the context is
$$C_t = (a_1,o_1,\dots,a_{t-1},o_{t-1})
$$

try to let llm learns when to and how to get $m_s'$
such that
$$|m_s| \ll |\tau|
$$
but
$$\pi_\theta(a_t \mid q, m_s) \approx \pi_\theta(a_t \mid q, \tau)
$$

with this, the context is folded so the trajectory can be longer without reaching context window constraint. 

---

# Training
use a modified GRPO.

to optimize 

$$J(\theta) ==
\mathbb{E}
\left[
\frac{1}{\sum_i |\tau_i|}
\sum_{i=1}^G \sum_{t=1}^{|\tau_i|}
\min
\left(
r_{i,t}(\theta) A_{i,t},
\text{clip}(r_{i,t}(\theta)) A_{i,t}
\right)
\right]
$$

and
$$r_{i,t}(\theta)==
\frac{
\pi_\theta(\tau_{i,t} \mid q, F(\tau_{i,<t}))
}{
\pi_{\theta_{\text{old}}}(\tau_{i,t} \mid q, F(\tau_{i,<t}))
}
$$

where
$$A_{i,t}==
\frac{
\text{clip}(R_i + Q_{i,t},0,1) - \mu
}{
\sigma
}
$$

---

# Reward
there are two kinds of reward. Task success and process reward
$$R_i \in {0,1}
$$
this can be get through llm as judge in the first bench or the execution success as in SWE bench.

the second is to get
$$Q_{i,t}
$$

这包括了Unfolded token penalty: When total context length of the main thread exceeds 50% of the working
context limit,等等对过程的控制

$$|m_s| > \alpha \Rightarrow Q_{i,t} = -1
$$

$$a_t \notin \text{subtask} \Rightarrow Q_{i,t} = -\lambda
$$

$$Q_{i,t} =
\begin{cases}
-1 \\
0
\end{cases}
$$

finally

$$\tilde{R}*{i,t} = R_i + Q*{i,t}
$$

---

# Inference
the inference is like normal inference process,

$$m_s^0 = \emptyset
$$

$$
a_t \sim \pi_\theta(\cdot \mid q, m_s^t)
$$

其中这个$a_t$ 可以包括两种tool，branch或者return，branch用来开启一段新的分支，return则将之前的分支收束，返回新分支总结的信息，回到分支开始的时候。这些内容都由大模型自己
完成。

$$
m_s^{t+1} = m_s^{\text{branch}} + \tilde{o}_t
$$

这里$\tilde{o}_t$指的是大模型对分支内容的总结
 
---

# conclusion
the paper get 2468 on ICLR 2026 , rejected.
