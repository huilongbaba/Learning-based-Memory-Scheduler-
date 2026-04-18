https://openreview.net/pdf?id=ddGsiaISXg
# 1. Problem Setting
给定：

$$
q
$$

trajectory

$$
\tau = [z_1, z_2, \dots, z_k]
$$

其中：

$$
z_i = (a_i, o_i, id_i)
$$


优化大模型 $\pi_{\theta}$, 而 $z_i \sim \pi_{\theta} (\cdots |C), 使得

$$
\max_\theta \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
$$

这里的优化目标包括了 提升任务完成率，降低constraint 被违反的比例。
---

# 2. Training


$$
\mathcal{L}(\theta) =
- \mathbb{E}_{u \sim D}
\left[
\frac{1}{|G(u)|}
\sum_{\tau \in G(u)} L_\tau
\right]
$$

$$
L_\tau =
\sum_{(C,y)\in \Sigma(\tau)}
J_{\text{clip}}(y \mid C, A(\tau))
$$

用了GRPO
---

# 3. Reward


$$
R(\tau) =
\begin{cases}
r_{task}, & \text{success} \\
r_{pen}, & \text{violation} \\
0, & \text{otherwise}
\end{cases}
$$

---

# 4. Inference

Action 由llm给出，包括基本action 和对memorz的修改
$$
a_t \sim \pi_\theta(a \mid m_s^t)
$$

Memory Action

$$
a_t^{mem} = (I_{target}, c)
$$

状态转移

$$
m_s^{t+1}
=
\{z_i \in m_s^t \mid id_i \notin I_{target}\}
\oplus
(a_t, o_t)
$$


$$
a_n​∼πθ​(a∣msT​)
$$

---

# 5. Conclusion
也是和memory R1类似，只是解决了一个问题，把rag modifier 和大模型本身统一了
