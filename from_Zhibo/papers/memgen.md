# MemGen Mathematical Formulation

## 1 Problem Setting

给定历史数据

$$
D = \{(x_i, \tau_i)\}_{i=1}^N
$$

训练一个agent $\pi_{\theta}$， 来最大化task reward。 

---

## Training

给定历史数据， 不管使用SFT还是RL，来优化价值函数

$$
\max_{\theta'} \mathbb{E}_{x,\tau \sim \Pi_{\theta,\theta'}} [R(x,\tau)]
$$

写成

$$
J(\theta') =
\mathbb{E} \Big[
A(\tau) \log \Pi_{\theta,\theta'}(\tau)
- \beta \, KL(\Pi_{\theta,\theta'} || \Pi_{ref})
\Big]
$$

$$
\max_{\phi} \;
\mathbb{E} \Big[
R(\tau) - \lambda \sum_{i,j} \max(0, d_{i,j} - \bar{p})
\Big]
$$

$$
\bar{p} =
\frac{1}{|H_{high}|}
\sum_{i \in H_{high}}
\frac{1}{|\tau_i|}
\sum_j d_{i,j}
$$

---

##  Reward

这里的reward的最终来源是任务本身的reward，可以是accuracy，也可以是task success rate

$$
R(\tau)
$$

$$
R_{trigger} =
R(\tau) - \lambda \sum_{i,j} \max(0, d_{i,j} - \bar{p})
$$

$$
A(\tau) = R(\tau) - \mathbb{E}[R]
$$

---

## Inference

起先没有记忆

$$m_0 = \varnothing$$

然后一旦被trigger了, 触发条件是

$$p_j = \sigma T_\phi (H_{t,\lt j})
$$

和

$$
d_j \sim \text{Bernoulli}(p_j)
$$

那么weaver就会接受hiddenstate， 从而输出新的Mt

$$
M_t = W_{\theta'}(H_{t, \lt j})
$$

这些新的M_t会作为context window里面的内容，

$$
z_{t,j} \sim \pi_\theta(\cdot \mid s_t, z_{t,\lt j}, M_t)
$$

最后

$$
A_n = \text{Decode}(z_{1:T})
$$

---

## Conclusion
ICLR 2026 8886 (high points!)
