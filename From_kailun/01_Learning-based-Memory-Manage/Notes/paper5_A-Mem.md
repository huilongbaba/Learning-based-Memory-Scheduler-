# A-Mem: Agentic Memory for LLM Agents

**Source:** [https://arxiv.org/abs/2502.12110](https://arxiv.org/abs/2502.12110) (Feb 2025, v11: Oct 2025)

---

## 符号映射表

|论文原始符号|框架符号|说明|
|---|---|---|
|$m_i$|$m_{\ell}$ 中的条目|单条记忆笔记，存储于外部长期记忆库|
|$M = \lbrace m_1, \dots, m_N \rbrace$|$m_{\ell}$（整体）|外部长期记忆集合|
|$c_i$（original content）|$o$ / $q$|来自交互的原始内容（可能包含用户输入与观测）|
|$X_i$（contextual description）|无直接对应，建议符号 $d_i$（LLM 生成的语义描述）|LLM 为每条记忆生成的上下文摘要|
|$K_i$（keywords）, $G_i$（tags）|无直接对应，建议符号 $\kappa_i$, $\gamma_i$|结构化属性，用于分类与链接|
|$e_i$（embedding）|无直接对应，建议符号 $e_i$|记忆的稠密向量表示|
|$L_i$（linked memories）|无直接对应，建议符号 $\mathcal{L}_i$|与第 $i$ 条记忆关联的记忆集合|
|$q$（query）|$q$|一致：用户查询|
|$f_{\text{enc}}$|$v$（retrieval algorithm 的一部分）|文本编码器，用于生成嵌入向量|
|$M_{\text{near}}^n$, $M_{\text{retrieved}}$|$v(q, m_{\ell})$ 的输出|Top-k 检索结果|
|$P_{s1}, P_{s2}, P_{s3}$（prompt templates）|可视为 $\pi$ 的一部分|控制 LLM 在记忆管理中行为的提示模板|
|LLM（论文中执行记忆操作的模型）|$M$|基础语言模型|

---

### 1. Problem Setting

- **记忆类型：** 本文处理的是 **cross-chat 长期记忆**，对应 $m_{\ell}$。系统在多轮、多 session 的长期对话中持续积累记忆，并在后续交互中检索使用。不涉及 in-chat 的 $m_s$ 管理（上下文窗口内的工作记忆由 LLM 本身处理）。
    
- **决策过程建模：** 本文 **未将记忆管理显式建模为 MDP/POMDP/bandit**。记忆的写入、链接生成、演化均由 LLM 通过 prompt 驱动的启发式规则完成，不存在显式的状态空间、动作空间或策略优化。
    
- **状态空间 $\mathcal{S}$：** 未显式定义。隐式地，状态可理解为当前记忆库 $m_{\ell}$ 的全体内容及其链接结构。
    
- **动作空间 $\mathcal{A}$：** 未显式定义。隐式动作包括：note construction（创建笔记）、link generation（建立链接）、memory evolution（更新已有记忆的属性）、retrieve（检索）。
    
- **观测空间 $\Omega$：** 未显式定义。每次交互的原始内容 $c_i$ 可视为观测。
    
- **记忆数据结构：** 每条记忆是一个多属性笔记（Zettelkasten 风格），包含：
    

|组件|框架符号|描述|
|---|---|---|
|原始内容 $c_i$|$o$|交互的原始文本|
|时间戳 $t_i$|—|交互发生时间|
|关键词 $K_i$|$\kappa_i$（新增）|LLM 生成的关键概念|
|标签 $G_i$|$\gamma_i$（新增）|LLM 生成的分类标签|
|上下文描述 $X_i$|$d_i$（新增）|LLM 生成的语义摘要|
|嵌入向量 $e_i$|$e_i$|稠密向量，用于相似度检索|
|链接集合 $L_i$|$\mathcal{L}_i$（新增）|语义关联的记忆集合|

记忆以向量库 + 结构化属性的混合形式存储，笔记之间通过 $\mathcal{L}_i$ 形成动态图结构（"Box"）。

> A-Mem 的记忆结构比纯向量库（如 MemoryBank）或纯 KV 对更丰富，但整个记忆管理流程完全依赖 LLM 的 prompt 驱动，缺乏形式化的决策框架。这使得系统的行为难以分析和优化，与 RL-based memory scheduling 方法形成鲜明对比。

---

### 2. Training Procedure

- **优化的组件：** **无组件被优化。** 本文不涉及任何训练过程。$M$（LLM）的参数 $\theta$ 完全冻结，检索算法 $v$（即 $f_{\text{enc}}$ + cosine similarity）也使用预训练的 all-minilm-l6-v2，无微调。
    
- **优化算法：** 无。系统完全基于 prompt engineering（$P_{s1}, P_{s2}, P_{s3}$）和预训练模型的 zero-shot 能力。
    
- **训练数据：** 无训练数据。评估使用 LoCoMo 和 DialSim 数据集，但仅用于测试。
    
- **LLM 参数是否冻结：** 是。所有基础模型（GPT-4o, Qwen2.5, Llama 3.2 等）均以 API 或本地推理方式使用，参数不变。
    
- **核心训练目标函数：** 不适用。
    

> 这是本文与 learning-based memory scheduling 方法的根本区别。A-Mem 不学习 $\pi$ 或 $v$，而是依赖 LLM 的 in-context 能力来执行记忆管理决策。这意味着记忆操作的质量完全取决于基础模型的能力和 prompt 设计，无法通过 reward signal 进行改进。

---

### 3. Reward Signal

- **奖励类型：** 不适用。系统无训练循环，不使用任何奖励信号。
    
- **奖励来源：** 不适用。
    
- **奖励分配：** 不适用。
    
- **辅助奖励/正则项：** 不适用。
    

---

### 4. Inference Procedure

- **记忆初始化：** 记忆库 $m_{\ell}$ 初始为空集 $M = \emptyset$。随着对话 session 推进，每次交互触发 note construction，逐步填充记忆库。
    
- **每步决策流程：**
    
    **写入阶段（每次新交互后）：**
    
    1. **Note Construction:** 将交互内容 $c_i$ 和时间戳 $t_i$ 输入 LLM，通过 prompt $P_{s1}$ 生成关键词 $\kappa_i$、标签 $\gamma_i$、上下文描述 $d_i$：
        
        $$\kappa_i, \gamma_i, d_i \leftarrow M(c_i \Vert t_i \Vert P_{s1})$$
        
    2. **Embedding:** 使用文本编码器计算嵌入向量：
        
        $$e_i = f_{\text{enc}}(\text{concat}(c_i, \kappa_i, \gamma_i, d_i))$$
        
    3. **Link Generation:** 对新笔记 $m_n$，通过 cosine similarity 检索 top-k 最近邻 $M_{\text{near}}^n$，再由 LLM 通过 prompt $P_{s2}$ 决定是否建立链接：
        
        $$s_{n,j} = \frac{e_n \cdot e_j}{\lvert e_n \rvert \lvert e_j \rvert}$$
        
        $$M_{\text{near}}^n = \lbrace m_j \mid \text{rank}(s_{n,j}) \leq k, m_j \in M \rbrace$$
        
        $$\mathcal{L}_i \leftarrow M(m_n \Vert M_{\text{near}}^n \Vert P_{s2})$$
        
    4. **Memory Evolution:** 对 $M_{\text{near}}^n$ 中的每条记忆 $m_j$，LLM 通过 prompt $P_{s3}$ 决定是否更新其上下文、关键词和标签：
        
        $$m_j^* \leftarrow M(m_n \Vert M_{\text{near}}^n \setminus m_j \Vert m_j \Vert P_{s3})$$
        
    
    **读取阶段（回答问题时）：**
    
    1. 将查询 $q$ 编码为 $e_q = f_{\text{enc}}(q)$
    2. 计算 $q$ 与所有记忆的 cosine similarity，检索 top-k 记忆
    3. 被检索记忆的关联记忆（同一 Box 内）也被自动获取
    4. 将检索到的记忆作为上下文注入 LLM prompt，生成最终回答 $A_n$
- **推理时额外策略：** Top-k 检索（$k$ 值在不同任务类别和模型间调优，范围 10–50）。无温度调节或多轮 replan 的讨论。
    
- **策略来源：** **完全由手工规则驱动。** 所有决策（何时写入、如何链接、是否演化）均通过固定 prompt 模板实现，非学习得到。
    

> A-Mem 的推理流程清晰且模块化，但每次写入操作需要多次 LLM 调用（note construction + link generation + memory evolution），这在大规模部署中的延迟成本值得关注。作者报告 GPT-4o-mini 平均 5.4 秒/次操作，本地 Llama 3.2 1B 为 1.1 秒。

---

### 5. RQ 分析

### **RQ1： What is memory?**

A-Mem 将记忆定义为多属性结构化笔记的集合 $m_{\ell} = \lbrace m_1, \dots, m_N \rbrace$，每条笔记包含原始内容、LLM 生成的关键词/标签/上下文描述、嵌入向量和链接集合。记忆以向量库 + 属性图的混合形式组织，笔记之间通过语义链接形成动态网络（"Box"结构，受 Zettelkasten 启发）。

### RQ2：How memory evolves, operates?

记忆在运行时通过三步演化：
(1) 新交互触发 note construction，LLM 自主生成语义属性；
(2) link generation 通过 embedding 检索 + LLM 判断建立笔记间链接；
(3) memory evolution 允许新记忆触发对已有记忆属性的更新（上下文描述、关键词、标签均可改变）。检索时，通过 cosine similarity 的 top-k 检索，并沿链接扩展获取关联记忆。

### RQ3：Which component is optimized? Which signal is used?

本文未直接涉及此问题。本文不优化任何组件，不使用任何训练信号。所有记忆操作由冻结的 LLM 通过 prompt engineering 执行，属于 zero-shot 方法。

### RQ4：Regarding online optimization

A-Mem 支持 cross-chat 的在线记忆积累（记忆库随交互持续增长并演化），但这不是"在线优化"，模型参数和策略不随交互更新。系统的"适应性"来源于记忆内容和结构的动态变化，而非 $\pi$ 或 $\theta$ 的更新。

---

### Conclusion

很明显能看的出来 A-Mem 是一个纯 prompt-engineering 驱动的记忆系统，不涉及任何学习或优化过程。这篇文章 提出了一个受 Zettelkasten 笔记方法启发的 LLM agent 记忆系统，核心创新在于让 LLM 自主管理记忆的结构化组织。每条记忆被构建为包含多种语义属性的"笔记"，新记忆加入时系统自动建立与已有记忆的链接，并可触发对历史记忆属性的更新（"记忆演化"）。新的经历不仅会添加新记忆，还会回溯性地修改已有记忆的属性和上下文，使记忆图谱像人类的联想学习一样持续演化。 我认为这本质上就是模型利用自己的理解能力 $\pi_\theta$​ 来主动构建和整理外部记忆 $m_l$，属于记忆从隐式到显式的过程。