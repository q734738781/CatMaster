slab_cut, slab_fix, gas_fill, adsorbate_placements - These are not registered in the tool registry, so they're not used by the LLM workflow and don't need updating.
But it will be introduced later

Consider Adding:
Knowledge graph for intelligent tool discovery (Tool, Results schema)
Manifest tracking for better artifact management
Stricter workspace isolation and workdir management
Future Enhancements:
Tool capability indexing for better planning
Checkpoint/resume functionality using LangGraph's built-in features
Multi-job parallelization support

MCP-Support- Complex in Engineering
如果你的四个智能体都在同一运行时内协同工作，主要调用的是本地 Python 函数与 jobflow‑remote 的客户端接口，且短期内不打算把这些能力对外部前端或异构代理开放，那么确实不需要上 MCP。此时的瓶颈更多在于图内的状态管理、任务拆解与失败重试，LangGraph 的有状态编排、检查点与本地工具绑定已经能把这条闭环跑得稳定而高效，任何跨进程协议层都会引入额外的序列化与网络开销，却不会让你的规划、编排、执行和总结更“聪明”。

MCP 的收益主要出现在边界被推开的场景。只要一旦出现跨语言或跨主机的工具服务，需要让不同模型前端共享同一组工具，需要对大文件用引用而非文本注入来传递，需要在协议层做统一的参数模式校验与调用审计，或者计划把某些工具交付给外部团队直接复用，那么通过 MCP 暴露这些“边界算子”的价值就会变得明显。它不会替代你的图，也不会改变 jobflow‑remote 的作业与产物追溯，而是把工具的发现与调用语义标准化，使同一后端能力可以被不同的代理栈无缝使用。反之，如果你的多智能体仍然是单团队、单框架、单语言的内部自洽系统，MCP 的好处会被工程复杂度与调用开销抵消。

在这种判断下，更稳妥的路线是继续用 LangGraph 作为唯一的控制与状态内核，把规划、编排、执行与总结的循环完全放在框架内完成，同时对工具接口做两件“面向未来”的约束。第一是把每个工具的输入输出都用清晰的模式声明与校验，尽量采用 JSON 可序列化的扁平结构，并且让返回值以工件引用而不是大块文本为主，这样你的图内实现已经具备了“协议友好”的形态。第二是让与 jobflow‑remote 交互的节点统一通过少量门面函数暴露，控制文件布局、日志路径与标识符的结构化规范，等到需要跨边界开放时，只需要把这些门面替换成 MCP 服务器即可，而上层四个智能体的逻辑与图结构都无需改动。

你可以用一个简单的自检标准来决定是否现在就引入 MCP。若你预计在可预见的周期内不会引入新的宿主模型与代理框架，不会把工具对外部合作者开放，也不会在同一流程中大规模搬运外部数据湖或实验设备产生的原始大文件，那么坚持 LangGraph 直接绑定就是性价比更高的选择。若你已经安排让不同团队通过各自的前端调用你的“设计—验证—解释”能力，或者计划把结构生成、作业单落地、结果采集等做成可共享的服务，那么尽早把这些跨边界节点做成 MCP 工具，会在复用与运维上带来更持久的回报。总之，在你的“规划—编排—执行—总结”的自我探索闭环里，MCP属于可选的扩展接口，而不是完成闭环所必需的组件。
