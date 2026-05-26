# langchain_v1 Agent 模块深度解析

> 模块路径：`libs/langchain_v1/langchain/agents/`

---

## 第一阶段：模块整体架构总览

### 1. 核心业务定位

本模块是 LangChain **新一代 Agent 实现**，基于 LangGraph 状态图 + 中间件架构构建。它解决的核心问题是：**如何以一种可组合、可插拔、可扩展的方式构建 LLM Agent，使其具备安全合规、成本控制、人工介入等企业级能力。**

与旧版 `langchain_classic/agents/` 基于 `AgentExecutor` 的线性循环不同，本模块采用：

- **LangGraph StateGraph** 作为底层执行引擎，将 Agent 循环建模为有状态图
- **Middleware 中间件体系** 作为扩展机制，通过钩子函数拦截和修改 Agent 行为
- **结构化输出** 作为一等公民，支持 `ToolStrategy` 和 `ProviderStrategy` 两种模式

### 2. 技术栈 / 依赖

| 依赖 | 用途 |
|------|------|
| `langgraph` | 状态图引擎（StateGraph、ToolNode、Send、Command、interrupt） |
| `langchain-core` | 基础消息类型（AIMessage、ToolMessage、SystemMessage）、BaseTool |
| `pydantic` | 数据校验、JSON Schema 生成 |
| `typing_extensions` | 类型扩展（TypedDict、TypeVar、NotRequired 等） |
| `langsmith` | 可观测性（traceable 装饰器） |

### 3. 设计模式

| 模式 | 应用场景 |
|------|---------|
| **工厂模式** | `create_agent()` 作为唯一入口，根据参数动态构建图 |
| **中间件/洋葱模型** | `AgentMiddleware` 的 `wrap_model_call` / `wrap_tool_call` 链式组合 |
| **策略模式** | `ToolStrategy` / `ProviderStrategy` / `AutoStrategy` 用于结构化输出 |
| **模板方法模式** | `AgentMiddleware` 基类定义钩子，子类选择性覆写 |
| **状态机模式** | LangGraph 的 `StateGraph` + 条件边实现 Agent 循环 |

### 4. 文件清单 + 独立功能

```
agents/
├── __init__.py              # 模块入口，导出 create_agent 和 AgentState
├── factory.py               # 核心工厂函数，构建 LangGraph 状态图
├── structured_output.py     # 结构化输出策略定义（Tool/Provider/Auto）
└── middleware/
    ├── __init__.py           # 中间件统一导出入口
    ├── types.py              # 核心类型定义（AgentMiddleware、ModelRequest、AgentState 等）
    ├── _retry.py             # 重试工具函数（共享于 ModelRetry 和 ToolRetry）
    ├── _redaction.py         # PII 检测与脱敏工具函数
    ├── _execution.py         # Shell 执行策略（Host/Docker/CodexSandbox）
    ├── pii.py                # PII 脱敏中间件
    ├── human_in_the_loop.py  # 人工介入中间件
    ├── tool_retry.py         # 工具调用重试中间件
    ├── model_retry.py        # 模型调用重试中间件
    ├── model_fallback.py     # 模型降级/备用中间件
    ├── model_call_limit.py   # 模型调用次数限制中间件
    ├── tool_call_limit.py    # 工具调用次数限制中间件
    ├── tool_selection.py     # LLM 工具选择中间件
    ├── tool_emulator.py      # 工具模拟中间件（测试用）
    ├── summarization.py      # 对话摘要中间件
    ├── context_editing.py    # 上下文编辑/裁剪中间件
    ├── shell_tool.py         # 持久化 Shell 工具中间件
    ├── file_search.py        # 文件搜索工具中间件（Glob/Grep）
    └── todo.py               # 任务列表管理中间件
```

### 5. 模块整体调用流程

```
用户调用 create_agent(model, tools, middleware, ...)
    │
    ▼
factory.py: 构建 LangGraph StateGraph
    │
    ├── 注册节点：model / tools / middleware hooks
    ├── 注册边：START → before_agent → before_model → model → after_model → tools → before_model → ...
    └── 编译图 → CompiledStateGraph
    │
    ▼
运行时执行流程：
    START
      │
      ▼
    before_agent (所有中间件的 before_agent 钩子，顺序执行)
      │
      ▼
    before_model (所有中间件的 before_model 钩子，顺序执行)
      │
      ▼
    model 节点 (wrap_model_call 洋葱模型 → LLM 调用)
      │
      ▼
    after_model (所有中间件的 after_model 钩子，逆序执行)
      │
      ├── 有 tool_calls? ──→ tools 节点 (wrap_tool_call 洋葱模型 → 工具执行)
      │                          │
      │                          ▼
      │                      回到 before_model (循环)
      │
      └── 无 tool_calls? ──→ after_agent → END
```

---

## 第二阶段：逐文件深度详细讲解

---

### 【__init__.py】

**文件定位：** 模块入口文件，负责对外导出公共 API。

**代码结构：** 仅 9 行代码，导出两个核心符号。

**执行流程详解：**

1. 从 `langchain.agents.factory` 导入 `create_agent` — 这是构建 Agent 的唯一工厂函数
2. 从 `langchain.agents.middleware.types` 导入 `AgentState` — 这是 Agent 的状态类型定义
3. 在 `__all__` 中声明公开导出列表：`["AgentState", "create_agent"]`

**补充说明：** 用户通过 `from langchain.agents import create_agent, AgentState` 即可使用本模块的全部功能。中间件通过 `from langchain.agents.middleware import ...` 单独导入。

---

### 【factory.py】

**文件定位：** 模块核心，实现 `create_agent()` 工厂函数，负责将模型、工具、中间件组装为 LangGraph 状态图。

**代码结构：**

| 函数/类 | 作用 |
|---------|------|
| `create_agent()` | 主入口，构建并返回 `CompiledStateGraph` |
| `_chain_model_call_handlers()` | 将多个 `wrap_model_call` 处理器组合为洋葱模型（同步） |
| `_chain_async_model_call_handlers()` | 同上（异步） |
| `_chain_tool_call_wrappers()` | 将多个 `wrap_tool_call` 处理器组合为洋葱模型（同步） |
| `_chain_async_tool_call_wrappers()` | 同上（异步） |
| `_resolve_schemas()` | 合并中间件状态 schema，生成最终的 state/input/output schema |
| `_get_bound_model()` | 根据请求参数绑定模型（工具、结构化输出） |
| `_handle_model_output()` | 处理模型输出，解析结构化响应 |
| `_execute_model_sync()` | 同步执行模型调用（洋葱模型最内层） |
| `_execute_model_async()` | 异步执行模型调用（洋葱模型最内层） |
| `model_node()` | LangGraph model 节点函数（同步） |
| `amodel_node()` | LangGraph model 节点函数（异步） |
| `_make_model_to_tools_edge()` | model → tools 条件边逻辑 |
| `_make_tools_to_model_edge()` | tools → model 条件边逻辑 |
| `_add_middleware_edge()` | 为中间件节点添加边（支持 jump_to 跳转） |
| `_ComposedExtendedModelResponse` | 组合中间件返回值的内部数据结构 |
| `_normalize_to_model_response()` | 将各种返回值统一为 `ModelResponse` |
| `_build_commands()` | 从模型响应构建 `Command` 列表 |
| `_supports_provider_strategy()` | 检测模型是否支持原生结构化输出 |
| `_handle_structured_output_error()` | 处理结构化输出错误 |
| `_scrub_inputs()` | 清理 LangSmith 追踪输入中的敏感字段 |

**执行流程详解（`create_agent()` 主流程）：**

1. **初始化模型**：如果 `model` 是字符串，调用 `init_chat_model()` 转为 `BaseChatModel` 实例
2. **转换 system_prompt**：字符串转为 `SystemMessage` 对象
3. **处理 response_format**：
   - `None` → 不使用结构化输出
   - `ToolStrategy` / `ProviderStrategy` / `AutoStrategy` → 直接使用
   - 原始 schema（Pydantic/dataclass/TypedDict/dict）→ 包装为 `AutoStrategy`
   - `AutoStrategy` 会在运行时根据模型能力自动选择 `ProviderStrategy` 或 `ToolStrategy`
4. **收集中间件工具**：从每个中间件的 `tools` 属性收集额外工具
5. **组合 wrap_tool_call 处理器**：将所有实现了 `wrap_tool_call` 的中间件按顺序链式组合（第一个为最外层）
6. **组合 wrap_model_call 处理器**：同上，按顺序链式组合
7. **创建 ToolNode**：包含所有可用工具 + wrap_tool_call 包装器
8. **解析状态 schema**：合并所有中间件的 `state_schema` + 基础 `AgentState`，生成最终的 state/input/output TypedDict
9. **创建 StateGraph**：使用解析后的 schema 初始化图
10. **注册节点**：
    - `model` 节点：执行 LLM 调用
    - `tools` 节点：执行工具调用（如果有工具）
    - 每个中间件的 `before_agent` / `before_model` / `after_model` / `after_agent` 节点（如果覆写了对应方法）
11. **注册边**：
    - `START → entry_node`（第一个 before_agent 或 before_model 或 model）
    - `before_agent` 链式连接
    - `before_model` 链式连接 → model
    - `model → after_model` 链式连接
    - `after_model → tools`（条件边，检查 tool_calls）
    - `tools → before_model`（循环回来）
    - `after_model → after_agent → END`（无 tool_calls 时）
12. **编译图**：调用 `graph.compile()` 返回 `CompiledStateGraph`

**洋葱模型组合逻辑（`_chain_model_call_handlers`）：**

```
中间件列表：[M1, M2, M3]
组合结果：M1.wrap_model_call(request, handler=M2.wrap_model_call(request, handler=M3.wrap_model_call(request, handler=_execute_model)))

请求流向：M1 → M2 → M3 → _execute_model
响应流向：_execute_model → M3 → M2 → M1
```

**条件边路由逻辑（`_make_model_to_tools_edge`）：**

1. 检查 `jump_to` 状态字段 → 如果有，按指定跳转
2. 查找最后一个 `AIMessage` → 如果不存在，跳到 END
3. 检查 `tool_calls` → 如果为空，跳到 END
4. 过滤出待执行的工具调用（排除已有 ToolMessage 的和结构化输出工具）
5. 如果有待执行工具 → 返回 `[Send("tools", [tool_call]) for ...]` 并行执行
6. 如果有 `structured_response` → 跳到 END
7. 否则 → 跳回 model（可能是中间件注入的人工 ToolMessage）

**补充说明：**

- `create_agent()` 返回的 `CompiledStateGraph` 可以直接调用 `.invoke()` / `.stream()` / `.astream()` 等方法
- 递归限制设为 9999，防止无限循环
- 所有中间件处理器都通过 `langsmith.traceable` 装饰器包装，支持 LangSmith 追踪
- 中间件名称必须唯一，否则抛出 `AssertionError`

---

### 【structured_output.py】

**文件定位：** 定义结构化输出的策略体系，支持三种方式让 LLM 输出符合 schema 的结果。

**代码结构：**

| 类/函数 | 作用 |
|---------|------|
| `_SchemaSpec` | Schema 描述符，统一处理 Pydantic/dataclass/TypedDict/JSON Schema |
| `ToolStrategy` | 通过工具调用实现结构化输出（将 schema 转为工具让 LLM 调用） |
| `ProviderStrategy` | 通过模型原生能力实现结构化输出（如 OpenAI 的 json_schema 模式） |
| `AutoStrategy` | 自动选择最佳策略（运行时根据模型能力决定） |
| `OutputToolBinding` | 工具策略的绑定信息（schema + tool + 解析逻辑） |
| `ProviderStrategyBinding` | 原生策略的绑定信息（schema + 解析逻辑） |
| `StructuredOutputError` | 结构化输出错误基类 |
| `MultipleStructuredOutputsError` | 多个结构化输出工具调用错误 |
| `StructuredOutputValidationError` | 结构化输出校验失败错误 |
| `_parse_with_schema()` | 通用解析函数，根据 schema 类型解析数据 |

**执行流程详解：**

**ToolStrategy 工作原理：**
1. 将用户提供的 schema 转换为 `StructuredTool`（名称、描述、参数 schema 来自原始 schema）
2. LLM 被绑定这些"伪工具"，通过 `tool_choice="any"` 强制调用
3. LLM 返回的 `tool_calls` 中包含结构化数据
4. 解析 `tool_call["args"]` 为目标类型实例
5. 如果解析失败，根据 `handle_errors` 配置决定是否重试

**ProviderStrategy 工作原理：**
1. 将 schema 转换为 OpenAI 格式的 `response_format`
2. 通过 `model.bind_tools(tools, **kwargs)` 传递给模型
3. 模型直接在 `content` 中返回 JSON 字符串
4. 从 `AIMessage.content` 提取文本，解析 JSON，校验 schema

**AutoStrategy 工作原理：**
1. 在 `create_agent()` 中被转换为 `ToolStrategy`（用于预注册工具）
2. 在运行时 `_get_bound_model()` 中，如果模型支持原生结构化输出，则替换为 `ProviderStrategy`
3. 否则保持 `ToolStrategy`

**补充说明：**

- `ToolStrategy` 支持 Union 类型 schema（如 `PydanticModelA | PydanticModelB`），会为每个变体创建独立工具
- `ProviderStrategy` 仅支持单一 schema
- `handle_errors` 参数支持多种配置：`True`（默认重试）、`False`（抛异常）、异常类型元组、自定义函数

---

### 【middleware/types.py】

**文件定位：** 定义中间件体系的核心类型，是所有中间件的基础。

**代码结构：**

| 类/类型 | 作用 |
|---------|------|
| `AgentMiddleware` | 中间件抽象基类，定义所有钩子方法 |
| `AgentState` | Agent 状态 TypedDict（messages + jump_to + structured_response） |
| `ModelRequest` | 模型请求数据类（model + messages + tools + system_message 等） |
| `ModelResponse` | 模型响应数据类（result + structured_response） |
| `ExtendedModelResponse` | 扩展模型响应（附带 Command 用于额外状态更新） |
| `ModelCallResult` | 类型别名，`ModelResponse | AIMessage | ExtendedModelResponse` |
| `OmitFromSchema` | 标注状态字段是否从 input/output schema 中省略 |
| `JumpTo` | 跳转目标类型，`Literal["tools", "model", "end"]` |
| `hook_config` | 装饰器，声明钩子支持的跳转目标 |
| `before_agent` / `after_agent` / `before_model` / `after_model` | 便捷装饰器 |
| `wrap_model_call` / `wrap_tool_call` | 便捷装饰器 |
| `dynamic_prompt` | 动态 Prompt 装饰器 |

**AgentMiddleware 钩子方法详解：**

| 钩子 | 触发时机 | 作用 | 返回值 |
|------|---------|------|--------|
| `before_agent(state, runtime)` | Agent 执行开始前（仅一次） | 初始化状态 | `dict` 状态更新或 `None` |
| `abefore_agent` | 同上（异步） | 同上 | 同上 |
| `before_model(state, runtime)` | 每次模型调用前 | 修改状态、检查限制 | `dict` 状态更新或 `None` |
| `abefore_model` | 同上（异步） | 同上 | 同上 |
| `after_model(state, runtime)` | 每次模型调用后 | 检查输出、注入消息 | `dict` 状态更新或 `None` |
| `aafter_model` | 同上（异步） | 同上 | 同上 |
| `after_agent(state, runtime)` | Agent 执行结束后（仅一次） | 清理资源 | `dict` 状态更新或 `None` |
| `aafter_agent` | 同上（异步） | 同上 | 同上 |
| `wrap_model_call(request, handler)` | 包裹模型调用 | 重试、缓存、修改请求/响应 | `ModelResponse` / `AIMessage` / `ExtendedModelResponse` |
| `awrap_model_call` | 同上（异步） | 同上 | 同上 |
| `wrap_tool_call(request, handler)` | 包裹工具调用 | 重试、模拟、修改请求/响应 | `ToolMessage` / `Command` |
| `awrap_tool_call` | 同上（异步） | 同上 | 同上 |

**ModelRequest 核心字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `model` | `BaseChatModel` | 当前使用的 LLM |
| `messages` | `list[AnyMessage]` | 对话消息列表（不含 system_message） |
| `system_message` | `SystemMessage \| None` | 系统提示词 |
| `tools` | `list[BaseTool \| dict]` | 可用工具列表 |
| `tool_choice` | `Any \| None` | 工具选择配置 |
| `response_format` | `ResponseFormat \| None` | 结构化输出配置 |
| `state` | `AgentState` | 当前 Agent 状态 |
| `runtime` | `Runtime` | LangGraph 运行时上下文 |
| `model_settings` | `dict` | 额外模型参数 |

**ModelRequest.override() 方法：** 不可变模式，返回新的 `ModelRequest` 实例，原始实例不变。支持覆盖任意字段。

**AgentState 核心字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `list[AnyMessage]` | 对话消息列表（使用 `add_messages` reducer） |
| `jump_to` | `JumpTo \| None` | 跳转指令（EphemeralValue，不持久化） |
| `structured_response` | `ResponseT \| None` | 结构化输出结果（从 input schema 省略） |

**补充说明：**

- `hook_config(can_jump_to=["end"])` 装饰器声明该钩子可以设置 `jump_to` 跳转目标
- `OmitFromSchema` 用于控制状态字段在 input/output schema 中的可见性
- `PrivateStateAttr` 是 `OmitFromSchema(input=True, output=True)` 的快捷方式，表示纯内部字段
- `wrap_model_call` 和 `wrap_tool_call` 采用洋葱模型，第一个中间件为最外层

---

### 【middleware/_retry.py】

**文件定位：** 重试工具函数，被 `ModelRetryMiddleware` 和 `ToolRetryMiddleware` 共享。

**代码结构：**

| 函数/类型 | 作用 |
|-----------|------|
| `RetryOn` | 类型别名：异常类型元组 或 判断函数 |
| `OnFailure` | 类型别名：失败处理策略（`"error"` / `"continue"` / 自定义函数） |
| `validate_retry_params()` | 校验重试参数（max_retries >= 0, delays >= 0 等） |
| `should_retry_exception()` | 判断异常是否应触发重试 |
| `calculate_delay()` | 计算指数退避延迟（支持 jitter） |

**calculate_delay 计算逻辑：**

```
delay = initial_delay * (backoff_factor ** retry_number)
delay = min(delay, max_delay)
if jitter:
    delay += random.uniform(-delay*0.25, delay*0.25)
    delay = max(0, delay)
```

---

### 【middleware/_redaction.py】

**文件定位：** PII 检测与脱敏工具函数，被 `PIIMiddleware` 和 `ShellToolMiddleware` 共享。

**代码结构：**

| 函数/类 | 作用 |
|---------|------|
| `RedactionRule` | 脱敏规则定义（PII 类型 + 策略 + 自定义检测器） |
| `ResolvedRedactionRule` | 解析后的脱敏规则（检测器已确定） |
| `PIIMatch` | PII 匹配结果（type + value + start + end） |
| `PIIDetectionError` | PII 检测到时抛出的异常（block 策略） |
| `detect_email()` | 检测邮箱地址 |
| `detect_credit_card()` | 检测信用卡号（含 Luhn 校验） |
| `detect_ip()` | 检测 IP 地址（IPv4/IPv6） |
| `detect_mac_address()` | 检测 MAC 地址 |
| `detect_url()` | 检测 URL |
| `apply_strategy()` | 根据策略处理 PII（block/redact/mask/hash） |

**脱敏策略详解：**

| 策略 | 行为 |
|------|------|
| `block` | 抛出 `PIIDetectionError`，终止运行 |
| `redact` | 替换为 `[REDACTED_{type}]` |
| `mask` | 保留首尾字符，中间用 `*` 替换 |
| `hash` | 用 SHA256 哈希前 8 位替换 |

---

### 【middleware/_execution.py】

**文件定位：** Shell 执行策略定义，被 `ShellToolMiddleware` 使用。

**代码结构：**

| 类 | 作用 |
|----|------|
| `BaseExecutionPolicy` | 抽象基类，定义超时、输出限制等通用配置 |
| `HostExecutionPolicy` | 直接在宿主机执行（支持 CPU/内存限制） |
| `CodexSandboxExecutionPolicy` | 通过 Codex CLI 沙箱执行 |
| `DockerExecutionPolicy` | 在 Docker 容器中执行（最强隔离） |

**三种策略对比：**

| 维度 | Host | CodexSandbox | Docker |
|------|------|-------------|--------|
| 隔离级别 | 无 | syscall/文件系统限制 | 容器级隔离 |
| 适用场景 | 可信环境 | 有 Codex CLI 的环境 | 不可信用户输入 |
| 网络控制 | 无 | 无 | 默认 `--network none` |
| 文件系统 | 完全访问 | 受限 | 可设只读 |
| 资源限制 | prlimit | 无 | Docker 原生 |

---

### 【middleware/pii.py】

**文件定位：** PII 脱敏中间件，在模型调用和工具调用过程中自动检测和脱敏敏感信息。

**代码结构：**

| 类 | 作用 |
|----|------|
| `PIIMiddleware` | 主中间件类 |
| `_PIIStreamTransformer` | 流式传输时的 PII 检测与脱敏 |

**核心逻辑：**

1. **before_model**：对工具返回结果中的 PII 进行脱敏（`apply_to_tool_results`）
2. **wrap_model_call**：对发送给模型的消息进行脱敏
3. **流式脱敏**：`_PIIStreamTransformer` 在流式传输过程中实时检测和脱敏 PII
   - 维护滑动窗口缓冲区（默认 128 字符），处理跨 delta 边界的 PII
   - 支持 text-delta、reasoning-delta、tool_call_chunk 等多种流式事件
   - 在 `content-block-finish` 时刷新缓冲区，确保完整检测

**补充说明：** 这是整个模块中最复杂的中间件之一，流式脱敏的实现需要处理多种事件类型和边界情况。

---

### 【middleware/human_in_the_loop.py】

**文件定位：** 人工介入中间件，在工具执行前暂停等待人工审批。

**代码结构：**

| 类/类型 | 作用 |
|---------|------|
| `HumanInTheLoopMiddleware` | 主中间件类 |
| `ActionRequest` | 动作请求（工具名 + 参数 + 描述） |
| `ReviewConfig` | 审批配置（允许的决策类型 + 参数 schema） |
| `HITLRequest` | 人工审批请求（多个 ActionRequest + ReviewConfig） |
| `HITLResponse` | 人工审批响应（决策列表） |
| `ApproveDecision` | 批准决策 |
| `EditDecision` | 编辑决策（修改工具名/参数） |
| `RejectDecision` | 拒绝决策（返回错误消息） |
| `RespondDecision` | 代替工具响应决策（跳过工具执行） |
| `InterruptOnConfig` | 中断配置（允许的决策 + 描述 + 参数 schema） |

**核心逻辑（after_model 钩子）：**

1. 获取最后一个 `AIMessage`，检查其 `tool_calls`
2. 对每个需要审批的工具调用，创建 `ActionRequest` 和 `ReviewConfig`
3. 调用 `interrupt(hitl_request)` 暂停执行，等待人工输入
4. 根据人工决策处理每个工具调用：
   - **approve**：保留原始 tool_call
   - **edit**：替换 tool_call 的 name/args
   - **reject**：生成错误 ToolMessage，跳过工具执行
   - **respond**：生成成功 ToolMessage，跳过工具执行（"ask user" 模式）
5. 更新 `AIMessage.tool_calls`，只保留批准/编辑后的调用
6. 返回更新后的消息列表

**补充说明：** 使用 LangGraph 的 `interrupt()` 机制实现暂停/恢复，支持持久化 checkpoint。

---

### 【middleware/tool_retry.py】

**文件定位：** 工具调用重试中间件，在工具执行失败时自动重试。

**核心逻辑（wrap_tool_call 钩子）：**

1. 检查当前工具是否在重试范围内（`_should_retry_tool`）
2. 循环执行 `max_retries + 1` 次：
   - 调用 `handler(request)` 执行工具
   - 成功则返回结果
   - 失败则检查异常是否可重试（`should_retry_exception`）
   - 如果可重试且还有剩余次数，计算延迟并等待（`calculate_delay`）
   - 如果不可重试或次数用尽，调用 `_handle_failure`
3. `_handle_failure` 根据 `on_failure` 配置：
   - `"error"`：重新抛出异常
   - `"continue"`：返回包含错误信息的 `ToolMessage`
   - 自定义函数：调用函数生成错误消息

**补充说明：** 支持通过 `tools` 参数指定只对特定工具重试。

---

### 【middleware/model_retry.py】

**文件定位：** 模型调用重试中间件，在模型调用失败时自动重试。

**核心逻辑：** 与 `ToolRetryMiddleware` 几乎完全对称，区别在于：
- 使用 `wrap_model_call` 钩子而非 `wrap_tool_call`
- 失败时返回 `AIMessage` 而非 `ToolMessage`
- 失败时返回 `ModelResponse` 包装

---

### 【middleware/model_fallback.py】

**文件定位：** 模型降级中间件，在主模型失败时依次尝试备用模型。

**核心逻辑（wrap_model_call 钩子）：**

1. 先尝试主模型（调用 `handler(request)`）
2. 如果失败，依次尝试每个备用模型（调用 `handler(request.override(model=fallback_model))`）
3. 如果所有模型都失败，抛出最后一个异常

**补充说明：** 备用模型通过字符串标识或 `BaseChatModel` 实例指定，字符串会通过 `init_chat_model()` 初始化。

---

### 【middleware/model_call_limit.py】

**文件定位：** 模型调用次数限制中间件，防止无限循环和成本失控。

**代码结构：**

| 类 | 作用 |
|----|------|
| `ModelCallLimitMiddleware` | 主中间件类 |
| `ModelCallLimitState` | 扩展状态（thread_model_call_count + run_model_call_count） |
| `ModelCallLimitExceededError` | 超限异常 |

**核心逻辑：**

- **before_model**：检查调用次数是否超限
  - 超限 + `exit_behavior="error"` → 抛出 `ModelCallLimitExceededError`
  - 超限 + `exit_behavior="end"` → 设置 `jump_to="end"` + 注入提示 AIMessage
- **after_model**：递增调用计数

**两级计数：**
- `thread_model_call_count`：线程级，跨多次运行持久化
- `run_model_call_count`：运行级，单次运行内计数（使用 `UntrackedValue`，不持久化）

---

### 【middleware/tool_call_limit.py】

**文件定位：** 工具调用次数限制中间件，控制工具使用频率。

**代码结构：**

| 类 | 作用 |
|----|------|
| `ToolCallLimitMiddleware` | 主中间件类 |
| `ToolCallLimitState` | 扩展状态（thread_tool_call_count + run_tool_call_count） |
| `ToolCallLimitExceededError` | 超限异常 |

**核心逻辑（after_model 钩子）：**

1. 获取最后一个 `AIMessage` 的 `tool_calls`
2. 将工具调用分为 allowed 和 blocked 两组
3. 对 blocked 的工具调用：
   - `exit_behavior="continue"`：生成错误 ToolMessage，允许其他工具继续
   - `exit_behavior="error"`：抛出 `ToolCallLimitExceededError`
   - `exit_behavior="end"`：设置 `jump_to="end"` + 生成 ToolMessage + AIMessage

**补充说明：** 支持按工具名限制（`tool_name` 参数），不指定则限制所有工具。

---

### 【middleware/tool_selection.py】

**文件定位：** LLM 工具选择中间件，在主模型调用前用一个小模型筛选最相关的工具。

**核心逻辑（wrap_model_call 钩子）：**

1. 准备选择请求：提取可用工具、最后一条用户消息、系统提示词
2. 创建动态结构化输出 schema：每个工具名为 `Literal` 类型，带描述
3. 调用选择模型（`model.with_structured_output(schema)`）获取工具选择结果
4. 根据选择结果过滤 `request.tools`，保留 `always_include` 工具
5. 调用 `handler(modified_request)` 执行主模型

**补充说明：** 适用于工具数量很多的场景（如 50+ 工具），减少 token 消耗并提高工具选择准确性。

---

### 【middleware/tool_emulator.py】

**文件定位：** 工具模拟中间件，用 LLM 模拟工具执行结果，用于测试。

**核心逻辑（wrap_tool_call 钩子）：**

1. 检查当前工具是否需要模拟（`emulate_all` 或在 `tools_to_emulate` 中）
2. 如果不需要模拟，调用 `handler(request)` 正常执行
3. 如果需要模拟，构建 Prompt 让 LLM 生成模拟响应
4. 返回包含模拟结果的 `ToolMessage`，跳过真实工具执行

**补充说明：** 默认使用 `anthropic:claude-sonnet-4-5-20250929` 作为模拟模型，`temperature=1` 以引入变化。

---

### 【middleware/summarization.py】

**文件定位：** 对话摘要中间件，在 token 数接近限制时自动摘要旧消息。

**代码结构：**

| 类/类型 | 作用 |
|---------|------|
| `SummarizationMiddleware` | 主中间件类 |
| `ContextSize` | 上下文大小规格（fraction / tokens / messages） |

**核心逻辑（before_model 钩子）：**

1. 计算当前消息总 token 数
2. 检查是否触发摘要条件（`_should_summarize`）：
   - 消息数 >= 阈值
   - token 数 >= 阈值
   - token 占模型最大输入的比例 >= 阈值
   - 最后一条 AIMessage 报告的 token 数 >= 阈值
3. 确定截断点（`_determine_cutoff_index`）：
   - 基于 token 的截断：使用二分查找确定最早可保留的消息索引
   - 基于消息数的截断：直接按数量截断
4. 分割消息：待摘要部分 + 保留部分
5. 调用摘要模型生成摘要
6. 构建新消息列表：`[RemoveMessage(REMOVE_ALL_MESSAGES), 摘要SystemMessage, ...保留消息]`

**补充说明：**

- 使用 `REMOVE_ALL_MESSAGES` + `add_messages` reducer 实现消息替换
- 摘要 Prompt 默认包含 SESSION INTENT / SUMMARY / ARTIFACTS / NEXT STEPS 四个部分
- 支持 Anthropic 模型的特殊 token 计数（chars_per_token=3.3）

---

### 【middleware/context_editing.py】

**文件定位：** 上下文编辑中间件，在 token 超限时自动清理旧的工具结果。

**代码结构：**

| 类 | 作用 |
|----|------|
| `ContextEditingMiddleware` | 主中间件类 |
| `ClearToolUsesEdit` | 清理工具结果的编辑策略 |
| `ContextEdit` | 编辑策略协议 |

**核心逻辑（wrap_model_call 钩子）：**

1. 深拷贝消息列表
2. 对每个编辑策略调用 `edit.apply(messages, count_tokens=...)`
3. `ClearToolUsesEdit.apply()` 逻辑：
   - 计算当前 token 数，如果未超限则返回
   - 找到所有 `ToolMessage`（排除最近 N 条）
   - 对每条旧 `ToolMessage`：
     - 找到对应的 `AIMessage` 和 `tool_call`
     - 检查是否在排除列表中
     - 将 `content` 替换为占位符（默认 `"[cleared]"`）
     - 可选：同时清理 `tool_call.args`
   - 在 `response_metadata` 中标记已清理
4. 使用修改后的消息调用 `handler(request.override(messages=edited_messages))`

**补充说明：** 灵感来自 Anthropic 的 `clear_tool_uses` 功能，保持与 Claude API 的行为一致。

---

### 【middleware/shell_tool.py】

**文件定位：** 持久化 Shell 工具中间件，为 Agent 提供一个持久的终端会话。

**代码结构：**

| 类 | 作用 |
|----|------|
| `ShellToolMiddleware` | 主中间件类 |
| `ShellSession` | 持久化 Shell 会话管理 |
| `ShellToolState` | 扩展状态（shell_session_resources） |
| `CommandExecutionResult` | 命令执行结果 |
| `_ShellToolInput` | Shell 工具输入 schema |

**核心逻辑：**

1. **初始化**：配置工作目录、执行策略、脱敏规则、启动/关闭命令
2. **before_agent**：创建临时工作目录，启动 Shell 进程，执行启动命令
3. **工具执行**：
   - 通过 `stdin` 发送命令
   - 发送带唯一标记的 `printf` 命令获取退出码
   - 通过 `stdout/stderr` 读取线程收集输出
   - 支持超时、输出行数/字节数限制
   - 超时后自动重启 Shell
4. **after_agent**：执行关闭命令，停止 Shell 进程，清理临时目录
5. **脱敏**：对命令输出应用 `RedactionRule`，在返回给模型前脱敏

**ShellSession 执行流程：**

```
写入命令到 stdin → 写入标记命令获取退出码 → 
从 queue 中读取输出直到标记出现 → 
收集 stderr 残留 → 返回 CommandExecutionResult
```

**补充说明：** 使用 `weakref.finalize` 确保资源清理，即使 Agent 异常退出也不会泄漏进程。

---

### 【middleware/file_search.py】

**文件定位：** 文件搜索工具中间件，为 Agent 提供 Glob 和 Grep 搜索能力。

**代码结构：**

| 类 | 作用 |
|----|------|
| `FilesystemFileSearchMiddleware` | 主中间件类 |
| `glob_search` | Glob 文件搜索工具 |
| `grep_search` | Grep 内容搜索工具 |

**核心逻辑：**

1. **glob_search**：
   - 验证路径安全性（防止路径遍历）
   - 使用 `pathlib.glob()` 匹配文件
   - 按修改时间排序返回

2. **grep_search**：
   - 优先使用 `ripgrep`（如果可用）
   - 回退到 Python 正则搜索
   - 支持三种输出模式：`files_with_matches` / `content` / `count`
   - 支持文件类型过滤（`include` 参数）

**补充说明：** 所有路径都经过验证，确保不会访问 `root_path` 之外的文件。

---

### 【middleware/todo.py】

**文件定位：** 任务列表管理中间件，为 Agent 提供任务规划和跟踪能力。

**代码结构：**

| 类/类型 | 作用 |
|---------|------|
| `TodoListMiddleware` | 主中间件类 |
| `PlanningState` | 扩展状态（todos 列表） |
| `Todo` | 单个任务（content + status） |
| `write_todos` | 任务管理工具 |

**核心逻辑：**

1. **wrap_model_call**：在系统提示词中追加任务管理指南
2. **after_model**：检测并阻止并行的 `write_todos` 调用（因为每次调用会替换整个列表）
3. **write_todos 工具**：
   - 接收 `todos` 列表
   - 返回 `Command(update={"todos": todos, "messages": [...]})`
   - 同时更新状态和消息

**补充说明：** 工具描述非常详细（约 60 行），指导 LLM 何时使用、如何管理任务状态。

---

### 【middleware/__init__.py】

**文件定位：** 中间件统一导出入口，将所有中间件类和辅助类型汇总导出。

**导出清单（37 个符号）：**

| 分类 | 导出项 |
|------|--------|
| 核心类型 | `AgentMiddleware`, `AgentState`, `ModelRequest`, `ModelResponse`, `ModelCallResult`, `ExtendedModelResponse`, `ToolCallRequest` |
| 钩子装饰器 | `before_agent`, `after_agent`, `before_model`, `after_model`, `wrap_model_call`, `wrap_tool_call`, `hook_config`, `dynamic_prompt` |
| 中间件类 | `PIIMiddleware`, `HumanInTheLoopMiddleware`, `ToolRetryMiddleware`, `ModelRetryMiddleware`, `ModelFallbackMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `LLMToolSelectorMiddleware`, `LLMToolEmulator`, `SummarizationMiddleware`, `ContextEditingMiddleware`, `ShellToolMiddleware`, `FilesystemFileSearchMiddleware`, `TodoListMiddleware` |
| 辅助类型 | `PIIDetectionError`, `InterruptOnConfig`, `ClearToolUsesEdit`, `RedactionRule`, `HostExecutionPolicy`, `DockerExecutionPolicy`, `CodexSandboxExecutionPolicy`, `Runtime` |

---

## 附录：中间件速查表

| 中间件 | 钩子 | 核心功能 | 企业场景 |
|--------|------|---------|---------|
| `PIIMiddleware` | before_model, wrap_model_call, 流式 | PII 检测与脱敏 | 合规审查、数据保护 |
| `HumanInTheLoopMiddleware` | after_model | 工具执行前人工审批 | 高危操作确认、审计 |
| `ToolRetryMiddleware` | wrap_tool_call | 工具调用失败重试 | 提高工具调用可靠性 |
| `ModelRetryMiddleware` | wrap_model_call | 模型调用失败重试 | 应对 API 限流/超时 |
| `ModelFallbackMiddleware` | wrap_model_call | 模型降级切换 | 高可用、成本优化 |
| `ModelCallLimitMiddleware` | before_model, after_model | 限制模型调用次数 | 成本控制 |
| `ToolCallLimitMiddleware` | after_model | 限制工具调用次数 | 防止无限循环 |
| `LLMToolSelectorMiddleware` | wrap_model_call | LLM 筛选相关工具 | 大量工具场景优化 |
| `LLMToolEmulator` | wrap_tool_call | LLM 模拟工具执行 | 测试、开发 |
| `SummarizationMiddleware` | before_model | 自动摘要长对话 | 长对话场景 |
| `ContextEditingMiddleware` | wrap_model_call | 清理旧工具结果 | 上下文窗口管理 |
| `ShellToolMiddleware` | before_agent, after_agent, 工具 | 持久化 Shell 会话 | 代码执行、系统管理 |
| `FilesystemFileSearchMiddleware` | 工具 | 文件搜索（Glob/Grep） | 代码分析、文件管理 |
| `TodoListMiddleware` | wrap_model_call, after_model, 工具 | 任务列表管理 | 复杂任务规划 |
