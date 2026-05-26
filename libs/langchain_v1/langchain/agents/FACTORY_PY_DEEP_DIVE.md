# factory.py 深度详解

> 文件路径：`e:\AI\GitHub\langchain\libs\langchain_v1\langchain\agents\factory.py`
> 总行数：1889 行

---

## 一、文件定位

`factory.py` 是 Agent 引擎的**构建工厂**。它实现了 `create_agent()` 函数，负责：

1. 解析中间件配置，提取各类钩子
2. 组合洋葱模型调用链（`wrap_model_call` / `wrap_tool_call`）
3. 构建 LangGraph 状态图（节点 + 边 + 条件边）
4. 编译并返回可执行的 `CompiledStateGraph`

**一句话总结：** 这个文件把"中间件配置"转化为"可运行的状态图"。

---

## 二、代码结构总览

```
factory.py
├── 导入区（L1-L55）
├── _ComposedExtendedModelResponse 数据类（L57-L72）
├── 类型别名定义（L74-L107）
├── 常量与模板（L109-L155）
├── _scrub_inputs 工具函数（L157-L166）
├── _normalize_to_model_response（L168-L184）
├── _build_commands（L186-L220）
├── _chain_model_call_handlers（L222-L349）★洋葱模型-同步★
├── _chain_async_model_call_handlers（L351-L478）★洋葱模型-异步★
├── _get_schema_type_hints（L480-L483）
├── _resolve_schemas / _resolve_schema（L485-L542）
├── _extract_metadata（L544-L563）
├── _get_can_jump_to（L565-L594）
├── _supports_provider_strategy（L596-L643）
├── _handle_structured_output_error（L645-L687）
├── _chain_tool_call_wrappers（L689-L735）★工具洋葱模型-同步★
├── _chain_async_tool_call_wrappers（L737-L797）★工具洋葱模型-异步★
├── create_agent 函数（L799-L1689）★★★核心★★★
│   ├── 参数处理与模型初始化（L900-L920）
│   ├── 结构化输出配置（L922-L950）
│   ├── 中间件工具收集（L952-L960）
│   ├── wrap_tool_call 链组合（L962-L990）
│   ├── awrap_tool_call 链组合（L992-L1010）
│   ├── ToolNode 创建（L1012-L1040）
│   ├── 中间件验证与分类（L1042-L1090）
│   ├── wrap_model_call 链组合（L1092-L1120）
│   ├── Schema 合并（L1122-L1135）
│   ├── StateGraph 创建（L1137-L1145）
│   ├── _handle_model_output（L1147-L1260）
│   ├── _get_bound_model（L1262-L1390）
│   ├── _execute_model_sync / _execute_model_async（L1392-L1440）
│   ├── model_node / amodel_node（L1442-L1490）
│   ├── 图节点添加（L1492-L1560）
│   ├── 边连接逻辑（L1562-L1660）
│   └── 图编译与配置（L1662-L1689）
├── _resolve_jump（L1691-L1703）
├── _fetch_last_ai_and_tool_messages（L1705-L1726）
├── _make_model_to_tools_edge（L1728-L1790）★条件边-模型到工具★
├── _make_model_to_model_edge（L1792-L1820）★条件边-模型到模型★
├── _make_tools_to_model_edge（L1822-L1860）★条件边-工具到模型★
└── _add_middleware_edge（L1862-L1889）★中间件边添加★
```

---

## 三、逐段深度详解

---

### 3.1 导入区（L1-L55）

```python
from langgraph._internal._runnable import RunnableCallable
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolCallTransformer
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Command, Send
```

**关键导入解析：**
- `RunnableCallable`：LangGraph 的可调用包装器，支持同步/异步双模式。用于将普通函数包装为图节点
- `StateGraph`：LangGraph 的状态图构建器。`create_agent()` 的核心输出就是一个 `StateGraph`
- `ToolNode`：LangGraph 的工具执行节点。它接收 `tool_calls`，执行对应工具，返回 `ToolMessage`
- `Send`：LangGraph 的并行分发指令。当模型返回多个 tool_calls 时，用 `Send` 并行执行
- `Command`：LangGraph 的状态更新指令，支持 `update`（更新状态）、`goto`（跳转节点）、`resume`（恢复中断）
- `ToolCallTransformer`：LangGraph 内置的流式转换器，处理工具调用的流式输出

```python
from langchain.agents.structured_output import (
    AutoStrategy, MultipleStructuredOutputsError, OutputToolBinding,
    ProviderStrategy, ProviderStrategyBinding, ResponseFormat,
    StructuredOutputError, StructuredOutputValidationError, ToolStrategy,
)
from langchain.chat_models import init_chat_model
```

**结构化输出相关导入：**
- `ToolStrategy`：通过工具调用实现结构化输出（通用，所有模型都支持）
- `ProviderStrategy`：通过模型原生能力实现结构化输出（如 OpenAI 的 response_format）
- `AutoStrategy`：自动检测模型能力，选择最佳策略
- `OutputToolBinding`：结构化输出工具的绑定信息（工具实例 + 解析函数）

---

### 3.2 _ComposedExtendedModelResponse 数据类（L57-L72）

```python
@dataclass
class _ComposedExtendedModelResponse(Generic[ResponseT]):
    model_response: ModelResponse[ResponseT]
    commands: list[Command[Any]] = field(default_factory=list)
```

**与 `ExtendedModelResponse` 的区别：**

| 维度 | ExtendedModelResponse | _ComposedExtendedModelResponse |
|------|----------------------|-------------------------------|
| 用途 | 用户面向，单个 Command | 内部使用，Command 列表 |
| Command 数量 | 0 或 1 个 | 0 到 N 个 |
| 使用场景 | 中间件返回值 | 洋葱模型组合过程中的中间结果 |

**为什么需要两个类？** 在洋葱模型组合过程中，多层中间件可能各自产生 Command。`_ComposedExtendedModelResponse` 收集所有 Command，然后在 `_build_commands` 中统一处理。

**commands 的顺序：** 内层先添加，外层后添加（inner-first, then outer）。对于非 reducer 字段，外层覆盖内层。

---

### 3.3 类型别名定义（L74-L107）

```python
_ModelCallHandler = Callable[
    [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], ModelResponse]],
    ModelResponse | AIMessage | ExtendedModelResponse,
]

_ComposedModelCallHandler = Callable[
    [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], ModelResponse]],
    _ComposedExtendedModelResponse,
]
```

**两种 Handler 的区别：**
- `_ModelCallHandler`：单个中间件的 wrap_model_call 签名，返回用户可见的三种类型
- `_ComposedModelCallHandler`：组合后的 Handler 签名，返回内部使用的 `_ComposedExtendedModelResponse`

**为什么组合后的返回类型不同？** 因为组合过程中需要收集所有中间件的 Command，而单个中间件只需要返回自己的响应。`_to_composed_result` 负责将用户可见的返回类型转换为内部类型。

---

### 3.4 常量与模板（L109-L155）

#### STRUCTURED_OUTPUT_ERROR_TEMPLATE

```python
STRUCTURED_OUTPUT_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."
```

**作用：** 结构化输出验证失败时，发送给模型的错误消息模板。模型看到这条消息后会尝试修正输出。

#### DYNAMIC_TOOL_ERROR_TEMPLATE

```python
DYNAMIC_TOOL_ERROR_TEMPLATE = """
Middleware added tools that the agent doesn't know how to execute.

Unknown tools: {unknown_tool_names}
Registered tools: {available_tool_names}

This happens when middleware modifies `request.tools` in `wrap_model_call` to include
tools that weren't passed to `create_agent()`.

How to fix this:
Option 1: Register tools at agent creation (recommended)
Option 2: Handle dynamic tools in middleware (implement wrap_tool_call)
"""
```

**作用：** 当中间件在 `wrap_model_call` 中动态添加了工具，但没有在 `create_agent()` 中注册时，抛出此错误。这是一个非常友好的错误提示，告诉用户两种修复方式。

#### FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT

```python
FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT = [
    "grok", "gpt-5", "gpt-4.1", "gpt-4o", "gpt-oss",
]
```

**作用：** 当模型没有 profile 数据时，用这个列表作为后备判断。如果模型名称包含列表中的任一部分，就认为它支持 ProviderStrategy。

**为什么需要后备列表？** 不是所有模型都有 LangChain 的 profile 数据（特别是自定义模型或新发布的模型）。后备列表确保常见模型即使没有 profile 也能正确工作。

---

### 3.5 _scrub_inputs 工具函数（L157-L166）

```python
def _scrub_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    filtered = inputs.copy()
    filtered.pop("handler", None)
    req = filtered.get("request")
    if isinstance(req, (ModelRequest, ToolCallRequest)):
        filtered["request"] = {
            f.name: getattr(req, f.name) for f in fields(req) if f.name != "runtime"
        }
    return filtered
```

**作用：** 在发送追踪数据到 LangSmith 之前，清理输入中的敏感信息。

**清理内容：**
1. 移除 `handler`（不需要追踪内部回调函数）
2. 将 `request` 对象展开为字典，但排除 `runtime`（运行时上下文可能包含敏感信息如 API key）

**使用位置：** 作为 `traceable(process_inputs=_scrub_inputs)` 的参数，在 `wrap_model_call` 和 `wrap_tool_call` 的 LangSmith 追踪中使用。

---

### 3.6 _normalize_to_model_response（L168-L184）

```python
def _normalize_to_model_response(
    result: ModelResponse | AIMessage | ExtendedModelResponse,
) -> ModelResponse:
    if isinstance(result, AIMessage):
        return ModelResponse(result=[result], structured_response=None)
    if isinstance(result, ExtendedModelResponse):
        return result.model_response
    return result
```

**作用：** 将中间件的三种可能返回值统一转换为 `ModelResponse`。

**转换规则：**
- `AIMessage` → `ModelResponse(result=[ai_msg], structured_response=None)`：简化返回自动包装
- `ExtendedModelResponse` → 提取内部的 `model_response`，Command 在外层处理
- `ModelResponse` → 直接返回，不做转换

**为什么需要归一化？** 在洋葱模型组合中，内层 handler 总是返回 `ModelResponse` 给外层。这样外层中间件不需要处理三种不同的返回类型。

---

### 3.7 _build_commands（L186-L220）

```python
def _build_commands(
    model_response: ModelResponse,
    middleware_commands: list[Command[Any]] | None = None,
) -> list[Command[Any]]:
    state: dict[str, Any] = {"messages": model_response.result}
    if model_response.structured_response is not None:
        state["structured_response"] = model_response.structured_response

    for cmd in middleware_commands or []:
        if cmd.goto:
            raise NotImplementedError("Command goto is not yet supported...")
        if cmd.resume:
            raise NotImplementedError("Command resume is not yet supported...")
        if cmd.graph:
            raise NotImplementedError("Command graph is not yet supported...")

    commands: list[Command[Any]] = [Command(update=state)]
    commands.extend(middleware_commands or [])
    return commands
```

**作用：** 将模型响应和中间件 Command 组合为 LangGraph 节点的返回值。

**执行步骤：**
1. 构建基础状态更新：`{"messages": model_response.result}`
2. 如果有结构化响应，添加 `"structured_response"` 字段
3. 验证中间件 Command 不包含 `goto`/`resume`/`graph`（目前不支持）
4. 创建第一个 Command（模型响应的状态更新）
5. 追加中间件的 Command

**返回值结构：**
```python
[
    Command(update={"messages": [AIMessage(...)], "structured_response": {...}}),  # 模型响应
    Command(update={"custom_field": "value"}),  # 中间件 Command 1
    Command(update={"another_field": 123}),     # 中间件 Command 2
]
```

**为什么返回 Command 列表？** LangGraph 节点可以返回 `Command` 列表，每个 Command 都会通过 reducer 应用到状态上。这样模型响应和中间件的状态更新可以独立处理，互不干扰。

---

### 3.8 _chain_model_call_handlers（L222-L349）★★★洋葱模型核心★★★

这是整个中间件体系最精巧的部分。

```python
def _chain_model_call_handlers(
    handlers: Sequence[_ModelCallHandler[ContextT]],
) -> _ComposedModelCallHandler[ContextT] | None:
```

**输入：** 多个 `wrap_model_call` handler，按中间件注册顺序排列
**输出：** 一个组合后的 handler，或 `None`（如果没有 handler）

#### 空处理

```python
if not handlers:
    return None
```

#### _to_composed_result 辅助函数

```python
def _to_composed_result(
    result: ModelResponse | AIMessage | ExtendedModelResponse | _ComposedExtendedModelResponse,
    extra_commands: list[Command[Any]] | None = None,
) -> _ComposedExtendedModelResponse:
    commands: list[Command[Any]] = list(extra_commands or [])
    if isinstance(result, _ComposedExtendedModelResponse):
        commands.extend(result.commands)
        model_response = result.model_response
    elif isinstance(result, ExtendedModelResponse):
        model_response = result.model_response
        if result.command is not None:
            commands.append(result.command)
    else:
        model_response = _normalize_to_model_response(result)
    return _ComposedExtendedModelResponse(model_response=model_response, commands=commands)
```

**作用：** 将任意类型的 handler 返回值归一化为 `_ComposedExtendedModelResponse`。

**Command 收集逻辑：**
1. 先复制 `extra_commands`（来自内层的累积 Command）
2. 根据返回值类型提取 Command：
   - `_ComposedExtendedModelResponse`：合并其 commands 列表
   - `ExtendedModelResponse`：提取单个 command
   - 其他：无 Command

#### 单个 handler 的特殊处理

```python
if len(handlers) == 1:
    single_handler = handlers[0]
    def normalized_single(request, handler):
        return _to_composed_result(single_handler(request, handler))
    return normalized_single
```

**为什么单独处理？** 只有一个 handler 时不需要 compose_two 的闭包逻辑，简化代码路径。

#### compose_two — 两两组合的核心

```python
def compose_two(
    outer: _ModelCallHandler | _ComposedModelCallHandler,
    inner: _ModelCallHandler | _ComposedModelCallHandler,
) -> _ComposedModelCallHandler:
    def composed(request, handler):
        accumulated_commands: list[Command[Any]] = []

        def inner_handler(req):
            accumulated_commands.clear()  # 重试安全性
            inner_result = inner(req, handler)
            if isinstance(inner_result, _ComposedExtendedModelResponse):
                accumulated_commands.extend(inner_result.commands)
                return inner_result.model_response
            if isinstance(inner_result, ExtendedModelResponse):
                if inner_result.command is not None:
                    accumulated_commands.append(inner_result.command)
                return inner_result.model_response
            return _normalize_to_model_response(inner_result)

        outer_result = outer(request, inner_handler)
        return _to_composed_result(outer_result, extra_commands=accumulated_commands or None)

    return composed
```

**逐行解析：**

1. `accumulated_commands`：闭包变量，用于收集内层所有中间件的 Command

2. `inner_handler`：传给外层中间件的 handler 回调
   - 每次调用时先 `clear()`，确保重试时不会累积旧 Command
   - 调用内层 handler，提取 Command 到 `accumulated_commands`
   - 返回 `ModelResponse`（归一化后），外层中间件只看到 `ModelResponse`

3. `outer(request, inner_handler)`：调用外层中间件
   - 外层拿到 request 和 inner_handler
   - 外层可以修改 request、调用 inner_handler 多次（重试）、或不调用（短路）

4. `_to_composed_result`：将外层返回值归一化，合并内层的 `accumulated_commands`

**重试安全性设计：** `accumulated_commands.clear()` 在每次调用 `inner_handler` 时清空列表。这确保了如果外层中间件重试（多次调用 inner_handler），不会累积前一次的 Command。

#### 从右到左组合

```python
composed_handler = compose_two(handlers[-2], handlers[-1])
for h in reversed(handlers[:-2]):
    composed_handler = compose_two(h, composed_handler)
```

**执行顺序示例：** 假设有 3 个 handler：`[A, B, C]`

```
第1步：compose_two(B, C) → BC
第2步：compose_two(A, BC) → ABC

最终调用链：A(outer) → B(inner of A) → C(inner of B) → handler(最内层)
```

**A 是最外层，C 是最内层。** 这符合"第一个注册的中间件是最外层"的设计约定。

---

### 3.9 _chain_async_model_call_handlers（L351-L478）

与同步版本完全相同的逻辑，只是所有函数都是 `async def`，调用时使用 `await`。

**关键区别：**
```python
# 同步
inner_result = inner(req, handler)

# 异步
inner_result = await inner(req, handler)
```

---

### 3.10 _get_schema_type_hints（L480-L483）

```python
@functools.lru_cache(maxsize=100)
def _get_schema_type_hints(schema: type) -> dict[str, Any]:
    return get_type_hints(schema, include_extras=True)
```

**作用：** 获取类型的注解字典，带 LRU 缓存。

**为什么需要缓存？** `get_type_hints()` 涉及大量的字符串解析和导入操作，对同一个类型多次调用是浪费。缓存 100 个 schema 的类型提示足够覆盖绝大多数场景。

**`include_extras=True`：** 保留 `Annotated` 的元数据。例如 `Annotated[int, PrivateStateAttr]` 不会退化为 `int`，而是保留 `PrivateStateAttr` 元数据，供 `_extract_metadata` 使用。

---

### 3.11 _resolve_schemas / _resolve_schema（L485-L542）

```python
def _resolve_schemas(schemas: list[type]) -> tuple[type, type, type]:
    schema_hints = {schema: _get_schema_type_hints(schema) for schema in schemas}
    return (
        _resolve_schema(schema_hints, "StateSchema", None),       # 完整状态
        _resolve_schema(schema_hints, "InputSchema", "input"),    # 输入 schema
        _resolve_schema(schema_hints, "OutputSchema", "output"),  # 输出 schema
    )
```

**作用：** 将多个 TypedDict schema 合并为一个，同时生成输入/输出 schema。

**合并顺序：** 按列表顺序合并，后面的覆盖前面的同名字段。在 `create_agent` 中，顺序为：

```python
state_schemas = [*(m.state_schema for m in middleware), base_state]
```

中间件的 schema 在前，用户的 `base_state`（AgentState 或自定义 schema）在后。这意味着**用户定义的字段会覆盖中间件定义的同名字段**。

#### _resolve_schema

```python
def _resolve_schema(schema_hints, schema_name, omit_flag):
    all_annotations = {}
    for hints in schema_hints.values():
        for field_name, field_type in hints.items():
            should_omit = False
            if omit_flag:
                metadata = _extract_metadata(field_type)
                for meta in metadata:
                    if isinstance(meta, OmitFromSchema) and getattr(meta, omit_flag):
                        should_omit = True
                        break
            if not should_omit:
                all_annotations[field_name] = field_type
    return TypedDict(schema_name, all_annotations)
```

**执行步骤：**
1. 遍历所有 schema 的类型提示
2. 如果指定了 `omit_flag`（"input" 或 "output"），检查字段的 `OmitFromSchema` 注解
3. 如果字段标记为省略，跳过
4. 否则添加到合并后的 annotations
5. 使用 `TypedDict()` 动态创建合并后的 schema

**动态 TypedDict 创建：** `TypedDict(schema_name, all_annotations)` 等价于：

```python
@TypedDict
class StateSchema:
    messages: Annotated[list[AnyMessage], add_messages]
    jump_to: Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]
    ...
```

---

### 3.12 _extract_metadata（L544-L563）

```python
def _extract_metadata(type_: type) -> list[Any]:
    if get_origin(type_) in {Required, NotRequired}:
        inner_type = get_args(type_)[0]
        if get_origin(inner_type) is Annotated:
            return list(get_args(inner_type)[1:])
    elif get_origin(type_) is Annotated:
        return list(get_args(type_)[1:])
    return []
```

**作用：** 从类型注解中提取 `Annotated` 的元数据。

**处理三种格式：**
1. `Annotated[int, PrivateStateAttr]` → `[PrivateStateAttr]`
2. `Required[Annotated[int, PrivateStateAttr]]` → `[PrivateStateAttr]`
3. `NotRequired[Annotated[int, PrivateStateAttr]]` → `[PrivateStateAttr]`

**为什么需要处理 Required/NotRequired？** TypedDict 的字段可能被 `Required` 或 `NotRequired` 包裹，需要先解包才能访问 `Annotated`。

---

### 3.13 _get_can_jump_to（L565-L594）

```python
def _get_can_jump_to(middleware: AgentMiddleware, hook_name: str) -> list[JumpTo]:
    base_sync_method = getattr(AgentMiddleware, hook_name, None)
    base_async_method = getattr(AgentMiddleware, f"a{hook_name}", None)

    sync_method = getattr(middleware.__class__, hook_name, None)
    if (sync_method and sync_method is not base_sync_method
        and hasattr(sync_method, "__can_jump_to__")):
        return sync_method.__can_jump_to__

    async_method = getattr(middleware.__class__, f"a{hook_name}", None)
    if (async_method and async_method is not base_async_method
        and hasattr(async_method, "__can_jump_to__")):
        return async_method.__can_jump_to__

    return []
```

**执行步骤：**
1. 获取基类的同步/异步方法（用于比较）
2. 检查中间件的同步方法是否被覆盖（不是基类方法），且带有 `__can_jump_to__` 属性
3. 如果同步方法没有，检查异步方法
4. 都没有则返回空列表

**为什么检查 `is not base_sync_method`？** 因为 `AgentMiddleware` 的所有钩子都有默认实现（空方法或抛异常）。如果中间件没有覆盖某个钩子，`getattr` 会返回基类方法，此时不应该读取 `__can_jump_to__`。

---

### 3.14 _supports_provider_strategy（L596-L643）

```python
def _supports_provider_strategy(model, tools=None) -> bool:
    model_name = None
    if isinstance(model, str):
        model_name = model
    elif isinstance(model, BaseChatModel):
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or ""
        model_profile = model.profile
        if (model_profile is not None
            and model_profile.get("structured_output")
            and not (tools and "gemini" in model_name.lower() and "gemini-3" not in model_name.lower())):
            return True

    return any(part in model_name.lower() for part in FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT) if model_name else False
```

**判断逻辑：**
1. 如果是字符串模型名，跳到后备列表检查
2. 如果是 `BaseChatModel` 实例：
   a. 优先检查 `model.profile`（LangChain 的模型能力数据）
   b. 如果 profile 存在且 `structured_output=True`，返回 `True`
   c. **特殊处理 Gemini**：Gemini < 3 系列不支持同时使用工具和结构化输出
3. 如果没有 profile，使用后备列表检查模型名称

**Gemini 特殊处理的原因：** Gemini 2.x 系列在同时启用 tool_use 和 response_schema 时会报错。Gemini 3 系列修复了这个问题。

---

### 3.15 _handle_structured_output_error（L645-L687）

```python
def _handle_structured_output_error(exception, response_format) -> tuple[bool, str]:
    if not isinstance(response_format, ToolStrategy):
        return False, ""

    handle_errors = response_format.handle_errors

    if handle_errors is False:
        return False, ""
    if handle_errors is True:
        return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
    if isinstance(handle_errors, str):
        return True, handle_errors
    if isinstance(handle_errors, type):
        if issubclass(handle_errors, Exception) and isinstance(exception, handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""
    if isinstance(handle_errors, tuple):
        if any(isinstance(exception, exc_type) for exc_type in handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""
    return True, handle_errors(exception)
```

**返回值：** `(should_retry, error_message)`
- `should_retry=True`：应该重试，将 error_message 作为 ToolMessage 发送给模型
- `should_retry=False`：不应该重试，直接抛出异常

**handle_errors 的五种配置方式：**
1. `False`：不处理错误，直接抛出
2. `True`：总是重试，使用默认错误模板
3. `str`：总是重试，使用自定义错误消息
4. `type`（Exception 子类）：只对指定类型的异常重试
5. `tuple`（Exception 子类元组）：对元组中任一类型的异常重试
6. `Callable`：自定义判断函数，返回错误消息字符串

**注意：** 只有 `ToolStrategy` 支持错误处理。`ProviderStrategy` 的错误由模型自身处理。

---

### 3.16 _chain_tool_call_wrappers（L689-L735）

```python
def _chain_tool_call_wrappers(wrappers: Sequence[ToolCallWrapper]) -> ToolCallWrapper | None:
    if not wrappers:
        return None
    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(outer, inner):
        def composed(request, execute):
            def call_inner(req):
                return inner(req, execute)
            return outer(request, call_inner)
        return composed

    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)
    return result
```

**与 `_chain_model_call_handlers` 的区别：**

| 维度 | _chain_model_call_handlers | _chain_tool_call_wrappers |
|------|---------------------------|--------------------------|
| 返回类型 | `_ComposedExtendedModelResponse` | `ToolMessage \| Command` |
| Command 收集 | 需要（多层中间件可能各自产生 Command） | 不需要（工具调用没有 Command） |
| 归一化 | 需要（三种返回类型 → ModelResponse） | 不需要（只有一种返回类型） |

**更简洁的原因：** 工具调用的返回值只有 `ToolMessage | Command`，不需要复杂的归一化和 Command 收集逻辑。

---

### 3.17 _chain_async_tool_call_wrappers（L737-L797）

异步版本，逻辑与同步版本相同，只是使用 `async def` 和 `await`。

---

### 3.18 create_agent 函数（L799-L1689）★★★核心★★★

这是整个模块的入口函数，负责将所有配置组合为一个可执行的状态图。

#### 函数签名

```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    response_format: ResponseFormat | type | dict | None = None,
    state_schema: type[AgentState] | None = None,
    context_schema: type | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph:
```

**参数详解：**

| 参数 | 类型 | 作用 |
|------|------|------|
| `model` | `str \| BaseChatModel` | 语言模型。字符串会通过 `init_chat_model` 转换 |
| `tools` | `Sequence \| None` | 工具列表。支持 BaseTool、Callable、dict（内置工具） |
| `system_prompt` | `str \| SystemMessage \| None` | 系统提示词 |
| `middleware` | `Sequence[AgentMiddleware]` | 中间件列表，顺序决定洋葱模型的层级 |
| `response_format` | `ResponseFormat \| type \| dict \| None` | 结构化输出配置 |
| `state_schema` | `type[AgentState] \| None` | 自定义状态 schema |
| `context_schema` | `type \| None` | 运行时上下文 schema |
| `checkpointer` | `Checkpointer \| None` | 状态持久化（对话记忆） |
| `store` | `BaseStore \| None` | 跨线程数据存储 |
| `interrupt_before` | `list[str] \| None` | 在指定节点前中断 |
| `interrupt_after` | `list[str] \| None` | 在指定节点后中断 |
| `debug` | `bool` | 调试模式 |
| `name` | `str \| None` | 图名称（用于子图嵌套） |
| `cache` | `BaseCache \| None` | 图执行缓存 |
| `transformers` | `Sequence \| None` | 额外的流式转换器 |

---

#### 第一阶段：初始化（L900-L960）

```python
if isinstance(model, str):
    model = init_chat_model(model)

if system_prompt is not None:
    if isinstance(system_prompt, SystemMessage):
        system_message = system_prompt
    else:
        system_message = SystemMessage(content=system_prompt)

if tools is None:
    tools = []
```

**步骤：**
1. 字符串模型名 → `BaseChatModel` 实例
2. 字符串 system_prompt → `SystemMessage` 实例
3. None tools → 空列表

---

#### 第二阶段：结构化输出配置（L962-L990）

```python
if response_format is None:
    initial_response_format = None
elif isinstance(response_format, (ToolStrategy, ProviderStrategy, AutoStrategy)):
    initial_response_format = response_format
else:
    initial_response_format = AutoStrategy(schema=response_format)

tool_strategy_for_setup = None
if isinstance(initial_response_format, AutoStrategy):
    tool_strategy_for_setup = ToolStrategy(schema=initial_response_format.schema)
elif isinstance(initial_response_format, ToolStrategy):
    tool_strategy_for_setup = initial_response_format

structured_output_tools = {}
if tool_strategy_for_setup:
    for response_schema in tool_strategy_for_setup.schema_specs:
        structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
        structured_output_tools[structured_tool_info.tool.name] = structured_tool_info
```

**逻辑流程：**
1. 如果用户传入原始 schema（Pydantic 类或 dict），包装为 `AutoStrategy`
2. 如果是 `AutoStrategy`，预先创建 `ToolStrategy` 作为后备（因为需要提前创建结构化输出工具）
3. 为每个 schema spec 创建 `OutputToolBinding`，注册到 `structured_output_tools` 字典

**为什么 AutoStrategy 需要预先创建 ToolStrategy？** 因为结构化输出工具需要在 `create_agent` 时就添加到 ToolNode 中，而 AutoStrategy 的策略选择（Tool vs Provider）发生在运行时。所以预先用 ToolStrategy 创建工具，运行时如果模型支持 ProviderStrategy，则切换策略但保留工具。

---

#### 第三阶段：中间件工具收集（L992-L1000）

```python
middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]
```

**作用：** 收集所有中间件注册的工具。例如 `ShellToolMiddleware.tools = [shell_tool]`，`TodoMiddleware.tools = [todo_write_tool, todo_read_tool]`。

---

#### 第四阶段：wrap_tool_call 链组合（L1002-L1050）

```python
middleware_w_wrap_tool_call = [
    m for m in middleware
    if m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
    or m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
]

wrap_tool_call_wrapper = None
if middleware_w_wrap_tool_call:
    wrappers = [
        traceable(name=f"{m.name}.wrap_tool_call", process_inputs=_scrub_inputs)(
            m.wrap_tool_call
        )
        for m in middleware_w_wrap_tool_call
    ]
    wrap_tool_call_wrapper = _chain_tool_call_wrappers(wrappers)
```

**中间件检测逻辑：** `m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call` 检查中间件是否覆盖了基类的 `wrap_tool_call` 方法。如果覆盖了，说明该中间件实现了工具调用拦截。

**为什么用 `or` 检查同步和异步？** 因为中间件可能只实现了异步版本。如果用户用同步方式调用 Agent，需要抛出 `NotImplementedError`（由基类的同步方法抛出）。所以即使只有异步实现，也要把中间件加入列表。

**traceable 包装：** 每个中间件的 `wrap_tool_call` 都被 `langsmith.traceable` 包装，用于 LangSmith 追踪。`process_inputs=_scrub_inputs` 确保追踪数据中不包含敏感信息。

---

#### 第五阶段：awrap_tool_call 链组合（L1052-L1070）

与第四阶段相同的逻辑，但组合异步版本。

---

#### 第六阶段：ToolNode 创建（L1072-L1100）

```python
built_in_tools = [t for t in tools if isinstance(t, dict)]
regular_tools = [t for t in tools if not isinstance(t, dict)]
available_tools = middleware_tools + regular_tools

tool_node = (
    ToolNode(
        tools=available_tools,
        wrap_tool_call=wrap_tool_call_wrapper,
        awrap_tool_call=awrap_tool_call_wrapper,
    )
    if available_tools or wrap_tool_call_wrapper or awrap_tool_call_wrapper
    else None
)
```

**工具分类：**
- `built_in_tools`：dict 格式的内置工具（如 OpenAI 的 `code_interpreter`），不需要客户端执行
- `regular_tools`：BaseTool 或 Callable 格式的普通工具，需要客户端执行
- `available_tools`：中间件工具 + 普通工具，传给 ToolNode

**ToolNode 创建条件：** 有可用工具，或者有 wrap_tool_call 中间件（可能处理动态工具），才创建 ToolNode。否则 Agent 只是一个纯对话模型，不需要工具节点。

```python
if tool_node:
    default_tools = list(tool_node.tools_by_name.values()) + built_in_tools
else:
    default_tools = list(built_in_tools)
```

**default_tools：** 传给 `ModelRequest` 的默认工具列表。ToolNode 会将 Callable 转换为 BaseTool，所以这里用 `tools_by_name.values()` 获取转换后的工具。

---

#### 第七阶段：中间件验证与分类（L1102-L1150）

```python
if len({m.name for m in middleware}) != len(middleware):
    msg = "Please remove duplicate middleware instances."
    raise AssertionError(msg)
```

**验证：** 检查中间件名称是否重复。重复名称会导致图节点名称冲突。

```python
middleware_w_before_agent = [
    m for m in middleware
    if m.__class__.before_agent is not AgentMiddleware.before_agent
    or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
]
middleware_w_before_model = [...]
middleware_w_after_model = [...]
middleware_w_after_agent = [...]
middleware_w_wrap_model_call = [...]
middleware_w_awrap_model_call = [...]
```

**分类：** 将中间件按实现的钩子类型分组。一个中间件可以出现在多个组中（如同时实现 `before_model` 和 `wrap_model_call`）。

---

#### 第八阶段：wrap_model_call 链组合（L1152-L1180）

```python
wrap_model_call_handler = None
if middleware_w_wrap_model_call:
    sync_handlers = [
        traceable(name=f"{m.name}.wrap_model_call", process_inputs=_scrub_inputs)(
            m.wrap_model_call
        )
        for m in middleware_w_wrap_model_call
    ]
    wrap_model_call_handler = _chain_model_call_handlers(sync_handlers)
```

与 wrap_tool_call 相同的模式：收集 → traceable 包装 → 链式组合。

---

#### 第九阶段：Schema 合并（L1182-L1195）

```python
base_state = state_schema if state_schema is not None else AgentState
state_schemas = [*(m.state_schema for m in middleware), base_state]
resolved_state_schema, input_schema, output_schema = _resolve_schemas(state_schemas)
```

**合并顺序：** 中间件 schema 在前，base_state 在后。后者的字段覆盖前者的同名字段。

---

#### 第十阶段：StateGraph 创建（L1197-L1205）

```python
graph = StateGraph(
    state_schema=resolved_state_schema,
    input_schema=input_schema,
    output_schema=output_schema,
    context_schema=context_schema,
)
```

**三个 schema 的区别：**
- `state_schema`：内部状态，包含所有字段（包括私有的 jump_to）
- `input_schema`：用户输入，排除了 OmitFromInput 标记的字段
- `output_schema`：用户输出，排除了 OmitFromOutput 标记的字段

---

#### 第十一阶段：_handle_model_output（L1207-L1320）

```python
def _handle_model_output(output, effective_response_format):
```

**作用：** 处理模型输出，提取结构化响应。

**三种情况：**

1. **ProviderStrategy**：模型原生结构化输出
   ```python
   if isinstance(effective_response_format, ProviderStrategy):
       if not output.tool_calls:
           # 模型直接返回结构化数据（非工具调用方式）
           structured_response = provider_strategy_binding.parse(output)
           return {"messages": [output], "structured_response": structured_response}
       return {"messages": [output]}
   ```

2. **ToolStrategy**：通过工具调用实现结构化输出
   ```python
   if isinstance(effective_response_format, ToolStrategy) and output.tool_calls:
       structured_tool_calls = [tc for tc in output.tool_calls if tc["name"] in structured_output_tools]
       if len(structured_tool_calls) > 1:
           # 多个结构化输出工具被调用 → 错误
           exception = MultipleStructuredOutputsError(tool_names, output)
           should_retry, error_message = _handle_structured_output_error(exception, ...)
           if not should_retry:
               raise exception
           # 重试：发送错误消息给模型
           tool_messages = [ToolMessage(content=error_message, tool_call_id=tc["id"]) for tc in structured_tool_calls]
           return {"messages": [output, *tool_messages]}

       # 单个结构化输出工具
       tool_call = structured_tool_calls[0]
       try:
           structured_response = structured_tool_binding.parse(tool_call["args"])
           return {"messages": [output, ToolMessage(...)], "structured_response": structured_response}
       except Exception as exc:
           # 验证失败 → 可能重试
           exception = StructuredOutputValidationError(tool_call["name"], exc, output)
           should_retry, error_message = _handle_structured_output_error(exception, ...)
           if not should_retry:
               raise exception from exc
           return {"messages": [output, ToolMessage(content=error_message, ...)]}
   ```

3. **无结构化输出**：直接返回消息
   ```python
   return {"messages": [output]}
   ```

---

#### 第十二阶段：_get_bound_model（L1322-L1450）

```python
def _get_bound_model(request):
```

**作用：** 根据请求配置绑定工具和结构化输出到模型。

**执行步骤：**

1. **验证工具**：检查中间件动态添加的工具是否有对应的执行器
   ```python
   if not has_wrap_tool_call:
       unknown_tool_names = [t.name for t in request.tools
                            if isinstance(t, BaseTool) and t.name not in available_tools_by_name]
       if unknown_tool_names:
           raise ValueError(DYNAMIC_TOOL_ERROR_TEMPLATE.format(...))
   ```
   如果没有 `wrap_tool_call` 中间件，动态添加的工具无法执行，抛出错误。

2. **归一化 response_format**：原始 schema → AutoStrategy
   ```python
   if response_format is not None and not isinstance(response_format, (AutoStrategy, ToolStrategy, ProviderStrategy)):
       response_format = AutoStrategy(schema=response_format)
   ```

3. **AutoStrategy 策略选择**：
   ```python
   if isinstance(response_format, AutoStrategy):
       if _supports_provider_strategy(request.model, tools=request.tools):
           effective_response_format = ProviderStrategy(schema=response_format.schema)
       elif ...:
           effective_response_format = tool_strategy_for_setup  # 复用已有的
       else:
           effective_response_format = ToolStrategy(schema=response_format.schema)
   ```

4. **构建最终工具列表**：普通工具 + 结构化输出工具
   ```python
   final_tools = list(request.tools)
   if isinstance(effective_response_format, ToolStrategy):
       final_tools.extend([info.tool for info in structured_output_tools.values()])
   ```

5. **绑定模型**：
   ```python
   if isinstance(effective_response_format, ProviderStrategy):
       kwargs = effective_response_format.to_model_kwargs()
       return request.model.bind_tools(final_tools, strict=True, **kwargs, **request.model_settings), effective_response_format

   if isinstance(effective_response_format, ToolStrategy):
       tool_choice = "any" if structured_output_tools else request.tool_choice
       return request.model.bind_tools(final_tools, tool_choice=tool_choice, **request.model_settings), effective_response_format

   if final_tools:
       return request.model.bind_tools(final_tools, tool_choice=request.tool_choice, **request.model_settings), None
   return request.model.bind(**request.model_settings), None
   ```

   **三种绑定方式：**
   - ProviderStrategy：`bind_tools(strict=True, **kwargs)` — kwargs 包含 response_format 参数
   - ToolStrategy：`bind_tools(tool_choice="any")` — 强制模型调用工具（因为需要结构化输出）
   - 无结构化输出：`bind_tools()` 或 `bind()`（无工具时）

---

#### 第十三阶段：_execute_model_sync / _execute_model_async（L1452-L1500）

```python
def _execute_model_sync(request):
    model_, effective_response_format = _get_bound_model(request)
    messages = request.messages
    if request.system_message:
        messages = [request.system_message, *messages]
    output = model_.invoke(messages)
    if name:
        output.name = name
    handled_output = _handle_model_output(output, effective_response_format)
    return ModelResponse(
        result=handled_output["messages"],
        structured_response=handled_output.get("structured_response"),
    )
```

**执行步骤：**
1. 获取绑定后的模型和有效 response_format
2. 拼接 system_message + messages
3. 调用模型（invoke / ainvoke）
4. 设置输出名称（如果指定了 agent name）
5. 处理模型输出（提取结构化响应）
6. 返回 ModelResponse

**这是洋葱模型的最内层——真正的 LLM 调用。**

---

#### 第十四阶段：model_node / amodel_node（L1502-L1550）

```python
def model_node(state, runtime):
    request = ModelRequest(
        model=model,
        tools=default_tools,
        system_message=system_message,
        response_format=initial_response_format,
        messages=state["messages"],
        tool_choice=None,
        state=state,
        runtime=runtime,
    )

    if wrap_model_call_handler is None:
        model_response = _execute_model_sync(request)
        return _build_commands(model_response)

    result = wrap_model_call_handler(request, _execute_model_sync)
    return _build_commands(result.model_response, result.commands)
```

**执行步骤：**
1. 构造 `ModelRequest`（使用默认工具、初始 response_format）
2. 如果没有 wrap_model_call 中间件，直接执行模型
3. 如果有，通过洋葱模型链执行，收集 Command
4. 返回 Command 列表

**关键设计：** `_execute_model_sync` 作为最内层 handler 传给洋葱模型链。外层中间件可以修改 request、重试、短路。

---

#### 第十五阶段：图节点添加（L1552-L1620）

```python
graph.add_node("model", RunnableCallable(model_node, amodel_node, trace=False))

if tool_node is not None:
    graph.add_node("tools", tool_node)

for m in middleware:
    if m.__class__.before_agent is not AgentMiddleware.before_agent or ...:
        sync_before_agent = m.before_agent if ... else None
        async_before_agent = m.abefore_agent if ... else None
        before_agent_node = RunnableCallable(sync_before_agent, async_before_agent, trace=False)
        graph.add_node(f"{m.name}.before_agent", before_agent_node, input_schema=resolved_state_schema)
    # 同理添加 before_model、after_model、after_agent 节点
```

**节点命名规则：** `{MiddlewareName}.{hook_name}`，如 `"ShellToolMiddleware.before_agent"`。

**RunnableCallable：** LangGraph 的节点包装器，支持同步/异步双模式。如果中间件只实现了异步版本，sync 参数传 None。

**为什么 `trace=False`？** 因为模型调用和中间件调用已经有自己的 `traceable` 包装，不需要 RunnableCallable 再添加一层追踪。

---

#### 第十六阶段：边连接逻辑（L1622-L1750）★★★图构建核心★★★

这是最复杂的部分，决定了 Agent 的执行流程。

##### 关键节点名称

```python
entry_node = before_agent[0] 或 before_model[0] 或 "model"        # 入口
loop_entry_node = before_model[0] 或 "model"                       # 循环入口
loop_exit_node = after_model[0] 或 "model"                         # 循环出口
exit_node = after_agent[-1] 或 END                                  # 出口
```

##### 入口边

```python
graph.add_edge(START, entry_node)
```

##### tools → model 条件边

```python
if tool_node is not None:
    tools_to_model_destinations = [loop_entry_node]
    if any(tool.return_direct for tool in tool_node.tools_by_name.values()) or structured_output_tools:
        tools_to_model_destinations.append(exit_node)

    graph.add_conditional_edges("tools", _make_tools_to_model_edge(...), tools_to_model_destinations)
```

**条件：** 工具执行后，默认回到模型。但如果所有工具都是 `return_direct` 或有结构化输出工具被执行，则可以跳到出口。

##### model → tools 条件边

```python
model_to_tools_destinations = ["tools", exit_node]
if response_format or loop_exit_node != "model":
    model_to_tools_destinations.append(loop_entry_node)

graph.add_conditional_edges(loop_exit_node, _make_model_to_tools_edge(...), model_to_tools_destinations)
```

**条件：** 模型输出后，根据是否有 tool_calls 决定去 tools 还是 exit。如果配置了 response_format 或有 after_model 钩子，还可以跳回 loop_entry（重试结构化输出或处理人工注入的消息）。

##### before_agent 链

```python
if middleware_w_before_agent:
    for m1, m2 in itertools.pairwise(middleware_w_before_agent):
        _add_middleware_edge(graph, name=f"{m1.name}.before_agent",
                            default_destination=f"{m2.name}.before_agent", ...)
    _add_middleware_edge(graph, name=f"{middleware_w_before_agent[-1].name}.before_agent",
                        default_destination=loop_entry_node, ...)
```

**连接方式：** 链式连接，每个 before_agent 节点连接到下一个。最后一个连接到 loop_entry_node。

##### before_model 链

```python
if middleware_w_before_model:
    for m1, m2 in itertools.pairwise(middleware_w_before_model):
        _add_middleware_edge(graph, name=f"{m1.name}.before_model",
                            default_destination=f"{m2.name}.before_model", ...)
    _add_middleware_edge(graph, name=f"{middleware_w_before_model[-1].name}.before_model",
                        default_destination="model", ...)
```

**连接方式：** 链式连接，最后一个连接到 "model" 节点。

##### after_model 链

```python
if middleware_w_after_model:
    graph.add_edge("model", f"{middleware_w_after_model[-1].name}.after_model")
    for idx in range(len(middleware_w_after_model) - 1, 0, -1):
        m1 = middleware_w_after_model[idx]
        m2 = middleware_w_after_model[idx - 1]
        _add_middleware_edge(graph, name=f"{m1.name}.after_model",
                            default_destination=f"{m2.name}.after_model", ...)
```

**连接方式：** model → after_model[-1] → after_model[-2] → ... → after_model[0]。注意顺序是**反的**——从最后一个 after_model 开始，依次连接到前一个。

**为什么反向？** 因为 after_model 钩子的执行顺序应该与 before_model 相反（洋葱模型的外层返回）。如果 before_model 是 A → B → C，那么 after_model 应该是 C → B → A。

##### after_agent 链

```python
if middleware_w_after_agent:
    for idx in range(len(middleware_w_after_agent) - 1, 0, -1):
        m1 = middleware_w_after_agent[idx]
        m2 = middleware_w_after_agent[idx - 1]
        _add_middleware_edge(graph, name=f"{m1.name}.after_agent",
                            default_destination=f"{m2.name}.after_agent", ...)
    _add_middleware_edge(graph, name=f"{middleware_w_after_agent[0].name}.after_agent",
                        default_destination=END, ...)
```

**连接方式：** 与 after_model 类似的反向链，最后一个连接到 END。

---

#### 第十七阶段：图编译与配置（L1752-L1689）

```python
config = {"recursion_limit": 9_999}
config["metadata"] = {"ls_integration": "langchain_create_agent"}
if name:
    config["metadata"]["lc_agent_name"] = name

middleware_transformers = [t for m in middleware for t in getattr(m, "transformers", ())]

return graph.compile(
    checkpointer=checkpointer,
    store=store,
    interrupt_before=interrupt_before,
    interrupt_after=interrupt_after,
    debug=debug,
    name=name,
    cache=cache,
    transformers=[ToolCallTransformer, *middleware_transformers, *(transformers or ())],
).with_config(config)
```

**recursion_limit = 9999：** 防止无限循环。正常 Agent 不会达到这个限制，但如果中间件配置错误导致死循环，这个限制会触发异常。

**transformers 顺序：** `ToolCallTransformer`（LangGraph 内置）→ 中间件 transformers → 用户 transformers。这确保工具调用的流式输出先被处理，然后是中间件的自定义转换。

---

### 3.19 _resolve_jump（L1691-L1703）

```python
def _resolve_jump(jump_to, *, model_destination, end_destination):
    if jump_to == "model":
        return model_destination
    if jump_to == "end":
        return end_destination
    if jump_to == "tools":
        return "tools"
    return None
```

**作用：** 将逻辑跳转目标（"model"/"end"/"tools"）映射为图中的实际节点名称。

**为什么需要映射？** "model" 在图中可能不是 "model" 节点，而是 `loop_entry_node`（如果有 before_model 中间件，则是 `"X.before_model"`）。"end" 可能是 `exit_node`（如果有 after_agent 中间件，则是 `"X.after_agent"`）。

---

### 3.20 _fetch_last_ai_and_tool_messages（L1705-L1726）

```python
def _fetch_last_ai_and_tool_messages(messages):
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai_message = cast("AIMessage", messages[i])
            tool_messages = [m for m in messages[i + 1:] if isinstance(m, ToolMessage)]
            return last_ai_message, tool_messages
    return None, []
```

**作用：** 从消息列表中找到最后一条 AIMessage 和它之后的所有 ToolMessage。

**为什么从后往前搜索？** 因为消息列表是按时间顺序排列的，最后一条 AIMessage 一定在列表末尾附近。从后往前搜索效率更高。

**为什么返回 AIMessage 之后的 ToolMessage？** 因为条件边需要判断：
1. AIMessage 有没有 tool_calls
2. 这些 tool_calls 是否已经有对应的 ToolMessage（已执行）
3. 是否还有 pending 的 tool_calls（需要执行）

---

### 3.21 _make_model_to_tools_edge（L1728-L1790）★★★核心路由逻辑★★★

```python
def _make_model_to_tools_edge(*, model_destination, structured_output_tools, end_destination):
    def model_to_tools(state):
        # 1. jump_to 优先
        if jump_to := state.get("jump_to"):
            return _resolve_jump(jump_to, model_destination=model_destination, end_destination=end_destination)

        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

        # 2. 没有 AIMessage → 结束
        if last_ai_message is None:
            return end_destination

        tool_message_ids = [m.tool_call_id for m in tool_messages]

        # 3. 没有 tool_calls → 结束（经典退出条件）
        if len(last_ai_message.tool_calls) == 0:
            return end_destination

        # 4. 有 pending 的 tool_calls → 去工具节点
        pending_tool_calls = [
            c for c in last_ai_message.tool_calls
            if c["id"] not in tool_message_ids and c["name"] not in structured_output_tools
        ]
        if pending_tool_calls:
            return [Send("tools", [tool_call]) for tool_call in pending_tool_calls]

        # 5. 有结构化响应 → 结束
        if "structured_response" in state:
            return end_destination

        # 6. 有 tool_calls 但没有 pending → 回模型（人工注入了 ToolMessage）
        return model_destination

    return model_to_tools
```

**六步路由逻辑详解：**

**第1步：jump_to 优先级最高。** 中间件通过 `before_model` 或 `after_model` 返回 `{"jump_to": "end"}` 来强制跳转。这覆盖所有其他逻辑。

**第2步：没有 AIMessage。** 可能是消息被清空了（如 context_editing 中间件清除了历史）。此时应该退出循环。

**第3步：经典退出条件。** 模型没有调用任何工具，说明它认为对话已经完成。这是最常见的退出路径。

**第4步：并行工具执行。** `Send("tools", [tool_call])` 让 LangGraph 并行执行多个工具调用。注意排除了结构化输出工具（它们由 _handle_model_output 处理，不需要真正执行）。

**第5步：结构化响应已生成。** 如果 state 中有 `structured_response`，说明结构化输出已经成功解析，可以退出。

**第6步：兜底回模型。** AIMessage 有 tool_calls，但所有 tool_calls 都已有对应的 ToolMessage（不是 pending 的），且不是结构化输出工具。这种情况通常发生在人工审批中间件注入了 ToolMessage（如拒绝工具调用后添加了错误消息），需要让模型重新处理。

---

### 3.22 _make_model_to_model_edge（L1792-L1820）

```python
def _make_model_to_model_edge(*, model_destination, end_destination):
    def model_to_model(state):
        if jump_to := state.get("jump_to"):
            return _resolve_jump(jump_to, model_destination=model_destination, end_destination=end_destination)
        if "structured_response" in state:
            return end_destination
        return model_destination
    return model_to_model
```

**使用场景：** 当没有工具节点但有结构化输出时，模型可能需要多次调用才能生成正确的结构化输出。

**路由逻辑：**
1. jump_to 优先
2. 有结构化响应 → 结束
3. 否则 → 回模型重试

---

### 3.23 _make_tools_to_model_edge（L1822-L1860）

```python
def _make_tools_to_model_edge(*, tool_node, model_destination, structured_output_tools, end_destination):
    def tools_to_model(state):
        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

        # 1. 没有 AIMessage → 回模型
        if last_ai_message is None:
            return model_destination

        # 2. 所有客户端工具都是 return_direct → 结束
        client_side_tool_calls = [c for c in last_ai_message.tool_calls if c["name"] in tool_node.tools_by_name]
        if client_side_tool_calls and all(
            tool_node.tools_by_name[c["name"]].return_direct for c in client_side_tool_calls
        ):
            return end_destination

        # 3. 结构化输出工具被执行 → 结束
        if any(t.name in structured_output_tools for t in tool_messages):
            return end_destination

        # 4. 默认 → 回模型
        return model_destination

    return tools_to_model
```

**四步路由逻辑：**

**第1步：** 没有 AIMessage 时回模型重新生成。

**第2步：return_direct 退出。** 如果工具标记了 `return_direct=True`，工具的输出直接作为 Agent 的最终响应，不需要再让模型处理。例如搜索工具可能直接返回结果。

**第3步：结构化输出退出。** 结构化输出工具的执行结果不需要再让模型处理。

**第4步：默认回模型。** 工具执行完成后，让模型处理工具结果并决定下一步。

---

### 3.24 _add_middleware_edge（L1862-L1889）

```python
def _add_middleware_edge(graph, *, name, default_destination, model_destination, end_destination, can_jump_to):
    if can_jump_to:
        def jump_edge(state):
            return _resolve_jump(state.get("jump_to"), model_destination=model_destination, end_destination=end_destination) or default_destination

        destinations = [default_destination]
        if "end" in can_jump_to:
            destinations.append(end_destination)
        if "tools" in can_jump_to:
            destinations.append("tools")
        if "model" in can_jump_to and name != model_destination:
            destinations.append(model_destination)

        graph.add_conditional_edges(name, RunnableCallable(jump_edge, trace=False), destinations)
    else:
        graph.add_edge(name, default_destination)
```

**两种边：**

1. **无条件边**（`can_jump_to` 为空）：直接连接到 default_destination
2. **条件边**（`can_jump_to` 非空）：根据 `jump_to` 状态决定跳转目标

**条件边逻辑：**
- 如果 state 中有 `jump_to` 且在 `can_jump_to` 列表中，跳转到对应目标
- 否则走 default_destination

**destinations 列表：** LangGraph 要求条件边的所有可能目标必须预先声明。这里根据 `can_jump_to` 配置动态构建目标列表。

**`name != model_destination` 检查：** 如果中间件节点本身就是 loop_entry_node（即 model_destination），不需要再添加跳转到自己的边，避免自环。

---

## 四、完整执行流程图

假设配置为：
```python
create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[ShellToolMiddleware(), HumanInTheLoopMiddleware()],
)
```

生成的状态图：

```
START
  │
  ▼
ShellToolMiddleware.before_agent  ←── 初始化 Shell 环境
  │
  ▼
HumanInTheLoopMiddleware.before_model  ←── 每次模型调用前
  │
  ▼
┌─────────────────────────────────────────────────────┐
│                    model 节点                        │
│  ┌──────────────────────────────────────────────┐   │
│  │ 洋葱模型：                                     │   │
│  │ HITL.wrap_model_call → Shell.wrap_model_call │   │
│  │ → _execute_model_sync (真正的 LLM 调用)       │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
  │
  ▼
HumanInTheLoopMiddleware.after_model  ←── 检查是否需要审批
  │
  ├── 有 tool_calls 且需要审批 → interrupt() 暂停
  │
  ├── 有 tool_calls 且不需要审批 ──┐
  │                                │
  ├── 无 tool_calls ───────────────┤
  │                                │
  │                    ┌───────────┘
  │                    │
  │         ┌──────────┤
  │         │          │
  │         ▼          ▼
  │      tools      ShellToolMiddleware.after_agent → END
  │     (并行)      (清理 Shell 环境)
  │         │
  │         ▼
  │    tools_to_model 路由
  │         │
  │         ▼
  └──→ HumanInTheLoopMiddleware.before_model (循环)
```

---

## 五、核心设计模式总结

### 5.1 工厂模式

`create_agent()` 是典型的工厂函数，根据配置动态创建不同结构的 StateGraph。

### 5.2 洋葱模型（Middleware Pipeline）

`_chain_model_call_handlers` 和 `_chain_tool_call_wrappers` 实现了函数式组合，将多个中间件组合为一个调用链。

### 5.3 状态机模式

使用 LangGraph 的 StateGraph 建模 Agent 循环，节点是处理逻辑，边是路由决策。

### 5.4 策略模式

结构化输出的三种策略（Tool/Provider/Auto）通过 `_supports_provider_strategy` 在运行时选择。

### 5.5 闭包组合

`compose_two` 使用闭包捕获内层的 `accumulated_commands`，实现了跨层级的 Command 收集。

### 5.6 延迟绑定

`_get_bound_model` 在运行时根据 ModelRequest 的内容动态绑定工具和结构化输出，而非编译时固定。
