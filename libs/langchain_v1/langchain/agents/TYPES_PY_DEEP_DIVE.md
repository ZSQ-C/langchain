# types.py 深度详解

> 文件路径：`e:\AI\GitHub\langchain\libs\langchain_v1\langchain\agents\middleware\types.py`
> 总行数：2065 行

---

## 一、文件定位

`types.py` 是 Agent 中间件体系的**类型定义与核心抽象层**。它定义了：

1. 中间件基类 `AgentMiddleware`（所有中间件的父类）
2. 模型请求/响应数据结构（`ModelRequest`、`ModelResponse`、`ExtendedModelResponse`）
3. Agent 状态 schema（`AgentState`）
4. 装饰器工厂函数（`before_model`、`after_model`、`before_agent`、`after_agent`、`wrap_model_call`、`wrap_tool_call`、`dynamic_prompt`）
5. 辅助类型（`JumpTo`、`OmitFromSchema`、`hook_config`）

**一句话总结：** 这个文件定义了中间件"能做什么"和"怎么做"的全部契约。

---

## 二、代码结构总览

```
types.py
├── 导入区（L1-L37）
├── __all__ 导出列表（L39-L61）
├── JumpTo 类型别名（L63）
├── ResponseT 类型变量（L66）
├── _ModelRequestOverrides TypedDict（L68-L76）
├── ModelRequest 数据类（L78-L282）
├── ModelResponse 数据类（L284-L297）
├── ExtendedModelResponse 数据类（L299-L326）
├── ModelCallResult 类型别名（L328-L340）
├── OmitFromSchema 数据类（L342-L364）
├── AgentState TypedDict（L376-L381）
├── _InputAgentState / _OutputAgentState（L383-L392）
├── StateT / StateT_co / StateT_contra 类型变量（L394-L397）
├── _DefaultAgentState（L399-L401）
├── AgentMiddleware 基类（L403-L797）★核心★
├── Protocol 类定义（L799-L840）
├── hook_config 装饰器（L842-L910）
├── before_model 装饰器工厂（L912-L1107）
├── after_model 装饰器工厂（L1109-L1296）
├── before_agent 装饰器工厂（L1298-L1493）
├── after_agent 装饰器工厂（L1495-L1689）
├── dynamic_prompt 装饰器工厂（L1691-L1790）
├── wrap_model_call 装饰器工厂（L1792-L1930）
└── wrap_tool_call 装饰器工厂（L1932-L2065）
```

---

## 三、逐段深度详解

---

### 3.1 导入区（L1-L37）

```python
from __future__ import annotations
```

**作用：** 启用 PEP 563 延迟注解求值。这意味着所有类型注解在运行时不会立即求值，而是存储为字符串。这样做的好处是：
- 允许在类型注解中引用尚未定义的类（前向引用）
- 减少导入时的循环依赖风险
- 提升模块导入速度

```python
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from inspect import iscoroutinefunction
```

**关键导入解析：**
- `replace`：来自 dataclasses，用于创建数据类的不可变副本。`ModelRequest.override()` 方法就是基于它实现的
- `iscoroutinefunction`：用于检测函数是否是协程函数。装饰器工厂用它来判断用户传入的函数是同步还是异步，从而生成对应的中间件

```python
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolCallWrapper
```

**关键导入解析：**
- `add_messages`：LangGraph 的消息 reducer 函数，当多个节点返回 messages 更新时，它会将新消息追加到已有列表中，而非覆盖
- `EphemeralValue`：LangGraph 的临时值 channel，不会持久化到 checkpoint。`jump_to` 字段使用它，因为跳转指令是一次性的
- `ToolCallRequest`：LangGraph 的工具调用请求对象，包含 `tool_call` 字典、`tool` 实例、`state` 和 `runtime`
- `ToolCallWrapper`：工具调用包装器类型，签名为 `(request, handler) -> result`

```python
from typing_extensions import NotRequired, Required, TypedDict, TypeVar, Unpack
```

**关键导入解析：**
- `Unpack`：用于在 `override()` 方法中将 TypedDict 的键作为 **kwargs 的类型提示。`_ModelRequestOverrides` 定义了 override 支持的参数，`Unpack` 让 IDE 能正确提示这些参数
- `NotRequired` / `Required`：TypedDict 字段的必选/可选标记

---

### 3.2 __all__ 导出列表（L39-L61）

```python
__all__ = [
    "AgentMiddleware",
    "AgentState",
    "ContextT",
    "ExtendedModelResponse",
    "ModelCallResult",
    "ModelRequest",
    "ModelResponse",
    "OmitFromSchema",
    "ResponseT",
    "StateT_co",
    "ToolCallRequest",
    "ToolCallWrapper",
    "after_agent",
    "after_model",
    "before_agent",
    "before_model",
    "dynamic_prompt",
    "hook_config",
    "wrap_model_call",
    "wrap_tool_call",
]
```

**设计意图：** `__all__` 定义了模块的公共 API。当其他模块使用 `from langchain.agents.middleware.types import *` 时，只有列表中的名称会被导入。这确保了内部实现细节（如 `_DefaultAgentState`、`_InputAgentState`）不会泄漏到公共 API 中。

---

### 3.3 JumpTo 类型别名（L63）

```python
JumpTo = Literal["tools", "model", "end"]
```

**作用：** 定义中间件可以跳转到的目标节点。这是一个受限的字符串字面量类型，只能是三个值之一：
- `"tools"`：跳转到工具执行节点
- `"model"`：跳转回模型节点（重新调用 LLM）
- `"end"`：跳转到结束节点（退出 Agent 循环）

**使用场景：** 中间件通过 `before_model` 或 `after_model` 钩子返回 `{"jump_to": "end"}` 来控制 Agent 流程。例如预算超限时跳转到 end，人工审批后跳转到 tools。

---

### 3.4 ResponseT 类型变量（L66）

```python
ResponseT = TypeVar("ResponseT", default=Any)
```

**作用：** 结构化响应的类型变量。当用户指定 `response_format` 时，`ResponseT` 就是结构化输出的类型。默认为 `Any`，表示不限制类型。

**泛型传播链：**
```
ResponseT → AgentState[ResponseT] → AgentMiddleware[StateT, ContextT, ResponseT]
         → ModelResponse[ResponseT] → ExtendedModelResponse[ResponseT]
```

---

### 3.5 _ModelRequestOverrides TypedDict（L68-L76）

```python
class _ModelRequestOverrides(TypedDict, total=False):
    model: BaseChatModel
    system_message: SystemMessage | None
    messages: list[AnyMessage]
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    model_settings: dict[str, Any]
    state: AgentState[Any]
```

**作用：** 定义 `ModelRequest.override()` 方法接受的参数类型。`total=False` 表示所有字段都是可选的。

**为什么用 TypedDict + Unpack？** 这是 Python 类型系统的技巧。`override()` 的签名是：

```python
def override(self, **overrides: Unpack[_ModelRequestOverrides]) -> ModelRequest:
```

`Unpack[_ModelRequestOverrides]` 让 IDE 能在用户调用 `override(model=..., tools=...)` 时提供精确的类型提示和自动补全。如果不这样做，`**overrides` 的类型只能是 `Any`。

---

### 3.6 ModelRequest 数据类（L78-L282）★核心数据结构★

```python
@dataclass(init=False)
class ModelRequest(Generic[ContextT]):
```

**为什么 `init=False`？** 因为 `ModelRequest` 需要自定义 `__init__` 方法来处理 `system_prompt` 和 `system_message` 的兼容逻辑。如果使用默认的 dataclass `__init__`，就无法添加这种自定义逻辑。

#### 字段定义

```python
model: BaseChatModel           # 要调用的语言模型
messages: list[AnyMessage]     # 对话消息列表（不含 system_message）
system_message: SystemMessage | None  # 系统提示词
tool_choice: Any | None        # 工具选择策略（如 "auto"、"any"、指定工具名）
tools: list[BaseTool | dict]   # 可用工具列表
response_format: ResponseFormat | None  # 结构化输出配置
state: AgentState[Any]         # 当前 Agent 状态
runtime: Runtime[ContextT]     # 运行时上下文
model_settings: dict[str, Any] = field(default_factory=dict)  # 模型额外设置
```

**字段设计要点：**
- `messages` 不包含 `system_message`，两者分开存储。这是因为中间件可能需要独立修改系统提示词或对话消息
- `tools` 同时支持 `BaseTool` 和 `dict`。dict 格式用于"内置工具"（如 OpenAI 的 `code_interpreter`），它们不需要客户端执行
- `model_settings` 使用 `field(default_factory=dict)` 而非 `= {}`，避免可变默认参数的陷阱

#### __init__ 方法（L108-L154）

```python
def __init__(
    self,
    *,
    model: BaseChatModel,
    messages: list[AnyMessage],
    system_message: SystemMessage | None = None,
    system_prompt: str | None = None,  # 已废弃，兼容旧 API
    ...
) -> None:
```

**关键逻辑：**

```python
if system_prompt is not None and system_message is not None:
    msg = "Cannot specify both system_prompt and system_message"
    raise ValueError(msg)

if system_prompt is not None:
    system_message = SystemMessage(content=system_prompt)
```

**设计意图：** `system_prompt` 是旧的字符串参数，`system_message` 是新的对象参数。两者互斥，不能同时指定。如果用户传入字符串 `system_prompt`，自动转换为 `SystemMessage` 对象。这是典型的**向后兼容**设计。

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    self.model = model
    ...
```

**为什么用 `catch_warnings`？** 因为 dataclass 的字段赋值可能触发 Pydantic 或其他库的弃用警告。这里显式忽略，确保 `__init__` 不会因为第三方警告而输出噪音。

#### system_prompt 属性（L156-L164）

```python
@property
def system_prompt(self) -> str | None:
    if self.system_message is None:
        return None
    return self.system_message.text
```

**设计意图：** 提供只读的 `system_prompt` 属性，从 `system_message` 中提取文本内容。这是一个便利属性，允许用户以字符串形式读取系统提示词。

#### __setattr__ 方法（L166-L198）

```python
def __setattr__(self, name: str, value: Any) -> None:
    if name == "system_prompt":
        warnings.warn(
            "Direct attribute assignment to ModelRequest.system_prompt is deprecated. "
            "Use request.override(system_message=SystemMessage(...)) instead...",
            DeprecationWarning,
            stacklevel=2,
        )
        if value is None:
            object.__setattr__(self, "system_message", None)
        else:
            object.__setattr__(self, "system_message", SystemMessage(content=value))
        return

    warnings.warn(
        f"Direct attribute assignment to ModelRequest.{name} is deprecated. "
        f"Use request.override({name}=...) instead...",
        DeprecationWarning,
        stacklevel=2,
    )
    object.__setattr__(self, name, value)
```

**设计意图：** 拦截所有属性赋值操作，发出弃用警告。这推动用户从**可变模式**（直接修改属性）迁移到**不可变模式**（使用 `override()` 创建新实例）。

**为什么推崇不可变模式？**
1. **线程安全**：多个中间件同时修改同一个 request 会导致竞态条件
2. **可审计**：每次修改都产生新对象，可以追踪修改历史
3. **可重放**：原始 request 不变，可以重复执行
4. **洋葱模型安全**：内层中间件的修改不会影响外层的 request 引用

**`system_prompt` 的特殊处理：** 对 `system_prompt` 的赋值会自动转换为对 `system_message` 的修改，保持数据模型的一致性。

#### override 方法（L200-L282）

```python
def override(self, **overrides: Unpack[_ModelRequestOverrides]) -> ModelRequest[ContextT]:
```

**核心逻辑：**

```python
if "system_prompt" in overrides and "system_message" in overrides:
    msg = "Cannot specify both system_prompt and system_message"
    raise ValueError(msg)

if "system_prompt" in overrides:
    system_prompt = cast("str | None", overrides.pop("system_prompt"))
    if system_prompt is None:
        overrides["system_message"] = None
    else:
        overrides["system_message"] = SystemMessage(content=system_prompt)

return replace(self, **overrides)
```

**执行步骤：**
1. 检查 `system_prompt` 和 `system_message` 是否同时传入，如果是则抛出异常
2. 如果传入了 `system_prompt`，将其转换为 `system_message` 并替换到 overrides 中
3. 调用 `dataclasses.replace(self, **overrides)` 创建新实例

**`replace` 的工作原理：** 它创建一个与原对象相同类型的新对象，所有字段值与原对象相同，只有 overrides 中指定的字段被替换。原对象完全不变。

---

### 3.7 ModelResponse 数据类（L284-L297）

```python
@dataclass
class ModelResponse(Generic[ResponseT]):
    result: list[BaseMessage]
    structured_response: ResponseT | None = None
```

**字段解析：**
- `result`：模型执行产生的消息列表。通常包含一个 `AIMessage`，但如果使用了 ToolStrategy 的结构化输出，可能还会包含一个 `ToolMessage`
- `structured_response`：如果指定了 `response_format` 且模型成功生成了结构化输出，这里存储解析后的结构化对象；否则为 `None`

**为什么 `result` 是列表而非单条消息？** 因为结构化输出的 ToolStrategy 模式下，模型会同时返回一个 AIMessage（包含 tool_calls）和一个 ToolMessage（工具执行结果），两条消息都需要添加到对话历史中。

---

### 3.8 ExtendedModelResponse 数据类（L299-L326）

```python
@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    model_response: ModelResponse[ResponseT]
    command: Command[Any] | None = None
```

**作用：** 允许 `wrap_model_call` 中间件在返回模型响应的同时，附带一个 `Command` 来更新 Agent 状态。

**为什么需要这个类？** 状态钩子（before/after）通过返回 dict 来更新状态，但 `wrap_model_call` 的返回值已经被模型响应占用了。`ExtendedModelResponse` 提供了额外的 `command` 字段来携带状态更新。

**Command 的限制：** 文档明确指出 `goto`、`resume`、`graph` 在 wrap_model_call 的 Command 中尚不支持。只支持 `update`（状态更新）。

**状态合并规则：**
- messages 字段：通过 `add_messages` reducer 追加（不会覆盖）
- 非 reducer 字段：外层中间件的 command 覆盖内层的（outermost wins）

---

### 3.9 ModelCallResult 类型别名（L328-L340）

```python
ModelCallResult: TypeAlias = (
    "ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]"
)
```

**作用：** 定义 `wrap_model_call` 钩子的返回值类型。中间件可以返回三种类型：
1. `ModelResponse`：完整的模型响应
2. `AIMessage`：简化返回，自动转换为 `ModelResponse(result=[ai_msg])`
3. `ExtendedModelResponse`：带 Command 的模型响应

**为什么允许返回 AIMessage？** 很多中间件只需要修改 AI 消息的内容（如添加前缀），不需要构造完整的 `ModelResponse`。允许直接返回 `AIMessage` 降低了使用门槛。

---

### 3.10 OmitFromSchema 数据类（L342-L364）

```python
@dataclass
class OmitFromSchema:
    input: bool = True
    output: bool = True
```

**作用：** 标记 AgentState 中的字段是否从输入/输出 schema 中排除。

**三个预定义实例：**

```python
OmitFromInput = OmitFromSchema(input=True, output=False)   # 从输入 schema 排除
OmitFromOutput = OmitFromSchema(input=False, output=True)   # 从输出 schema 排除
PrivateStateAttr = OmitFromSchema(input=True, output=True)  # 从两者都排除（纯内部字段）
```

**使用场景：** 在 `AgentState` 的字段注解中使用：

```python
class MyState(AgentState):
    internal_counter: NotRequired[Annotated[int, PrivateStateAttr]]  # 用户看不到
    user_input: NotRequired[Annotated[str, OmitFromOutput]]  # 只在输入时可见
```

**为什么需要这个？** LangGraph 的 StateGraph 会根据 state_schema 自动生成输入/输出 schema。某些字段是中间件内部使用的（如 `jump_to`），不应该暴露给调用者。`OmitFromSchema` 让 `_resolve_schema()` 在合并 schema 时能正确过滤这些字段。

---

### 3.11 AgentState TypedDict（L376-L381）★核心状态定义★

```python
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

**逐字段详解：**

#### messages

```python
messages: Required[Annotated[list[AnyMessage], add_messages]]
```

- `Required`：必填字段，调用 Agent 时必须提供
- `add_messages`：LangGraph 的 reducer 函数。当多个节点返回 `{"messages": [new_msg]}` 时，`add_messages` 会将新消息追加到已有列表，而非覆盖
- 这是 Agent 的核心数据——对话历史

#### jump_to

```python
jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
```

- `NotRequired`：可选字段，调用 Agent 时不需要提供
- `EphemeralValue`：临时值 channel，不会持久化到 checkpoint。这意味着如果 Agent 被中断后恢复，`jump_to` 的值会丢失（这是正确的行为，因为跳转指令是一次性的）
- `PrivateStateAttr`：从输入和输出 schema 中排除，用户不会看到这个字段
- 中间件通过返回 `{"jump_to": "end"}` 来控制流程跳转

#### structured_response

```python
structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

- `OmitFromInput`：从输入 schema 排除（用户不需要传入结构化响应，它是 Agent 生成的）
- 但在输出 schema 中保留（用户可以从结果中获取结构化响应）
- 当模型成功生成结构化输出时，解析后的对象存储在这里

---

### 3.12 _InputAgentState / _OutputAgentState（L383-L392）

```python
class _InputAgentState(TypedDict):
    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]

class _OutputAgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]
```

**设计意图：** 分离输入和输出 schema，提供更精确的类型约束。

**输入 schema 的特殊处理：** `messages` 字段允许 `dict[str, Any]` 类型，这样用户可以直接传入 `{"role": "user", "content": "hello"}` 格式的消息，而不需要先转换为 `HumanMessage` 对象。降低了使用门槛。

**输出 schema：** 只包含 `messages` 和 `structured_response`，不包含 `jump_to`（因为它是私有的）。

---

### 3.13 StateT / StateT_co / StateT_contra 类型变量（L394-L397）

```python
StateT = TypeVar("StateT", bound=AgentState[Any], default=AgentState[Any])
StateT_co = TypeVar("StateT_co", bound=AgentState[Any], default=AgentState[Any], covariant=True)
StateT_contra = TypeVar("StateT_contra", bound=AgentState[Any], contravariant=True)
```

**三个类型变量的区别：**

| 变量 | 变性 | 用途 |
|------|------|------|
| `StateT` | 不变 | 用于 `AgentMiddleware` 的类定义，既出现在参数位置也出现在返回值位置 |
| `StateT_co` | 协变 | 用于装饰器工厂的返回类型，允许返回更具体的中间件类型 |
| `StateT_contra` | 逆变 | 用于 Protocol 的参数类型，允许接受更一般的状态类型 |

**为什么需要三种？** 这是 Python 类型系统的要求。如果一个泛型类型同时出现在函数参数和返回值中，它必须是不变的。如果只出现在返回值中，可以是协变的。如果只出现在参数中，可以是逆变的。

---

### 3.14 AgentMiddleware 基类（L403-L797）★★★核心★★★

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
```

**三个泛型参数：**
- `StateT`：中间件使用的状态类型，必须是 `AgentState` 的子类型
- `ContextT`：运行时上下文类型，用于跨请求共享数据
- `ResponseT`：结构化响应的类型

#### 类属性

```python
state_schema: type[StateT] = cast("type[StateT]", _DefaultAgentState)
tools: Sequence[BaseTool] = ()
transformers: Sequence[TransformerFactory] = ()
```

**state_schema：** 中间件的状态 schema。当中间件需要自定义状态字段时，定义一个扩展 `AgentState` 的 TypedDict 并赋值给 `state_schema`。`create_agent()` 会合并所有中间件的 state_schema。

**tools：** 中间件注册的额外工具。这些工具会被添加到 ToolNode 中。例如 `ShellToolMiddleware` 注册了 `shell` 工具，`TodoMiddleware` 注册了 `todo` 工具。

**transformers：** 流式转换器工厂。每个工厂是一个 `scope → Transformer` 的函数，用于在流式输出时转换数据（如 PII 脱敏）。在图编译时合并，顺序为：`ToolCallTransformer` → 中间件 transformers → 用户 transformers。

#### name 属性

```python
@property
def name(self) -> str:
    return self.__class__.__name__
```

**作用：** 返回中间件的名称，默认为类名。在图节点命名中使用（如 `"ShellToolMiddleware.before_agent"`）。可以被子类覆盖以提供自定义名称。

---

#### 状态钩子详解

##### before_agent（L418-L432）

```python
def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
```

**执行时机：** Agent 开始执行前，只运行一次。

**参数：**
- `state`：当前 Agent 状态（只读，不应直接修改）
- `runtime`：运行时上下文，包含 `stream_writer`（发送自定义流事件）和 `context`（用户传入的上下文对象）

**返回值：**
- `dict[str, Any]`：状态更新，会被合并到 Agent 状态中
- `None`：不做任何更新

**典型用途：** 初始化资源（如启动 Shell 进程、创建临时目录）

##### abefore_agent（L434-L448）

异步版本的 `before_agent`，签名和语义完全相同，只是用 `async def` 定义。

##### before_model（L450-L464）

```python
def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
```

**执行时机：** 每次调用模型之前。在 Agent 循环中，每次 LLM 调用前都会执行。

**典型用途：** 检查调用次数、生成摘要、检查预算

##### abefore_model（L466-L480）

异步版本。

##### after_model（L482-L496）

```python
def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
```

**执行时机：** 每次模型调用之后。此时 state 中已经包含了模型返回的 AIMessage。

**典型用途：** 人工审批（检查 tool_calls 是否需要审批）、工具调用计数、PII 检测

##### aafter_model（L498-L512）

异步版本。

##### after_agent（L721-L735）

```python
def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
```

**执行时机：** Agent 执行完成后，只运行一次。

**典型用途：** 清理资源（如停止 Shell 进程、删除临时目录）

##### aafter_agent（L737-L751）

异步版本。

---

#### 包裹钩子详解

##### wrap_model_call（L514-L620）

```python
def wrap_model_call(
    self,
    request: ModelRequest[ContextT],
    handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
```

**这是洋葱模型的核心钩子。**

**参数：**
- `request`：模型请求对象，包含模型、消息、工具、状态等所有信息
- `handler`：执行内层逻辑的回调函数。调用 `handler(request)` 会执行内层中间件或最终的模型调用

**返回值：** 三种类型之一（见 ModelCallResult）

**默认实现：** 抛出 `NotImplementedError`，并给出详细的错误提示。错误信息告诉用户：
1. 你可能只定义了异步版本但用了同步调用
2. 解决方案：实现同步版本、使用装饰器、或改用异步调用

**为什么默认抛异常而非 pass？** 因为如果用户只定义了 `awrap_model_call` 但用同步方式调用 Agent，需要一个清晰的错误提示，而不是静默跳过。

**五种典型用法（代码注释中的示例）：**

1. **重试**：循环调用 `handler(request)` 直到成功
2. **改写响应**：调用 `handler` 后修改 AIMessage 内容
3. **降级**：捕获异常后返回兜底响应
4. **缓存/短路**：有缓存时不调用 `handler`，直接返回缓存结果
5. **简化返回**：直接返回 `AIMessage`，自动转换为 `ModelResponse`

##### awrap_model_call（L622-L680）

异步版本。`handler` 的类型变为 `Awaitable[ModelResponse]`，需要 `await handler(request)`。

##### wrap_tool_call（L753-L820）

```python
def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
```

**参数：**
- `request`：工具调用请求，包含 `tool_call`（调用字典）、`tool`（工具实例）、`state`、`runtime`
- `handler`：执行内层逻辑的回调函数

**返回值：** `ToolMessage` 或 `Command`

**典型用法：**
1. 修改请求参数（如将参数值翻倍）
2. 重试失败的调用
3. 条件重试（检查 ToolMessage.status）
4. 缓存

##### awrap_tool_call（L822-L877）

异步版本。

---

### 3.15 Protocol 类定义（L799-L840）

```python
class _CallableWithStateAndRuntime(Protocol[StateT_contra, ContextT]):
    def __call__(
        self, state: StateT_contra, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | Command[Any] | None | Awaitable[...]:
        ...

class _CallableReturningSystemMessage(Protocol[StateT_contra, ContextT]):
    def __call__(
        self, request: ModelRequest[ContextT]
    ) -> str | SystemMessage | Awaitable[str | SystemMessage]:
        ...

class _CallableReturningModelResponse(Protocol[StateT_contra, ContextT, ResponseT]):
    def __call__(
        self, request, handler
    ) -> ModelResponse[ResponseT] | AIMessage:
        ...

class _CallableReturningToolResponse(Protocol):
    def __call__(
        self, request, handler
    ) -> ToolMessage | Command[Any]:
        ...
```

**作用：** 定义装饰器工厂接受的函数签名协议。这些 Protocol 不用于运行时检查，只用于类型提示。

**四个 Protocol 对应四种装饰器：**
1. `_CallableWithStateAndRuntime` → `before_model` / `after_model` / `before_agent` / `after_agent`
2. `_CallableReturningSystemMessage` → `dynamic_prompt`
3. `_CallableReturningModelResponse` → `wrap_model_call`
4. `_CallableReturningToolResponse` → `wrap_tool_call`

---

### 3.16 hook_config 装饰器（L842-L910）

```python
def hook_config(
    *,
    can_jump_to: list[JumpTo] | None = None,
) -> Callable[[CallableT], CallableT]:
```

**作用：** 配置钩子方法的行为。目前只支持 `can_jump_to` 参数。

**执行逻辑：**

```python
def decorator(func: CallableT) -> CallableT:
    if can_jump_to is not None:
        func.__can_jump_to__ = can_jump_to
    return func
return decorator
```

**设计意图：** 将 `can_jump_to` 元数据附加到函数对象上。`factory.py` 中的 `_get_can_jump_to()` 函数会读取这个元数据来决定是否为该中间件节点添加条件边。

**为什么用装饰器而非参数？** 两种方式都支持：

```python
# 方式1：装饰器
class MyMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end", "model"])
    def before_model(self, state, runtime):
        ...

# 方式2：装饰器工厂参数
@before_model(can_jump_to=["end"])
def my_hook(state, runtime):
    ...
```

---

### 3.17 before_model 装饰器工厂（L912-L1107）

```python
@overload
def before_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...

@overload
def before_model(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[[_CallableWithStateAndRuntime], AgentMiddleware]: ...

def before_model(
    func=None, *, state_schema=None, tools=None, can_jump_to=None, name=None
):
```

**两个重载：**
1. **无括号用法**：`@before_model` — 直接装饰函数
2. **有括号用法**：`@before_model(can_jump_to=["end"])` — 带参数装饰

**核心逻辑（decorator 内部）：**

```python
def decorator(func) -> AgentMiddleware:
    is_async = iscoroutinefunction(func)
    func_can_jump_to = can_jump_to or getattr(func, "__can_jump_to__", [])

    if is_async:
        async def async_wrapped(_self, state, runtime):
            return await func(state, runtime)
        if func_can_jump_to:
            async_wrapped.__can_jump_to__ = func_can_jump_to
        return type(
            name or func.__name__,
            (AgentMiddleware,),
            {"state_schema": state_schema or AgentState, "tools": tools or [],
             "abefore_model": async_wrapped},
        )()

    def wrapped(_self, state, runtime):
        return func(state, runtime)
    if func_can_jump_to:
        wrapped.__can_jump_to__ = func_can_jump_to
    return type(
        name or func.__name__,
        (AgentMiddleware,),
        {"state_schema": state_schema or AgentState, "tools": tools or [],
         "before_model": wrapped},
    )()
```

**执行步骤：**
1. 检测函数是同步还是异步
2. 提取 `can_jump_to` 元数据（优先使用装饰器参数，其次使用函数属性）
3. 创建一个包装函数，签名为 `(self, state, runtime)`，内部调用原始函数 `(state, runtime)`
4. 使用 `type()` 动态创建 `AgentMiddleware` 子类
5. 实例化并返回

**`type()` 动态类创建的精妙之处：**

```python
type(
    "MyMiddleware",          # 类名
    (AgentMiddleware,),      # 基类
    {                        # 属性字典
        "state_schema": AgentState,
        "tools": [],
        "before_model": wrapped,  # 或 "abefore_model": async_wrapped
    },
)()
```

这等价于：

```python
class MyMiddleware(AgentMiddleware):
    state_schema = AgentState
    tools = []
    def before_model(self, state, runtime):
        return func(state, runtime)

MyMiddleware()
```

但 `type()` 是运行时动态创建，不需要预先定义类。这让装饰器可以接受任意函数并自动生成中间件。

**包装函数的 `_self` 参数：** 因为动态创建的类方法会接收 `self` 作为第一个参数，但原始函数不需要 `self`。所以包装函数用 `_self` 接收并忽略它。

---

### 3.18 after_model 装饰器工厂（L1109-L1296）

结构与 `before_model` 完全相同，只是：
- 钩子方法名从 `before_model` 变为 `after_model`
- 默认类名从 `BeforeModelMiddleware` 变为 `AfterModelMiddleware`

---

### 3.19 before_agent 装饰器工厂（L1298-L1493）

结构与 `before_model` 完全相同，只是：
- 钩子方法名从 `before_model` 变为 `before_agent`
- 默认类名从 `BeforeModelMiddleware` 变为 `BeforeAgentMiddleware`

---

### 3.20 after_agent 装饰器工厂（L1495-L1689）

结构与 `before_model` 完全相同，只是：
- 钩子方法名从 `before_model` 变为 `after_agent`
- 默认类名从 `BeforeModelMiddleware` 变为 `AfterAgentMiddleware`

---

### 3.21 dynamic_prompt 装饰器工厂（L1691-L1790）★特殊装饰器★

```python
def dynamic_prompt(
    func: _CallableReturningSystemMessage[StateT, ContextT] | None = None,
) -> AgentMiddleware:
```

**与其他装饰器的区别：**
- 接受的函数签名为 `(request: ModelRequest) -> str | SystemMessage`，而非 `(state, runtime)`
- 内部使用 `wrap_model_call` 而非 `before_model` 来实现

**为什么用 wrap_model_call？** 因为动态提示词需要访问 `ModelRequest` 来获取运行时上下文和状态，而 `before_model` 只能访问 `(state, runtime)`。通过 `wrap_model_call`，可以在调用模型之前修改 `request.system_message`。

**核心逻辑：**

```python
def wrapped(_self, request, handler):
    prompt = func(request)
    if isinstance(prompt, SystemMessage):
        request = request.override(system_message=prompt)
    else:
        request = request.override(system_message=SystemMessage(content=prompt))
    return handler(request)
```

**执行步骤：**
1. 调用用户函数获取提示词（字符串或 SystemMessage）
2. 使用 `override()` 创建新的 request，替换 system_message
3. 调用 handler 继续执行（内层中间件或模型调用）

**同步函数的特殊处理：** 即使原始函数是同步的，也会生成 `awrap_model_call` 实现（`async_wrapped_from_sync`），确保异步调用路径也能工作。

---

### 3.22 wrap_model_call 装饰器工厂（L1792-L1930）

```python
def wrap_model_call(
    func=None, *, state_schema=None, tools=None, name=None
) -> AgentMiddleware:
```

**与 before_model 装饰器的区别：**
- 不支持 `can_jump_to`（因为 wrap 钩子通过修改 request/响应来控制流程，而非跳转）
- 接受的函数签名为 `(request, handler) -> ModelResponse | AIMessage`
- 动态创建的类设置 `wrap_model_call` 或 `awrap_model_call` 属性

**核心逻辑：**

```python
def decorator(func) -> AgentMiddleware:
    is_async = iscoroutinefunction(func)
    if is_async:
        async def async_wrapped(_self, request, handler):
            return await func(request, handler)
        return type(name or func.__name__, (AgentMiddleware,), {
            "awrap_model_call": async_wrapped,
        })()

    def wrapped(_self, request, handler):
        return func(request, handler)
    return type(name or func.__name__, (AgentMiddleware,), {
        "wrap_model_call": wrapped,
    })()
```

---

### 3.23 wrap_tool_call 装饰器工厂（L1932-L2065）

结构与 `wrap_model_call` 类似，区别：
- 接受的函数签名为 `(request: ToolCallRequest, handler) -> ToolMessage | Command`
- 不支持 `state_schema` 参数（工具调用不需要自定义状态）
- 动态创建的类设置 `wrap_tool_call` 或 `awrap_tool_call` 属性

---

## 四、核心设计模式总结

### 4.1 不可变模式

`ModelRequest` 通过 `override()` + `dataclasses.replace()` 实现不可变修改。每次修改都产生新对象，原对象不变。

### 4.2 装饰器工厂模式

`before_model` 等装饰器支持两种用法（带括号/不带括号），通过 `@overload` 和 `func=None` 默认值实现。

### 4.3 动态类创建

使用 `type(name, bases, dict)()` 在运行时动态创建 `AgentMiddleware` 子类并实例化，避免用户为每个钩子函数手写类定义。

### 4.4 协变/逆变类型变量

`StateT_co`（协变）用于返回类型，`StateT_contra`（逆变）用于参数类型，`StateT`（不变）用于两者兼有的场景。

### 4.5 元数据传递

`__can_jump_to__` 属性附加在方法对象上，由 `factory.py` 在构建图时读取。这是一种轻量级的元数据传递机制，不需要额外的注册表或配置对象。
