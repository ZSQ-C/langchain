# 工具中间件深度详解

> 涵盖文件：`shell_tool.py`、`todo.py`、`tool_emulator.py`、`file_search.py`、`tool_selection.py`

---

## 第一部分：shell_tool.py

---

### 1. 文件定位

| 维度 | 说明 |
|------|------|
| **所属模块** | `langchain.agents.middleware` |
| **核心职责** | 持久化 Shell 工具中间件，为 Agent 提供一个持久的 Shell 会话，支持顺序执行多条命令 |
| **解决什么问题** | Agent 需要执行 Shell 命令来完成开发任务（如安装依赖、运行测试、查看文件），但每次调用工具都启动新 Shell 效率低下且丢失上下文。本中间件维护一个长生命周期的 Shell 进程，命令之间共享工作目录、环境变量和进程状态 |
| **安全特性** | 支持三种执行策略（Host/Docker/CodexSandbox）、PII 脱敏规则、输出截断、超时控制 |

---

### 2. 代码结构总览

```
shell_tool.py
├── 常量
│   ├── DEFAULT_TOOL_DESCRIPTION    # 默认工具描述
│   ├── SHELL_TOOL_NAME             # 默认工具名 "shell"
│   └── _DONE_MARKER_PREFIX         # 命令完成标记前缀
├── 工具函数
│   └── _cleanup_resources()        # 清理 Shell 会话和临时目录
├── 数据类
│   ├── _SessionResources           # 每次运行的 Shell 资源容器
│   └── CommandExecutionResult      # 命令执行结果
├── ShellSession                    # 持久化 Shell 会话
│   ├── __init__()                  # 初始化进程参数
│   ├── start()                     # 启动 Shell 子进程
│   ├── restart()                   # 重启 Shell
│   ├── stop()                      # 停止 Shell
│   ├── execute()                   # 执行命令（核心方法）
│   ├── _collect_output()           # 收集命令输出
│   ├── _collect_output_after_exit() # Shell 意外退出后收集输出
│   ├── _kill_process()             # 强制杀死进程
│   ├── _enqueue_stream()           # 将流输出入队列
│   ├── _drain_queue()              # 清空队列
│   ├── _drain_remaining_stderr()   # 排空残余 stderr
│   └── _safe_int()                 # 安全整数转换
├── _ShellToolInput                 # Shell 工具输入 Schema
└── ShellToolMiddleware             # 核心中间件
    ├── __init__()                  # 初始化配置和工具
    ├── before_agent()              # Agent 启动前创建 Shell 会话
    ├── after_agent()               # Agent 结束后清理资源
    ├── _get_or_create_resources()  # 获取或创建资源
    ├── _create_resources()         # 创建新资源
    ├── _run_startup_commands()     # 执行启动命令
    ├── _run_shutdown_commands()    # 执行关闭命令
    ├── _apply_redactions()         # 应用脱敏规则
    ├── _run_shell_tool()           # Shell 工具执行逻辑
    └── _format_tool_message()      # 格式化工具消息
```

---

### 3. 逐段深度详解

---

#### 3.1 命令完成标记机制

```python
_DONE_MARKER_PREFIX = "__LC_SHELL_DONE__"
```

**核心问题**：如何判断一条 Shell 命令已经执行完毕？

**解决方案**：在每条命令后注入一个 `printf` 命令，输出唯一标记 + 退出码：

```bash
用户命令
printf '__LC_SHELL_DONE_<uuid> %s\n' $?
```

- `<uuid>` 是每次执行生成的唯一标识，避免与命令输出混淆
- `$?` 是上一条命令的退出码
- 当从 stdout 读取到以 `__LC_SHELL_DONE_` 开头的行时，表示命令执行完毕

---

#### 3.2 _SessionResources

```python
@dataclass
class _SessionResources:
    session: ShellSession
    tempdir: tempfile.TemporaryDirectory[str] | None
    policy: BaseExecutionPolicy
    finalizer: weakref.finalize = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.finalizer = weakref.finalize(
            self, _cleanup_resources, self.session, self.tempdir, self.policy.termination_timeout
        )
```

**关键设计**：`weakref.finalize` 确保即使中间件被垃圾回收，Shell 会话和临时目录也能被正确清理

- `finalizer` 在 `_SessionResources` 对象被 GC 时自动调用 `_cleanup_resources()`
- 这是一种"安全网"机制，防止资源泄漏

---

#### 3.3 ShellSession — 持久化 Shell 会话

##### `__init__` 方法

```python
def __init__(self, workspace, policy, command, environment):
    self._workspace = workspace
    self._policy = policy
    self._command = command
    self._environment = dict(environment)
    self._process: subprocess.Popen[str] | None = None
    self._stdin: Any = None
    self._queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    self._lock = threading.Lock()
    self._stdout_thread: threading.Thread | None = None
    self._stderr_thread: threading.Thread | None = None
    self._terminated = False
```

**关键属性**：
- `_process`：Shell 子进程（`subprocess.Popen`）
- `_stdin`：子进程的标准输入，用于写入命令
- `_queue`：线程安全的输出队列，stdout/stderr 读取线程将输出放入此队列
- `_lock`：线程锁，确保同一时间只有一个命令在执行
- `_terminated`：标记进程是否已终止

##### `start()` 方法详解

```python
def start(self) -> None:
```

**内部执行步骤**：

1. 如果进程已在运行，直接返回
2. 调用 `self._policy.spawn()` 启动子进程
3. 验证 stdin/stdout/stderr 管道已建立
4. 清空输出队列
5. 启动 stdout 和 stderr 读取线程（`_enqueue_stream`），将输出放入 `_queue`

##### `execute()` 方法详解 — 核心方法

```python
def execute(self, command: str, *, timeout: float) -> CommandExecutionResult:
```

**内部执行步骤**：

1. 验证 Shell 进程正在运行
2. 生成唯一标记：`marker = f"{_DONE_MARKER_PREFIX}{uuid.uuid4().hex}"`
3. 计算截止时间：`deadline = time.monotonic() + timeout`
4. 获取线程锁（确保命令顺序执行）
5. 清空队列中的残留输出
6. 写入命令和标记命令：
   ```python
   self._stdin.write(payload)          # 用户命令
   self._stdin.write(f"printf '{marker} %s\\n' $?\n")  # 完成标记
   self._stdin.flush()
   ```
7. 如果写入失败（`BrokenPipeError`），说明 Shell 已退出，调用 `_collect_output_after_exit()`
8. 否则调用 `_collect_output()` 收集输出

##### `_collect_output()` 方法详解

```python
def _collect_output(self, marker, deadline, timeout) -> CommandExecutionResult:
```

**内部执行步骤**：

1. 初始化收集变量：`collected`、`total_lines`、`total_bytes`、`truncated_by_lines`、`truncated_by_bytes`、`exit_code`、`timed_out`
2. 进入循环，从队列中读取输出：
   - 计算剩余时间 `remaining = deadline - time.monotonic()`
   - 如果超时，设置 `timed_out = True`，跳出循环
   - 从队列获取 `(source, data)` 元组
   - 如果 `source == "stdout"` 且 `data.startswith(marker)`，提取退出码，排空残余 stderr，跳出循环
   - 否则，累加行数和字节数，检查是否超出限制
   - stderr 行添加 `[stderr]` 前缀
3. 如果超时，重启 Shell 会话并返回超时结果
4. 否则返回正常结果

**输出截断机制**：
- `max_output_lines`：超过此行数的输出被丢弃
- `max_output_bytes`：超过此字节数的输出被丢弃
- 截断后仍然统计 `total_lines` 和 `total_bytes`，让调用方知道实际输出量

##### `_drain_remaining_stderr()` 方法详解

```python
def _drain_remaining_stderr(self, collected, deadline, drain_timeout=0.05):
```

**核心问题**：stdout 和 stderr 读取线程独立运行。当 stdout 上的完成标记到达时，stderr 可能还有输出在传输中。

**解决方案**：短暂等待（50ms），从队列中排空所有 stderr 输出

---

#### 3.4 _ShellToolInput

```python
class _ShellToolInput(BaseModel):
    command: str | None = None
    restart: bool | None = None
    runtime: Annotated[Any, SkipJsonSchema()] = None

    @model_validator(mode="after")
    def validate_payload(self) -> _ShellToolInput:
        if self.command is None and not self.restart:
            raise ValueError("Shell tool requires either 'command' or 'restart'.")
        if self.command is not None and self.restart:
            raise ValueError("Specify only one of 'command' or 'restart'.")
        return self
```

- 互斥验证：`command` 和 `restart` 只能选一个
- `runtime` 字段使用 `SkipJsonSchema()` 排除在 JSON Schema 之外（不暴露给模型）

---

#### 3.5 ShellToolMiddleware

##### `__init__` 方法详解

```python
def __init__(
    self,
    workspace_root: str | Path | None = None,
    *,
    startup_commands: tuple[str, ...] | list[str] | str | None = None,
    shutdown_commands: tuple[str, ...] | list[str] | str | None = None,
    execution_policy: BaseExecutionPolicy | None = None,
    redaction_rules: tuple[RedactionRule, ...] | list[RedactionRule] | None = None,
    tool_description: str | None = None,
    tool_name: str = SHELL_TOOL_NAME,
    shell_command: Sequence[str] | str | None = None,
    env: Mapping[str, Any] | None = None,
) -> None:
```

**入参解析**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `workspace_root` | None | 工作目录，None 则创建临时目录 |
| `startup_commands` | None | Shell 启动后执行的命令 |
| `shutdown_commands` | None | Shell 关闭前执行的命令 |
| `execution_policy` | HostExecutionPolicy() | 执行策略 |
| `redaction_rules` | None | 输出脱敏规则 |
| `tool_description` | 默认描述 | 工具描述 |
| `tool_name` | `"shell"` | 工具名称 |
| `shell_command` | `("/bin/bash",)` | Shell 可执行文件 |
| `env` | None | 环境变量 |

**内部执行步骤**：

1. 标准化各种输入格式（字符串→元组、字典值→字符串等）
2. 解析脱敏规则：`RedactionRule.resolve()` → `ResolvedRedactionRule`
3. 创建 Shell 工具闭包：`@tool(...)` 装饰器创建工具函数，捕获 `self`
4. 将工具注册到 `self.tools`

##### `before_agent()` 钩子详解

```python
def before_agent(self, state, runtime) -> dict[str, Any] | None:
```

**执行时机**：Agent 启动前

**内部执行步骤**：
1. 调用 `_get_or_create_resources(state)` 获取或创建 Shell 会话资源
2. 返回 `{"shell_session_resources": resources}`，将资源存入状态

**关键设计**：`_get_or_create_resources()` 先检查状态中是否已有资源（如从 interrupt 恢复后），避免重复创建

##### `after_agent()` 钩子详解

```python
def after_agent(self, state, runtime) -> None:
```

**执行时机**：Agent 结束后

**内部执行步骤**：
1. 从状态中获取 `_SessionResources`
2. 执行关闭命令
3. 调用 `resources.finalizer()` 清理资源

##### `_run_shell_tool()` 方法详解 — 工具执行逻辑

```python
def _run_shell_tool(self, resources, payload, *, tool_call_id) -> Any:
```

**内部执行步骤**：

1. **处理重启请求**：如果 `payload.get("restart")`，重启 Shell 会话并重新执行启动命令
2. **执行命令**：`session.execute(command, timeout=...)`
3. **处理超时**：返回错误 ToolMessage
4. **应用脱敏规则**：`_apply_redactions(result.output)`
   - 如果触发 `block` 策略（`PIIDetectionError`），返回错误消息
5. **处理截断**：在输出末尾添加截断提示
6. **处理非零退出码**：在输出末尾添加退出码，设置 `status="error"`
7. **构建 artifact**：包含执行元数据（超时、退出码、截断信息、脱敏匹配）
8. 返回 `ToolMessage`

---

## 第二部分：todo.py

---

### 4. 文件定位

| 维度 | 说明 |
|------|------|
| **所属模块** | `langchain.agents.middleware` |
| **核心职责** | 任务管理中间件，为 Agent 提供 `write_todos` 工具来创建和管理结构化任务列表 |
| **解决什么问题** | 复杂任务需要分步骤执行和跟踪进度。本中间件让 Agent 能够创建任务列表、更新任务状态，并向用户展示执行进度 |

---

### 5. 代码结构总览

```
todo.py
├── 类型定义
│   ├── Todo                      # 单个任务 TypedDict（content + status）
│   ├── PlanningState             # 扩展状态（增加 todos 字段）
│   └── WriteTodosInput           # 工具输入 Schema
├── 常量
│   ├── WRITE_TODOS_TOOL_DESCRIPTION  # 工具描述（非常详细）
│   └── WRITE_TODOS_SYSTEM_PROMPT     # 系统提示词
├── 工具函数
│   ├── write_todos()             # 模块级工具函数（简化版）
│   ├── _write_todos()            # 中间件级工具函数（带 runtime）
│   └── _awrite_todos()           # 异步版本
└── TodoListMiddleware            # 核心中间件
    ├── __init__()                # 初始化工具和提示词
    ├── wrap_model_call()         # 注入系统提示词
    ├── awrap_model_call()        # 异步版本
    ├── after_model()             # 检测并行调用冲突
    └── aafter_model()            # 异步版本
```

---

### 6. 逐段深度详解

---

#### 6.1 Todo TypedDict

```python
class Todo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]
```

- `content`：任务描述
- `status`：任务状态，三种取值
  - `"pending"`：未开始
  - `"in_progress"`：进行中
  - `"completed"`：已完成

#### 6.2 PlanningState

```python
class PlanningState(AgentState[ResponseT]):
    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
```

- 扩展 `AgentState`，增加 `todos` 字段
- `NotRequired`：字段可选（新 Agent 没有 todos）
- `OmitFromInput`：在 `create_agent(input=...)` 时自动排除，避免用户输入覆盖 Agent 的任务列表

#### 6.3 WRITE_TODOS_TOOL_DESCRIPTION

这是一个非常详细的工具描述，包含以下部分：

| 部分 | 内容 |
|------|------|
| **何时使用** | 复杂多步骤任务、用户请求 todo 列表、用户提供多个任务 |
| **如何使用** | 开始前标记 in_progress、完成后标记 completed、可同时更新多个任务 |
| **何时不使用** | 单一简单任务、3 步以内的任务、纯对话任务 |
| **任务状态管理** | 状态含义、管理规则、完成要求 |
| **任务分解** | 创建具体可操作的项、将复杂任务拆分为小步骤 |

**设计意图**：工具描述本身就是"使用指南"，通过详细的描述引导模型正确使用 todo 功能

#### 6.4 write_todos 工具

```python
@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)],
        }
    )
```

**关键设计**：
- 返回 `Command` 而非简单的字符串——同时更新 `todos` 状态字段和 `messages`
- `InjectedToolCallId`：自动注入工具调用 ID，不需要模型提供
- `Command(update={...})`：LangGraph 的状态更新命令，可以同时更新多个通道

#### 6.5 TodoListMiddleware

##### `__init__` 方法

```python
def __init__(self, *, system_prompt=..., tool_description=...):
    self.system_prompt = system_prompt
    self.tool_description = tool_description
    self.tools = [
        StructuredTool.from_function(
            name="write_todos",
            description=tool_description,
            func=_write_todos,
            coroutine=_awrite_todos,
            args_schema=WriteTodosInput,
            infer_schema=False,
        )
    ]
```

- 创建 `StructuredTool` 实例，注册到 `self.tools`
- 同时提供同步和异步实现

##### `wrap_model_call()` 方法详解

```python
def wrap_model_call(self, request, handler):
```

**内部执行步骤**：

1. 检查 `request.system_message` 是否存在
2. 如果存在，在现有系统消息的内容块列表末尾追加 todo 系统提示词
3. 如果不存在，创建新的 `SystemMessage`
4. 使用 `request.override(system_message=new_system_message)` 创建新请求
5. 调用 `handler(overridden_request)`

**设计意图**：使用 `wrap_model_call`（洋葱模型）而非 `before_model`（状态钩子），因为系统提示词的注入只影响当前模型调用，不需要修改图状态

##### `after_model()` 方法详解 — 并行调用检测

```python
def after_model(self, state, runtime) -> dict[str, Any] | None:
```

**核心问题**：`write_todos` 工具替换整个 todo 列表，如果模型并行调用多次，会产生冲突

**内部执行步骤**：

1. 获取最后一条 AIMessage
2. 统计 `write_todos` 工具调用次数
3. 如果超过 1 次，为每个调用创建错误 ToolMessage：
   ```
   "Error: The `write_todos` tool should never be called multiple times in parallel."
   ```
4. 返回 `{"messages": error_messages}`

**设计意图**：通过返回错误 ToolMessage，让模型知道并行调用不被允许，下次会只调用一次

---

## 第三部分：tool_emulator.py

---

### 7. 文件定位

| 维度 | 说明 |
|------|------|
| **所属模块** | `langchain.agents.middleware` |
| **核心职责** | 工具模拟中间件，使用 LLM 代替真实工具执行，用于测试 |
| **解决什么问题** | 在测试 Agent 行为时，不希望调用真实的外部工具（如 API、数据库），但需要工具返回合理的响应。本中间件拦截工具调用，用 LLM 生成模拟响应 |

---

### 8. 代码结构总览

```
tool_emulator.py
└── LLMToolEmulator               # 核心中间件
    ├── __init__()                 # 初始化模拟工具列表和模型
    ├── wrap_tool_call()           # 拦截并模拟工具调用
    └── awrap_tool_call()          # 异步版本
```

---

### 9. 逐段深度详解

---

#### 9.1 LLMToolEmulator.__init__()

```python
def __init__(self, *, tools=None, model=None):
    self.emulate_all = tools is None
    self.tools_to_emulate: set[str] = set()

    if not self.emulate_all and tools is not None:
        for tool in tools:
            if isinstance(tool, str):
                self.tools_to_emulate.add(tool)
            else:
                self.tools_to_emulate.add(tool.name)

    if model is None:
        self.model = init_chat_model("anthropic:claude-sonnet-4-5-20250929", temperature=1)
    elif isinstance(model, BaseChatModel):
        self.model = model
    else:
        self.model = init_chat_model(model, temperature=1)
```

**关键参数**：
- `tools=None`：模拟所有工具（默认）
- `tools=[]`：不模拟任何工具
- `tools=["weather", calculator]`：只模拟指定工具
- `model`：默认使用 Claude Sonnet，`temperature=1` 增加响应多样性

#### 9.2 wrap_tool_call() 核心方法

```python
def wrap_tool_call(self, request, handler):
```

**内部执行步骤**：

1. 提取工具名称
2. 判断是否需要模拟：`should_emulate = self.emulate_all or tool_name in self.tools_to_emulate`
3. 如果不需要模拟，直接调用 `handler(request)` 透传
4. 如果需要模拟：
   a. 提取工具参数和描述
   b. 构建 LLM 提示词：
      ```
      You are emulating a tool call for testing purposes.
      Tool: {tool_name}
      Description: {tool_description}
      Arguments: {tool_args}
      Generate a realistic response that this tool would return given these arguments.
      Return ONLY the tool's output, no explanation or preamble. Introduce variation into your responses.
      ```
   c. 调用模拟模型：`self.model.invoke([HumanMessage(prompt)])`
   d. 返回 `ToolMessage(content=response.content, tool_call_id=..., name=...)`

**关键设计**：
- 模拟响应直接返回，不调用真实工具（`handler` 不被调用）
- `temperature=1` 确保每次模拟响应不同，更接近真实场景
- 提示词要求"只返回工具输出"，避免 LLM 添加额外解释

---

## 第四部分：file_search.py

---

### 10. 文件定位

| 维度 | 说明 |
|------|------|
| **所属模块** | `langchain.agents.middleware` |
| **核心职责** | 文件搜索中间件，为 Agent 提供 Glob 和 Grep 搜索工具 |
| **解决什么问题** | Agent 需要在文件系统中搜索文件（按名称模式）和内容（按正则表达式），类似于 IDE 的搜索功能 |

---

### 11. 代码结构总览

```
file_search.py
├── 工具函数
│   ├── _expand_include_patterns()     # 展开花括号模式
│   ├── _is_valid_include_pattern()    # 验证 glob 模式
│   └── _match_include_pattern()       # 匹配文件名
└── FilesystemFileSearchMiddleware     # 核心中间件
    ├── __init__()                     # 初始化搜索工具
    ├── _validate_and_resolve_path()   # 路径验证和安全检查
    ├── _ripgrep_search()              # ripgrep 搜索
    ├── _python_search()               # Python 回退搜索
    ├── _format_grep_results()         # 格式化搜索结果
    ├── glob_search                    # Glob 工具（闭包）
    └── grep_search                    # Grep 工具（闭包）
```

---

### 12. 逐段深度详解

---

#### 12.1 _expand_include_patterns()

```python
def _expand_include_patterns(pattern: str) -> list[str] | None:
```

**方法作用**：展开 `*.{py,pyi}` 这样的花括号模式为 `["*.py", "*.pyi"]`

**内部执行步骤**：

1. 如果 `}` 存在但 `{` 不存在，返回 None（无效模式）
2. 递归展开函数 `_expand(current)`：
   - 找到 `{` 和 `}` 的位置
   - 提取前缀、选项列表、后缀
   - 对每个选项递归展开（支持嵌套花括号）
3. 如果展开失败（ValueError），返回 None

#### 12.2 _validate_and_resolve_path()

```python
def _validate_and_resolve_path(self, path: str) -> Path:
```

**安全检查**：

1. 确保路径以 `/` 开头（虚拟路径格式）
2. 检查路径穿越：`..` 和 `~` 不允许
3. 解析为绝对路径：`full_path = (self.root_path / relative).resolve()`
4. 确保路径在根目录内：`full_path.relative_to(self.root_path)` 不抛出异常

**设计意图**：防止 Agent 通过路径穿越访问根目录之外的文件

#### 12.3 glob_search 工具

```python
@tool
def glob_search(pattern: str, path: str = "/") -> str:
```

**内部执行步骤**：

1. 验证和解析路径
2. 使用 `pathlib.glob(pattern)` 匹配文件
3. 只保留文件（排除目录）
4. 转换为虚拟路径格式（相对于 root_path）
5. 获取修改时间
6. 按修改时间降序排序
7. 返回换行分隔的文件路径列表

#### 12.4 grep_search 工具

```python
@tool
def grep_search(
    pattern: str, path: str = "/", include: str | None = None,
    output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
) -> str:
```

**内部执行步骤**：

1. 编译正则表达式（验证语法）
2. 验证 include 模式
3. **优先使用 ripgrep**：`self._ripgrep_search(pattern, path, include)`
4. **ripgrep 不可用时回退到 Python**：`self._python_search(pattern, path, include)`
5. 格式化结果

#### 12.5 _ripgrep_search()

```python
def _ripgrep_search(self, pattern, base_path, include):
```

**内部执行步骤**：

1. 构建命令：`rg --json [--glob include] -- pattern base_path`
2. 执行子进程（30 秒超时）
3. 解析 JSON 输出：
   - 只处理 `type == "match"` 的行
   - 提取虚拟路径、行号、行文本
4. 返回 `{virtual_path: [(line_num, line_text), ...]}` 格式

#### 12.6 _python_search()

```python
def _python_search(self, pattern, base_path, include):
```

**内部执行步骤**：

1. 编译正则表达式
2. 递归遍历目录：`base_full.rglob("*")`
3. 应用 include 过滤
4. 跳过超大文件
5. 逐行搜索匹配
6. 返回与 `_ripgrep_search` 相同格式的结果

#### 12.7 _format_grep_results()

三种输出模式：

| 模式 | 格式 | 示例 |
|------|------|------|
| `files_with_matches` | 文件路径列表 | `/src/main.py\n/src/utils.py` |
| `content` | `file:line:content` | `/src/main.py:42:def hello():` |
| `count` | `file:count` | `/src/main.py:5` |

---

## 第五部分：tool_selection.py

---

### 13. 文件定位

| 维度 | 说明 |
|------|------|
| **所属模块** | `langchain.agents.middleware` |
| **核心职责** | LLM 工具选择中间件，在主模型调用前使用一个小模型筛选相关工具 |
| **解决什么问题** | 当 Agent 有大量工具时（如 50+），将所有工具描述发送给主模型会消耗大量 Token 且降低选择准确性。本中间件先用一个小模型筛选出最相关的工具，再发送给主模型 |

---

### 14. 代码结构总览

```
tool_selection.py
├── 常量
│   └── DEFAULT_SYSTEM_PROMPT      # 默认选择提示词
├── 数据类
│   └── _SelectionRequest          # 选择请求封装
├── 工具函数
│   ├── _create_tool_selection_response()  # 动态创建选择响应 Schema
│   └── _render_tool_list()                # 渲染工具列表为 Markdown
└── LLMToolSelectorMiddleware      # 核心中间件
    ├── __init__()                 # 初始化模型和配置
    ├── _prepare_selection_request()  # 准备选择请求
    ├── _process_selection_response() # 处理选择响应
    ├── wrap_model_call()          # 洋葱模型钩子
    └── awrap_model_call()         # 异步版本
```

---

### 15. 逐段深度详解

---

#### 15.1 _create_tool_selection_response()

```python
def _create_tool_selection_response(tools: list[BaseTool]) -> TypeAdapter[Any]:
```

**方法作用**：动态创建结构化输出 Schema，让选择模型返回选中的工具名称列表

**内部执行步骤**：

1. 为每个工具创建 `Annotated[Literal["tool_name"], Field(description="...")]` 类型
2. 将所有类型组合为 `Union[...]`
3. 创建 `ToolSelectionResponse` TypedDict：
   ```python
   class ToolSelectionResponse(TypedDict):
       tools: Annotated[list[selected_tool_type], Field(description="...")]
   ```
4. 返回 `TypeAdapter(ToolSelectionResponse)`

**设计意图**：使用 `Literal` 类型约束选择模型只能返回有效的工具名称，避免幻觉。每个 `Literal` 都带有工具描述，帮助选择模型理解每个工具的用途

**示例**：假设有 `get_weather` 和 `calculator` 两个工具，生成的 Schema 类似：

```python
tools: list[Union[
    Annotated[Literal["get_weather"], Field(description="Get weather for a location")],
    Annotated[Literal["calculator"], Field(description="Perform calculations")]
]]
```

#### 15.2 LLMToolSelectorMiddleware.__init__()

```python
def __init__(self, *, model=None, system_prompt=DEFAULT_SYSTEM_PROMPT,
             max_tools=None, always_include=None):
```

**关键参数**：
- `model`：选择模型，None 则使用主模型
- `max_tools`：最多选择几个工具
- `always_include`：始终包含的工具名称（不计入 max_tools）

#### 15.3 _prepare_selection_request()

```python
def _prepare_selection_request(self, request) -> _SelectionRequest | None:
```

**内部执行步骤**：

1. 如果没有工具，返回 None（不需要选择）
2. 过滤出 `BaseTool` 实例（排除 Provider 特定的工具字典）
3. 验证 `always_include` 中的工具是否存在
4. 分离"始终包含"和"待选择"的工具
5. 如果没有待选择的工具，返回 None
6. 找到最后一条 HumanMessage
7. 确定 model：`self.model or request.model`
8. 返回 `_SelectionRequest`

#### 15.4 _process_selection_response()

```python
def _process_selection_response(self, response, available_tools, valid_tool_names, request):
```

**内部执行步骤**：

1. 遍历响应中的工具名称列表
2. 验证每个名称是否在有效列表中
3. 应用 `max_tools` 限制
4. 去重
5. 合并"选中的工具" + "始终包含的工具" + "Provider 工具字典"
6. 返回 `request.override(tools=filtered_tools)`

#### 15.5 wrap_model_call() 核心方法

```python
def wrap_model_call(self, request, handler):
```

**内部执行步骤**：

1. 准备选择请求：`_prepare_selection_request(request)`
2. 如果不需要选择，直接调用 `handler(request)`
3. 创建动态 Schema：`_create_tool_selection_response(available_tools)`
4. 使用 `model.with_structured_output(schema)` 绑定结构化输出
5. 调用选择模型：
   ```python
   response = structured_model.invoke([
       {"role": "system", "content": system_message},
       last_user_message,
   ])
   ```
6. 处理选择响应：`_process_selection_response(response, ...)`
7. 使用过滤后的工具调用主模型：`handler(modified_request)`

**完整流程**：

```
用户消息 + 50 个工具
    ↓
选择模型（小模型）→ 筛选出 5 个最相关的工具
    ↓
主模型（大模型）→ 使用 5 个工具生成响应
```

---

## 16. 核心设计模式总结

### 16.1 持久化会话模式（Shell）

```
before_agent → 创建 Shell 会话
    ↓
工具调用 → 写入命令 → 读取输出 → 返回结果
    ↓
工具调用 → 写入命令 → 读取输出 → 返回结果（同一会话）
    ↓
after_agent → 清理 Shell 会话
```

- Shell 进程在整个 Agent 生命周期内保持运行
- 命令之间共享工作目录、环境变量和历史
- 使用 `weakref.finalize` 确保资源清理

### 16.2 Command 模式（Todo）

```python
Command(update={
    "todos": todos,                                          # 更新 todos 状态通道
    "messages": [ToolMessage(...)],                          # 更新 messages 通道
})
```

- 一个工具调用同时更新多个状态通道
- LangGraph 的 `Command` 是比简单返回值更强大的状态更新机制

### 16.3 双引擎搜索模式（FileSearch）

```
grep_search
    ├── 优先：ripgrep（快速，C 实现）
    └── 回退：Python regex（兼容性好）
```

- ripgrep 不可用时自动回退
- 两种引擎返回相同格式的结果

### 16.4 两阶段模型调用模式（ToolSelection）

```
第一阶段：小模型 + 结构化输出 → 选择工具
第二阶段：大模型 + 选中的工具 → 生成响应
```

- 减少主模型的 Token 消耗
- 提高工具选择的准确性
- 使用 `Literal` 类型约束防止幻觉

### 16.5 闭包工具模式（Shell/FileSearch）

```python
@tool(name, args_schema=schema, description=desc)
def shell_tool(*, runtime, command, restart):
    resources = self._get_or_create_resources(runtime.state)
    return self._run_shell_tool(resources, ...)
```

- 工具函数作为闭包捕获 `self`（中间件实例）
- 通过 `ToolRuntime` 访问 Agent 状态
- 工具在中间件 `__init__` 中创建，注册到 `self.tools`
