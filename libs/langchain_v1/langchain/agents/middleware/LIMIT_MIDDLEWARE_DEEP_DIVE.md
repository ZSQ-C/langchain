# 限制中间件深度详解

> 涵盖文件：`tool_call_limit.py`、`model_call_limit.py`

---

## 第一部分：tool_call_limit.py

---

### 1. 文件定位

| 维度 | 说明 |
|------|------|
| **所属模块** | `langchain.agents.middleware` |
| **核心职责** | 工具调用限制中间件，跟踪工具调用次数并在超限时采取行动 |
| **解决什么问题** | 防止 Agent 陷入无限循环（反复调用同一工具）、控制 API 调用成本、限制单个工具的使用频率 |
| **独特能力** | 支持线程级（跨运行持久化）和运行级（单次运行）两种计数维度，支持按工具名过滤 |

---

### 2. 代码结构总览

```
tool_call_limit.py
├── 类型定义
│   └── ExitBehavior              # 退出行为字面量类型
├── 状态扩展
│   └── ToolCallLimitState        # 扩展 AgentState，增加计数字段
├── 工具函数
│   ├── _build_tool_message_content()    # 构建给模型的错误消息
│   └── _build_final_ai_message_content() # 构建给用户的最终消息
├── 异常类
│   └── ToolCallLimitExceededError       # 超限异常
└── ToolCallLimitMiddleware       # 核心中间件
    ├── __init__()                # 初始化限制配置
    ├── name 属性                  # 包含工具名的中间件名称
    ├── _would_exceed_limit()     # 判断是否超限
    ├── _matches_tool_filter()    # 判断工具是否匹配过滤器
    ├── _separate_tool_calls()    # 分离允许/阻止的工具调用
    ├── after_model()             # 核心钩子
    └── aafter_model()            # 异步版本
```

---

### 3. 逐段深度详解

---

#### 3.1 ExitBehavior 类型

```python
ExitBehavior = Literal["continue", "error", "end"]
```

| 行为 | 说明 |
|------|------|
| `"continue"` | 阻止超限的工具调用（返回错误 ToolMessage），其他工具继续执行 |
| `"error"` | 抛出 `ToolCallLimitExceededError` 异常 |
| `"end"` | 立即跳转到结束节点，注入 ToolMessage + AIMessage |

---

#### 3.2 ToolCallLimitState

```python
class ToolCallLimitState(AgentState[ResponseT]):
    thread_tool_call_count: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    run_tool_call_count: NotRequired[Annotated[dict[str, int], UntrackedValue, PrivateStateAttr]]
```

**两个字段的区别**：

| 字段 | 生命周期 | 可见性 | 用途 |
|------|----------|--------|------|
| `thread_tool_call_count` | 跨运行持久化 | `PrivateStateAttr`（不暴露给模型） | 线程级限制 |
| `run_tool_call_count` | 单次运行 | `UntrackedValue` + `PrivateStateAttr` | 运行级限制 |

**为什么用字典而非整数？**：
- 字典的键是工具名（如 `"search"`、`"calculator"`）
- 特殊键 `"__all__"` 用于全局计数
- 支持多个 `ToolCallLimitMiddleware` 实例独立跟踪不同工具

**`UntrackedValue` 的作用**：
- `run_tool_call_count` 使用 `UntrackedValue` 标注，意味着 LangGraph 不会在 checkpoint 中持久化这个值
- 每次新运行时，`run_tool_call_count` 从 0 开始
- 而 `thread_tool_call_count` 会跨运行累加

---

#### 3.3 _build_tool_message_content()

```python
def _build_tool_message_content(tool_name: str | None) -> str:
    if tool_name:
        return f"Tool call limit exceeded. Do not call '{tool_name}' again."
    return "Tool call limit exceeded. Do not make additional tool calls."
```

**设计意图**：这是发送给模型的错误消息，不应该包含 `thread/run` 等概念（模型不理解这些），只需告诉模型"不要再调用了"

---

#### 3.4 _build_final_ai_message_content()

```python
def _build_final_ai_message_content(
    thread_count, run_count, thread_limit, run_limit, tool_name
) -> str:
```

**与 `_build_tool_message_content` 的区别**：这是发送给用户的消息，包含详细的限制信息

**输出示例**：
```
'search' tool call limit reached: thread limit exceeded (21/20 calls) and run limit exceeded (11/10 calls).
```

---

#### 3.5 ToolCallLimitExceededError

```python
class ToolCallLimitExceededError(Exception):
    def __init__(self, thread_count, run_count, thread_limit, run_limit, tool_name=None):
        ...
```

- 携带完整的计数信息，便于调用方做精细化处理
- 错误消息使用 `_build_final_ai_message_content()` 生成

---

#### 3.6 ToolCallLimitMiddleware.__init__()

```python
def __init__(
    self,
    *,
    tool_name: str | None = None,
    thread_limit: int | None = None,
    run_limit: int | None = None,
    exit_behavior: ExitBehavior = "continue",
) -> None:
```

**入参解析**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tool_name` | None | 限制特定工具，None 则限制所有工具 |
| `thread_limit` | None | 线程级限制 |
| `run_limit` | None | 运行级限制 |
| `exit_behavior` | `"continue"` | 超限行为 |

**验证逻辑**：

1. 至少指定一个限制：`thread_limit is None and run_limit is None` → `ValueError`
2. `exit_behavior` 必须是 `"continue"` / `"error"` / `"end"` 之一
3. `run_limit` 不能超过 `thread_limit`（逻辑约束：单次运行不能超过线程总量）

---

#### 3.7 name 属性

```python
@property
def name(self) -> str:
    base_name = self.__class__.__name__
    if self.tool_name:
        return f"{base_name}[{self.tool_name}]"
    return base_name
```

- 包含工具名，如 `ToolCallLimitMiddleware[search]`
- 允许多个实例跟踪不同工具

---

#### 3.8 _would_exceed_limit()

```python
def _would_exceed_limit(self, thread_count, run_count) -> bool:
    return (self.thread_limit is not None and thread_count + 1 > self.thread_limit) or (
        self.run_limit is not None and run_count + 1 > self.run_limit
    )
```

- 检查"再加一次调用"是否会超限
- 两个限制是 OR 关系：任一超限即返回 True

---

#### 3.9 _matches_tool_filter()

```python
def _matches_tool_filter(self, tool_call) -> bool:
    return self.tool_name is None or tool_call["name"] == self.tool_name
```

- `tool_name is None`：匹配所有工具（全局限制）
- 否则只匹配指定工具

---

#### 3.10 _separate_tool_calls()

```python
def _separate_tool_calls(
    self, tool_calls, thread_count, run_count
) -> tuple[list[ToolCall], list[ToolCall], int, int]:
```

**内部执行步骤**：

1. 初始化 `allowed_calls` 和 `blocked_calls`
2. 遍历工具调用：
   - 如果不匹配过滤器，跳过
   - 如果 `_would_exceed_limit()`，加入 `blocked_calls`
   - 否则加入 `allowed_calls`，递增计数
3. 返回 `(allowed_calls, blocked_calls, new_thread_count, new_run_count)`

**关键设计**：按顺序处理工具调用，先到先得。如果限制为 5，第 6 个调用会被阻止

---

#### 3.11 after_model() 核心钩子

```python
@hook_config(can_jump_to=["end"])
def after_model(self, state, runtime) -> dict[str, Any] | None:
```

**执行时机**：模型返回 AIMessage 后

**内部执行步骤**：

1. **获取最后一条 AIMessage**：从消息列表末尾向前搜索
2. **如果没有工具调用**：返回 None
3. **确定计数键**：`count_key = self.tool_name or "__all__"`
4. **获取当前计数**：从 `thread_tool_call_count` 和 `run_tool_call_count` 中读取
5. **分离工具调用**：`_separate_tool_calls()` → `(allowed, blocked, new_thread, new_run)`
6. **更新计数**：
   - `thread_counts[count_key] = new_thread_count`（只计算允许的调用）
   - `run_counts[count_key] = new_run_count + len(blocked_calls)`（阻止的调用也计入运行级计数）
7. **如果没有阻止的调用**：只返回计数更新
8. **处理阻止的调用**：

##### `"error"` 行为

```python
raise ToolCallLimitExceededError(...)
```

- 直接抛出异常，终止执行

##### `"continue"` 行为

```python
artificial_messages = [
    ToolMessage(
        content=tool_msg_content,
        tool_call_id=tool_call["id"],
        name=tool_call.get("name"),
        status="error",
    )
    for tool_call in blocked_calls
]
return {
    "thread_tool_call_count": thread_counts,
    "run_tool_call_count": run_counts,
    "messages": artificial_messages,
}
```

- 为每个被阻止的工具调用创建错误 ToolMessage
- 允许的工具调用正常执行
- 模型收到错误消息后可能会调整策略

##### `"end"` 行为

```python
# 检查是否有其他工具的调用
other_tools = [tc for tc in last_ai_message.tool_calls
               if self.tool_name is not None and tc["name"] != self.tool_name]
if other_tools:
    raise NotImplementedError("Cannot end execution with other tool calls pending.")

# 添加最终 AIMessage
artificial_messages.append(AIMessage(content=final_msg_content))

return {
    "thread_tool_call_count": thread_counts,
    "run_tool_call_count": run_counts,
    "jump_to": "end",
    "messages": artificial_messages,
}
```

- `"end"` 行为的限制：如果有其他工具的并行调用，抛出 `NotImplementedError`
- 原因：`jump_to="end"` 会跳过所有后续处理，其他工具的调用会被丢弃
- 注入最终 AIMessage 告诉用户限制已超

**`@hook_config(can_jump_to=["end"])`**：声明此钩子可能返回 `jump_to="end"`，factory.py 会据此配置状态图的边

---

## 第二部分：model_call_limit.py

---

### 4. 文件定位

| 维度 | 说明 |
|------|------|
| **所属模块** | `langchain.agents.middleware` |
| **核心职责** | 模型调用限制中间件，跟踪模型调用次数并在超限时采取行动 |
| **解决什么问题** | 控制模型调用成本、防止无限循环、限制单次运行的最大推理步数 |
| **与 ToolCallLimit 的区别** | ToolCallLimit 跟踪工具调用次数，ModelCallLimit 跟踪模型调用次数 |

---

### 5. 代码结构总览

```
model_call_limit.py
├── 状态扩展
│   └── ModelCallLimitState       # 扩展 AgentState，增加计数字段
├── 工具函数
│   └── _build_limit_exceeded_message()  # 构建超限消息
├── 异常类
│   └── ModelCallLimitExceededError      # 超限异常
└── ModelCallLimitMiddleware      # 核心中间件
    ├── __init__()                # 初始化限制配置
    ├── before_model()            # 模型调用前检查限制
    ├── abefore_model()           # 异步版本
    ├── after_model()             # 模型调用后递增计数
    └── aafter_model()            # 异步版本
```

---

### 6. 逐段深度详解

---

#### 6.1 ModelCallLimitState

```python
class ModelCallLimitState(AgentState[ResponseT]):
    thread_model_call_count: NotRequired[Annotated[int, PrivateStateAttr]]
    run_model_call_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
```

**与 ToolCallLimitState 的区别**：
- 使用 `int` 而非 `dict[str, int]`——因为模型调用不需要按名称分类
- 同样区分线程级（持久化）和运行级（不持久化）

---

#### 6.2 ModelCallLimitMiddleware.__init__()

```python
def __init__(
    self,
    *,
    thread_limit: int | None = None,
    run_limit: int | None = None,
    exit_behavior: Literal["end", "error"] = "end",
) -> None:
```

**与 ToolCallLimitMiddleware 的区别**：

| 维度 | ToolCallLimit | ModelCallLimit |
|------|---------------|----------------|
| `tool_name` 过滤 | 支持 | 不支持 |
| `exit_behavior` | `"continue"` / `"error"` / `"end"` | `"end"` / `"error"` |
| 没有 `"continue"` | — | 模型调用无法"部分阻止" |

**为什么没有 `"continue"` 行为？**：
- 工具调用可以部分阻止（阻止某些工具，允许其他工具）
- 模型调用是原子的——要么调用，要么不调用，不存在"部分调用"
- 所以只有 `"end"`（停止执行）和 `"error"`（抛出异常）两种行为

---

#### 6.3 before_model() 核心钩子

```python
@hook_config(can_jump_to=["end"])
def before_model(self, state, runtime) -> dict[str, Any] | None:
```

**执行时机**：模型调用前

**内部执行步骤**：

1. 获取当前计数：
   ```python
   thread_count = state.get("thread_model_call_count", 0)
   run_count = state.get("run_model_call_count", 0)
   ```
2. 检查是否超限：
   ```python
   thread_limit_exceeded = self.thread_limit is not None and thread_count >= self.thread_limit
   run_limit_exceeded = self.run_limit is not None and run_count >= self.run_limit
   ```
3. 如果超限：
   - `"error"` → 抛出 `ModelCallLimitExceededError`
   - `"end"` → 返回 `{"jump_to": "end", "messages": [AIMessage(content=limit_message)]}`
4. 如果未超限，返回 None（允许模型调用）

**关键设计**：
- 检查在模型调用**之前**进行（预防性检查）
- 使用 `>=` 而非 `>`：因为计数在 `after_model` 中递增，`before_model` 检查时计数已经是上一次调用后的值

---

#### 6.4 after_model() 钩子

```python
def after_model(self, state, runtime) -> dict[str, Any] | None:
    return {
        "thread_model_call_count": state.get("thread_model_call_count", 0) + 1,
        "run_model_call_count": state.get("run_model_call_count", 0) + 1,
    }
```

- 模型调用成功后，递增两个计数器
- 无论模型是否返回工具调用，都递增（模型调用本身就算一次）

---

## 7. 两种限制中间件的对比

| 维度 | ToolCallLimitMiddleware | ModelCallLimitMiddleware |
|------|------------------------|--------------------------|
| **跟踪对象** | 工具调用次数 | 模型调用次数 |
| **计数粒度** | `dict[str, int]`（按工具名） | `int`（全局） |
| **工具过滤** | 支持 `tool_name` 参数 | 不支持 |
| **退出行为** | `"continue"` / `"error"` / `"end"` | `"end"` / `"error"` |
| **使用钩子** | `after_model` | `before_model` + `after_model` |
| **检查时机** | 模型返回后（检查工具调用） | 模型调用前（预防性检查） |
| **部分阻止** | 支持（阻止超限工具，允许其他工具） | 不支持（模型调用是原子的） |

---

## 8. 核心设计模式总结

### 8.1 双层计数模式

```
线程级计数（thread_*_count）
    ├── 跨运行持久化
    ├── 保存在 checkpoint 中
    └── 用于长期限制

运行级计数（run_*_count）
    ├── 单次运行有效
    ├── 不保存在 checkpoint 中（UntrackedValue）
    └── 用于单次运行限制
```

- `UntrackedValue` 是 LangGraph 的特殊通道标注，表示该值不会被持久化
- 每次新运行时，运行级计数从 0 开始

### 8.2 PrivateStateAttr 模式

```python
thread_tool_call_count: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
```

- `PrivateStateAttr` 标注表示该字段是中间件的私有状态
- 不会暴露给模型（不会出现在系统提示词或工具参数中）
- 只在中间件内部使用

### 8.3 渐进式退出策略

```
continue → 最温和：阻止超限调用，其他继续
    ↓
end → 中等：跳转到结束，注入最终消息
    ↓
error → 最严格：抛出异常，调用方必须处理
```

### 8.4 jump_to 状态跳转

```python
@hook_config(can_jump_to=["end"])
def after_model(self, state, runtime):
    return {"jump_to": "end", "messages": [...]}
```

- `jump_to="end"` 是 LangGraph 的状态图跳转机制
- factory.py 读取 `@hook_config(can_jump_to=["end"])`，在状态图中添加从当前节点到 `"end"` 节点的条件边
- 运行时，如果钩子返回 `jump_to="end"`，LangGraph 直接跳转到结束节点

### 8.5 预防性检查 vs 事后检查

- **ModelCallLimit**：`before_model` 预防性检查（调用前判断是否超限）
- **ToolCallLimit**：`after_model` 事后检查（模型返回后判断哪些工具调用超限）

原因：
- 模型调用是"请求-响应"模式，可以在请求前拦截
- 工具调用是模型决定的，只能在模型返回后才知道有哪些调用
