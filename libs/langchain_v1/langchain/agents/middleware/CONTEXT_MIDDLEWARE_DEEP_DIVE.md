# 上下文管理中间件深度详解

> 涵盖文件：`summarization.py`、`context_editing.py`

---

## 第一部分：summarization.py

---

### 1. 文件定位

| 维度 | 说明 |
|------|------|
| **所属模块** | `langchain.agents.middleware` |
| **核心职责** | 对话摘要中间件，当对话历史接近 Token 限制时自动摘要旧消息，保留近期消息 |
| **解决什么问题** | LLM 有上下文窗口限制，长对话会超出限制导致失败。本中间件在 Token 数达到阈值时，自动将旧消息摘要为一条 HumanMessage，释放 Token 空间 |
| **核心机制** | `before_model` 钩子 + LangGraph 的 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 全量替换机制 |

---

### 2. 代码结构总览

```
summarization.py
├── 常量
│   ├── DEFAULT_SUMMARY_PROMPT       # 默认摘要提示词
│   ├── _DEFAULT_MESSAGES_TO_KEEP    # 默认保留消息数
│   ├── _DEFAULT_TRIM_TOKEN_LIMIT    # 摘要调用最大 Token
│   └── _DEFAULT_FALLBACK_MESSAGE_COUNT  # 摘要失败时保留的消息数
├── 类型定义
│   ├── ContextFraction              # ("fraction", float) 分数形式
│   ├── ContextTokens                # ("tokens", int) 绝对 Token 数
│   ├── ContextMessages              # ("messages", int) 绝对消息数
│   ├── ContextSize                  # 三种形式的联合类型
│   └── TokenCounter                 # Token 计数函数签名
├── 工具函数
│   ├── _get_approximate_token_counter()  # 根据模型类型调优近似计数器
│   └── _provider_matches()               # Provider 名称匹配（含别名）
└── SummarizationMiddleware           # 核心中间件类
    ├── __init__()                    # 初始化触发条件、保留策略、摘要模型
    ├── before_model()                # 核心钩子
    ├── abefore_model()               # 异步版本
    ├── _should_summarize()           # 判断是否需要摘要
    ├── _should_summarize_based_on_reported_tokens()  # 基于 Provider 报告的 Token 判断
    ├── _determine_cutoff_index()     # 确定截断点
    ├── _find_token_based_cutoff()    # 基于 Token 数的二分查找截断点
    ├── _find_safe_cutoff()           # 安全截断点（保持 AI/Tool 消息对完整）
    ├── _find_safe_cutoff_point()     # 静态方法：查找不拆分消息对的截断点
    ├── _create_summary()             # 同步生成摘要
    ├── _acreate_summary()            # 异步生成摘要
    ├── _trim_messages_for_summary()  # 裁剪待摘要消息
    ├── _partition_messages()         # 分割消息为待摘要和保留两部分
    ├── _build_new_messages()         # 构建摘要消息
    ├── _ensure_message_ids()         # 确保所有消息有 ID
    ├── _validate_context_size()      # 验证上下文配置
    └── _get_profile_limits()         # 获取模型 Profile 中的 Token 限制
```

---

### 3. 逐段深度详解

---

#### 3.1 DEFAULT_SUMMARY_PROMPT

这是一个精心设计的摘要提示词，包含以下结构化部分：

| 部分 | 作用 |
|------|------|
| `<role>` | 角色定义：上下文提取助手 |
| `<primary_objective>` | 核心目标：提取最高质量/最相关的上下文 |
| `<objective_information>` | 背景说明：即将达到 Token 限制，必须提取最重要的信息 |
| `<instructions>` | 输出格式要求：SESSION INTENT / SUMMARY / ARTIFACTS / NEXT STEPS |

**设计意图**：不是简单的"总结对话"，而是结构化提取——确保关键信息（会话意图、已完成的操作、创建的文件、下一步计划）不会在摘要中丢失

#### 3.2 ContextSize 类型体系

```python
ContextFraction = tuple[Literal["fraction"], float]
ContextTokens = tuple[Literal["tokens"], int]
ContextMessages = tuple[Literal["messages"], int]
ContextSize = ContextFraction | ContextTokens | ContextMessages
```

- **设计模式**：带标签的联合类型（Tagged Union / Discriminated Union）
- 第一个元素是标签，第二个元素是值
- 用于 `trigger`（何时触发摘要）和 `keep`（保留多少上下文）两个参数

**示例**：
```python
trigger=[("fraction", 0.8), ("messages", 100)]  # 80% Token 或 100 条消息时触发
keep=("messages", 20)                             # 保留最近 20 条消息
```

#### 3.3 _get_approximate_token_counter()

```python
def _get_approximate_token_counter(model: BaseChatModel) -> TokenCounter:
    if model._llm_type.startswith("anthropic-chat"):
        return partial(count_tokens_approximately, use_usage_metadata_scaling=True, chars_per_token=3.3)
    return partial(count_tokens_approximately, use_usage_metadata_scaling=True)
```

- Anthropic 模型使用 `chars_per_token=3.3`（通过离线实验校准）
- 其他模型使用默认的 `chars_per_token` 值
- `use_usage_metadata_scaling=True`：利用模型报告的 Token 使用量进行缩放

#### 3.4 _provider_matches()

```python
_LS_PROVIDER_ALIASES: dict[str, frozenset[str]] = {
    "amazon_bedrock": frozenset({"bedrock", "bedrock_converse"}),
}
```

- 解决 Provider 名称不一致问题
- 例如：LangSmith 中记录为 `amazon_bedrock`，但模型返回的 `model_provider` 可能是 `bedrock`

---

#### 3.5 SummarizationMiddleware.__init__()

```python
def __init__(
    self,
    model: str | BaseChatModel,
    *,
    trigger: ContextSize | list[ContextSize] | None = None,
    keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
    token_counter: TokenCounter = count_tokens_approximately,
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
    trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
    **deprecated_kwargs: Any,
) -> None:
```

**入参解析**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | 必填 | 用于生成摘要的模型 |
| `trigger` | None | 触发条件（None 表示不自动触发） |
| `keep` | `("messages", 20)` | 保留策略 |
| `token_counter` | 近似计数 | Token 计数函数 |
| `summary_prompt` | 默认模板 | 摘要提示词 |
| `trim_tokens_to_summarize` | 4000 | 摘要调用最大 Token |

**内部执行步骤**：

1. **处理废弃参数**：`max_tokens_before_summary` → `trigger=("tokens", value)`，`messages_to_keep` → `keep=("messages", value)`
2. **初始化摘要模型**：字符串形式通过 `init_chat_model()` 转换
3. **验证触发条件**：调用 `_validate_context_size()` 验证每个条件
4. **调优 Token 计数器**：如果使用默认计数器，根据模型类型调优
5. **检查 Profile 依赖**：如果使用了 `fraction` 类型，检查模型是否有 Profile 信息

---

#### 3.6 before_model() 核心钩子

```python
def before_model(self, state, runtime) -> dict[str, Any] | None:
```

**执行时机**：模型调用前

**内部执行步骤**：

1. **确保消息有 ID**：`_ensure_message_ids(messages)` — LangGraph 的 `add_messages` reducer 需要 ID 来标识消息
2. **计算总 Token 数**：`total_tokens = self.token_counter(messages)`
3. **判断是否需要摘要**：`_should_summarize(messages, total_tokens)`
4. **确定截断点**：`_determine_cutoff_index(messages)`
5. **分割消息**：`_partition_messages(messages, cutoff_index)`
6. **生成摘要**：`_create_summary(messages_to_summarize)`
7. **构建新消息列表**：
   ```python
   [
       RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 删除所有旧消息
       *new_messages,                           # 摘要消息
       *preserved_messages,                     # 保留的近期消息
   ]
   ```
8. 返回 `{"messages": [...]}`

**关键设计**：使用 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 一次性删除所有旧消息，然后添加摘要 + 保留消息。这比逐条删除更高效

---

#### 3.7 _should_summarize()

```python
def _should_summarize(self, messages, total_tokens) -> bool:
```

**判断逻辑**：遍历所有触发条件，任一满足即返回 True

| 触发条件类型 | 判断方式 |
|-------------|----------|
| `("messages", N)` | `len(messages) >= N` |
| `("tokens", N)` | `total_tokens >= N` 或 Provider 报告的 Token >= N |
| `("fraction", F)` | `total_tokens >= max_input_tokens * F` 或 Provider 报告的 Token >= 阈值 |

**双重检测机制**：
1. 近似 Token 计数（快速但不精确）
2. Provider 报告的 Token（精确但需要上一轮模型返回 `usage_metadata`）

---

#### 3.8 _should_summarize_based_on_reported_tokens()

```python
def _should_summarize_based_on_reported_tokens(self, messages, threshold) -> bool:
```

**内部执行步骤**：

1. 找到最后一条 AIMessage
2. 检查 `usage_metadata.total_tokens` 是否 >= threshold
3. 检查 `response_metadata.model_provider` 是否与当前模型匹配（通过 `_provider_matches`）
4. 三个条件都满足才返回 True

**设计意图**：Provider 报告的 Token 数是最精确的，但需要确认是同一个 Provider 报告的（避免用 OpenAI 的 Token 数来判断 Anthropic 的限制）

---

#### 3.9 _find_token_based_cutoff() — 二分查找

```python
def _find_token_based_cutoff(self, messages) -> int | None:
```

**算法**：二分查找找到最早的消息索引，使得从该索引到末尾的消息 Token 数 <= 目标值

**内部执行步骤**：

1. 计算目标 Token 数（根据 `keep` 的类型）
2. 如果总 Token 数 <= 目标值，返回 0（不需要截断）
3. **二分查找**：
   - `left = 0, right = len(messages)`
   - `mid = (left + right) // 2`
   - 如果 `messages[mid:]` 的 Token 数 <= 目标值，`right = mid, cutoff_candidate = mid`
   - 否则 `left = mid + 1`
4. 调用 `_find_safe_cutoff_point()` 确保不拆分 AI/Tool 消息对

---

#### 3.10 _find_safe_cutoff_point() — 消息对完整性保护

```python
@staticmethod
def _find_safe_cutoff_point(messages, cutoff_index) -> int:
```

**核心问题**：如果截断点落在 ToolMessage 上，会导致 ToolMessage 没有对应的 AIMessage（包含 tool_calls）

**解决步骤**：

1. 检查 `messages[cutoff_index]` 是否为 ToolMessage
2. 如果不是，直接返回 cutoff_index（安全）
3. 如果是，收集从 cutoff_index 开始的所有连续 ToolMessage 的 `tool_call_id`
4. 向前搜索找到包含这些 `tool_call_id` 的 AIMessage
5. 将截断点调整到该 AIMessage 的位置（包含它）
6. 如果找不到对应的 AIMessage（边界情况），前进到 ToolMessage 序列之后

---

#### 3.11 _create_summary()

```python
def _create_summary(self, messages_to_summarize) -> str:
```

**内部执行步骤**：

1. 如果没有待摘要消息，返回 `"No previous conversation history."`
2. 裁剪消息：`_trim_messages_for_summary()` — 限制摘要调用的 Token 数
3. 格式化消息：`get_buffer_string(trimmed_messages)` — 避免元数据导致的 Token 膨胀
4. 调用摘要模型：`self.model.invoke(self.summary_prompt.format(messages=formatted_messages))`
5. 添加 `lc_source: "summarization"` 元数据
6. 如果摘要失败，返回错误消息而非抛出异常

---

#### 3.12 _trim_messages_for_summary()

```python
def _trim_messages_for_summary(self, messages) -> list[AnyMessage]:
```

- 使用 `trim_messages()` 裁剪待摘要的消息
- 参数：`max_tokens=4000`、`start_on="human"`、`strategy="last"`、`allow_partial=True`
- 如果裁剪失败，回退到保留最后 15 条消息

---

## 第二部分：context_editing.py

---

### 4. 文件定位

| 维度 | 说明 |
|------|------|
| **所属模块** | `langchain.agents.middleware` |
| **核心职责** | 上下文编辑中间件，当对话超出 Token 阈值时自动清理旧的工具结果 |
| **解决什么问题** | 与 SummarizationMiddleware 不同，本中间件不生成摘要，而是直接清除旧的工具输出（替换为占位符），保留近期工具结果 |
| **灵感来源** | Anthropic 的 `clear_tool_uses_20250919` 行为 |

---

### 5. 代码结构总览

```
context_editing.py
├── 类型定义
│   ├── TokenCounter              # Token 计数函数签名
│   └── ContextEdit               # 编辑策略协议
├── ClearToolUsesEdit             # 清除工具输出策略
│   ├── __init__()                # 配置触发阈值、保留数量等
│   ├── apply()                   # 执行清除逻辑
│   └── _build_cleared_tool_input_message()  # 清除工具输入参数
└── ContextEditingMiddleware      # 核心中间件
    ├── __init__()                # 初始化编辑策略列表
    ├── wrap_model_call()         # 洋葱模型钩子
    └── awrap_model_call()        # 异步版本
```

---

### 6. 逐段深度详解

---

#### 6.1 ContextEdit 协议

```python
class ContextEdit(Protocol):
    def apply(self, messages: list[AnyMessage], *, count_tokens: TokenCounter) -> None:
        ...
```

- 协议类型，定义编辑策略的接口
- `apply()` 方法直接修改 `messages` 列表（原地修改）
- 接受 `count_tokens` 函数用于判断是否需要触发编辑

---

#### 6.2 ClearToolUsesEdit

```python
@dataclass(slots=True)
class ClearToolUsesEdit(ContextEdit):
    trigger: int = 100_000
    clear_at_least: int = 0
    keep: int = 3
    clear_tool_inputs: bool = False
    exclude_tools: Sequence[str] = ()
    placeholder: str = DEFAULT_TOOL_PLACEHOLDER  # "[cleared]"
```

**参数解析**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trigger` | 100,000 | Token 阈值，超过时触发清除 |
| `clear_at_least` | 0 | 最少回收的 Token 数 |
| `keep` | 3 | 保留最近 N 个工具结果 |
| `clear_tool_inputs` | False | 是否同时清除 AIMessage 中的工具调用参数 |
| `exclude_tools` | () | 排除的工具名称列表 |
| `placeholder` | `"[cleared]"` | 清除后的占位文本 |

##### apply() 方法详解

```python
def apply(self, messages, *, count_tokens) -> None:
```

**内部执行步骤**：

1. **检查是否需要触发**：`tokens = count_tokens(messages)`，如果 <= trigger，直接返回
2. **收集候选 ToolMessage**：
   - 找到所有 ToolMessage
   - 排除最近 `keep` 个
   - 剩余的是清除候选
3. **遍历候选并清除**：
   - 跳过已经清除过的（`response_metadata.context_editing.cleared == True`）
   - 找到对应的 AIMessage（向前搜索）
   - 找到对应的 tool_call（通过 `tool_call_id` 匹配）
   - 检查工具名是否在排除列表中
   - **清除工具输出**：替换 ToolMessage 的 content 为 placeholder，设置 `artifact=None`
   - **可选清除工具输入**：将 AIMessage 中对应 tool_call 的 args 设为空字典 `{}`
   - 在 `response_metadata` 中标记 `context_editing.cleared = True`
4. **如果设置了 `clear_at_least`**：清除后重新计算 Token 数，如果已回收足够则停止

##### _build_cleared_tool_input_message() 静态方法

```python
@staticmethod
def _build_cleared_tool_input_message(message, tool_call_id) -> AIMessage:
```

**内部执行步骤**：

1. 遍历 `message.tool_calls`，找到 `id == tool_call_id` 的调用
2. 将其 `args` 设为空字典 `{}`
3. 在 `response_metadata.context_editing.cleared_tool_inputs` 中记录已清除的 tool_call_id
4. 使用 `model_copy(update=...)` 创建新的 AIMessage

**设计意图**：清除工具输入参数可以进一步节省 Token，但也可能导致模型丢失上下文。默认不开启

---

#### 6.3 ContextEditingMiddleware

```python
class ContextEditingMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
```

##### __init__() 方法

```python
def __init__(
    self,
    *,
    edits: Iterable[ContextEdit] | None = None,
    token_count_method: Literal["approximate", "model"] = "approximate",
) -> None:
```

- `edits`：编辑策略列表，默认为单个 `ClearToolUsesEdit()`
- `token_count_method`：Token 计数方式
  - `"approximate"`：使用 `count_tokens_approximately()`（快速但不精确）
  - `"model"`：使用 `model.get_num_tokens_from_messages()`（精确但可能较慢）

##### wrap_model_call() 核心方法

```python
def wrap_model_call(self, request, handler):
```

**内部执行步骤**：

1. 如果没有消息，直接调用 handler
2. 根据 `token_count_method` 创建 Token 计数函数
3. **深拷贝消息列表**：`edited_messages = deepcopy(list(request.messages))`
   - 关键：必须深拷贝，因为 `edit.apply()` 会原地修改消息
4. **依次应用所有编辑策略**：`for edit in self.edits: edit.apply(edited_messages, count_tokens=count_tokens)`
5. 使用 `request.override(messages=edited_messages)` 创建新请求
6. 调用 `handler(overridden_request)`

**关键设计**：
- 使用 `wrap_model_call`（洋葱模型）而非 `before_model`（状态钩子）
- 这意味着编辑只影响发送给模型的消息，不修改图状态中的原始消息
- 模型看到的是清理后的上下文，但原始消息仍然完整保留在状态中
- 这与 SummarizationMiddleware 的 `before_model` 方式不同——后者直接修改状态

---

## 7. 两种上下文管理策略的对比

| 维度 | SummarizationMiddleware | ContextEditingMiddleware |
|------|------------------------|--------------------------|
| **策略** | 生成摘要替换旧消息 | 清除工具输出占位符替换 |
| **信息保留** | 通过摘要保留关键信息 | 直接丢弃，只保留占位符 |
| **使用钩子** | `before_model`（修改状态） | `wrap_model_call`（洋葱模型，不修改状态） |
| **Token 节省** | 较多（摘要远短于原文） | 较少（占位符仍有开销） |
| **适用场景** | 长对话、需要保留历史上下文 | 工具密集型对话、工具输出很长 |
| **对模型的影响** | 模型看到摘要，知道有历史 | 模型看到 `[cleared]`，知道输出被清除 |
| **可组合性** | 可与 ContextEditing 组合使用 | 可与 Summarization 组合使用 |

---

## 8. 核心设计模式总结

### 8.1 标签联合类型（Tagged Union）

`ContextSize` 使用 `("fraction", 0.8)` / `("tokens", 3000)` / `("messages", 50)` 的标签联合类型，而非三个独立参数。优点：
- 类型安全：编译时检查标签和值的对应关系
- 可扩展：未来添加新类型只需新增标签
- 自文档化：标签即含义

### 8.2 二分查找优化

`_find_token_based_cutoff()` 使用二分查找确定截断点，时间复杂度 O(n log n)，优于线性搜索的 O(n²)

### 8.3 消息对完整性保护

两种中间件都确保 AI/Tool 消息对不被拆分：
- SummarizationMiddleware：`_find_safe_cutoff_point()` 向前调整截断点
- ContextEditingMiddleware：`ClearToolUsesEdit.apply()` 通过 `tool_call_id` 关联 AIMessage 和 ToolMessage

### 8.4 双重 Token 检测

SummarizationMiddleware 同时使用：
1. 近似 Token 计数（快速预判）
2. Provider 报告的 Token 数（精确确认）

这种双重检测在保证精度的同时不牺牲性能

### 8.5 原地修改 vs 不可变替换

- ContextEditingMiddleware：深拷贝后原地修改，通过 `request.override()` 传递
- SummarizationMiddleware：直接修改状态，通过 `RemoveMessage` + 新消息替换

两种方式各有优劣，ContextEditing 的方式更安全（不影响状态），Summarization 的方式更彻底（真正释放 Token）
