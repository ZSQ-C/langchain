# 核心中间件深度详解

> 涵盖文件：`human_in_the_loop.py`、`pii.py`、`_redaction.py`、`tool_retry.py`、`model_retry.py`、`model_fallback.py`、`_retry.py`

---

## 第一部分：human_in_the_loop.py

---

### 1. 文件定位

| 维度 | 说明 |
|------|------|
| **所属模块** | `langchain.agents.middleware` |
| **核心职责** | 实现人工介入（Human-in-the-Loop）中间件，允许人类审批、编辑、拒绝或代替工具执行 |
| **解决什么问题** | 在生产环境中，某些工具调用（如删除数据、发送邮件、执行支付）需要人工确认后才能执行。本中间件在模型返回工具调用后、工具执行前插入中断点，等待人类决策 |
| **核心机制** | 使用 LangGraph 的 `interrupt()` 函数暂停执行，等待人类输入后恢复 |

---

### 2. 代码结构总览

```
human_in_the_loop.py
├── TypedDict 类型定义
│   ├── Action                  # 动作（name + args）
│   ├── ActionRequest           # 动作请求（name + args + description）
│   ├── ReviewConfig            # 审批配置（action_name + allowed_decisions + args_schema）
│   ├── HITLRequest             # 人工审批请求（action_requests + review_configs）
│   ├── ApproveDecision         # 批准决策
│   ├── EditDecision            # 编辑决策（含 edited_action）
│   ├── RejectDecision          # 拒绝决策（含 message）
│   ├── RespondDecision         # 代替工具响应决策（含 message）
│   ├── HITLResponse            # 人工审批响应（decisions 列表）
│   └── InterruptOnConfig       # 中断配置（allowed_decisions + description + args_schema）
├── Protocol
│   └── _DescriptionFactory     # 描述生成器协议
├── 类型别名
│   └── DecisionType            # Literal["approve", "edit", "reject", "respond"]
│   └── Decision                # 四种决策的联合类型
└── HumanInTheLoopMiddleware    # 核心中间件类
    ├── __init__()              # 初始化，解析 interrupt_on 配置
    ├── _create_action_and_config()  # 创建 ActionRequest 和 ReviewConfig
    ├── _process_decision()     # 处理单个决策
    └── after_model()           # 核心钩子：模型返回后触发中断
```

---

### 3. 逐段深度详解

---

#### 3.1 类型定义体系

##### Action

```python
class Action(TypedDict):
    name: str
    args: dict[str, Any]
```

- 最基础的动作表示：名称 + 参数
- 用于 `EditDecision` 中表示编辑后的动作

##### ActionRequest

```python
class ActionRequest(TypedDict):
    name: str
    args: dict[str, Any]
    description: NotRequired[str]
```

- 在 Action 基础上增加了可选的 `description` 字段
- 这是发送给人类审阅者的请求格式

##### ReviewConfig

```python
class ReviewConfig(TypedDict):
    action_name: str
    allowed_decisions: list[DecisionType]
    args_schema: NotRequired[dict[str, Any]]
```

- 每个动作的审批策略配置
- `allowed_decisions`：限制人类可以做出的决策类型
- `args_schema`：如果允许编辑，提供参数的 JSON Schema

##### DecisionType 与四种决策

```python
DecisionType = Literal["approve", "edit", "reject", "respond"]
```

| 决策类型 | 含义 | 效果 |
|----------|------|------|
| `approve` | 批准执行 | 原样传递工具调用 |
| `edit` | 编辑后执行 | 替换工具调用的 name/args |
| `reject` | 拒绝执行 | 返回错误 ToolMessage，工具不执行 |
| `respond` | 代替工具响应 | 跳过工具执行，直接返回人类提供的消息 |

##### RespondDecision 的特殊设计

```python
class RespondDecision(TypedDict):
    type: Literal["respond"]
    message: str
```

- **设计意图**：用于"询问用户"类工具，工具的真实实现就是人类的回答
- 工具不执行，而是创建一个 `status="success"` 的合成 ToolMessage

##### InterruptOnConfig

```python
class InterruptOnConfig(TypedDict):
    allowed_decisions: list[DecisionType]
    description: NotRequired[str | _DescriptionFactory]
    args_schema: NotRequired[dict[str, Any]]
```

- 用户配置格式，`description` 支持两种形式：
  - 静态字符串
  - 动态生成函数 `(tool_call, state, runtime) -> str`

---

#### 3.2 HumanInTheLoopMiddleware

##### `__init__` 方法详解

```python
def __init__(
    self,
    interrupt_on: dict[str, bool | InterruptOnConfig],
    *,
    description_prefix: str = "Tool execution requires approval",
) -> None:
```

**入参**：
- `interrupt_on`：工具名到配置的映射
  - `True`：允许所有四种决策
  - `False`：自动批准（不中断）
  - `InterruptOnConfig`：自定义决策配置
- `description_prefix`：默认描述前缀

**内部执行步骤**：

1. 遍历 `interrupt_on` 字典
2. 如果值是 `True`，创建包含所有四种决策的 `InterruptOnConfig`
3. 如果值是 `InterruptOnConfig` 且 `allowed_decisions` 非空，直接使用
4. `False` 值被跳过（不加入 `resolved_configs`）
5. 存储到 `self.interrupt_on`

##### `_create_action_and_config` 方法详解

```python
def _create_action_and_config(
    self, tool_call: ToolCall, config: InterruptOnConfig,
    state: AgentState[Any], runtime: Runtime[ContextT],
) -> tuple[ActionRequest, ReviewConfig]:
```

**内部执行步骤**：

1. 提取 `tool_name` 和 `tool_args`
2. 生成描述：
   - 如果 `description` 是可调用的，调用 `description(tool_call, state, runtime)` 动态生成
   - 如果 `description` 是字符串，直接使用
   - 否则使用默认格式：`"{description_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"`
3. 创建 `ActionRequest(name, args, description)`
4. 创建 `ReviewConfig(action_name, allowed_decisions)`

##### `_process_decision` 静态方法详解

```python
@staticmethod
def _process_decision(
    decision: Decision, tool_call: ToolCall, config: InterruptOnConfig,
) -> tuple[ToolCall | None, ToolMessage | None]:
```

**返回值**：`(修订后的工具调用, 人工工具消息)`

**四种决策的处理逻辑**：

| 决策 | 修订后工具调用 | 人工消息 | 说明 |
|------|---------------|----------|------|
| approve | 原始 tool_call | None | 原样通过 |
| edit | 新 ToolCall（name/args 替换，id 保留） | None | 编辑后执行 |
| reject | 原始 tool_call | ToolMessage(status="error") | 返回拒绝消息 |
| respond | 原始 tool_call | ToolMessage(status="success") | 人类代替工具响应 |

**关键细节**：
- `reject` 返回的 ToolMessage `status="error"`，模型会看到错误消息并可能调整策略
- `respond` 返回的 ToolMessage `status="success"`，模型认为工具成功执行
- 如果决策类型不在 `allowed_decisions` 中，抛出 `ValueError`

##### `after_model` 核心钩子详解

```python
def after_model(
    self, state: AgentState[Any], runtime: Runtime[ContextT]
) -> dict[str, Any] | None:
```

**执行时机**：模型返回 AIMessage 后、工具执行前

**内部执行步骤**：

1. **获取最后一条 AIMessage**：从消息列表末尾向前搜索
2. **如果没有工具调用**：返回 None（不需要中断）
3. **遍历工具调用**：检查每个工具调用是否需要中断
   - 如果工具名在 `self.interrupt_on` 中，创建 ActionRequest 和 ReviewConfig
   - 记录需要中断的工具调用索引
4. **如果没有需要中断的工具调用**：返回 None
5. **创建 HITLRequest**：包含所有 action_requests 和 review_configs
6. **调用 `interrupt(hitl_request)`**：暂停执行，等待人类输入
   - 这是 LangGraph 的核心中断机制
   - 返回的 `decisions` 是人类做出的决策列表
7. **验证决策数量**：决策数必须等于中断的工具调用数
8. **处理决策并重建工具调用列表**：
   - 遍历所有工具调用
   - 对于被中断的工具调用，处理对应决策
   - 对于未被中断的工具调用，原样保留
   - 收集所有人工 ToolMessage
9. **更新 AIMessage 的 tool_calls**：只保留批准/编辑后的工具调用
10. **返回更新后的消息**：`{"messages": [last_ai_msg, *artificial_tool_messages]}`

**关键设计**：
- 一次中断处理所有需要审批的工具调用（批量审批）
- 未被中断的工具调用自动批准
- 人工 ToolMessage 直接注入消息流，工具不会实际执行

---

## 第二部分：pii.py + _redaction.py

---

### 4. _redaction.py 文件定位

| 维度 | 说明 |
|------|------|
| **核心职责** | PII 检测和脱敏的底层工具库 |
| **解决什么问题** | 提供 PII 检测器（email/credit_card/ip/mac_address/url）、脱敏策略（block/redact/mask/hash）和规则解析基础设施 |

---

### 5. _redaction.py 代码结构总览

```
_redaction.py
├── 类型定义
│   ├── RedactionStrategy       # 脱敏策略字面量类型
│   ├── PIIMatch                # PII 匹配结果 TypedDict
│   ├── PIIDetectionError       # 检测阻断异常
│   └── Detector                # 检测器函数签名
├── 内置检测器
│   ├── detect_email()          # 邮箱检测
│   ├── detect_credit_card()    # 信用卡号检测（含 Luhn 校验）
│   ├── detect_ip()             # IP 地址检测
│   ├── detect_mac_address()    # MAC 地址检测
│   ├── detect_url()            # URL 检测
│   └── BUILTIN_DETECTORS       # 内置检测器注册表
├── 脱敏策略实现
│   ├── _apply_redact_strategy()  # 替换为 [REDACTED_TYPE]
│   ├── _apply_mask_strategy()    # 部分遮蔽
│   ├── _apply_hash_strategy()    # 哈希替换
│   └── apply_strategy()          # 策略分发入口
├── 工具函数
│   ├── _passes_luhn()           # Luhn 校验算法
│   └── resolve_detector()       # 检测器解析
└── 数据类
    ├── RedactionRule             # 脱敏规则配置
    └── ResolvedRedactionRule     # 解析后的脱敏规则
```

---

### 6. _redaction.py 逐段深度详解

---

#### 6.1 PIIMatch

```python
class PIIMatch(TypedDict):
    type: str
    value: str
    start: int
    end: int
```

- `type`：PII 类型名（如 "email"、"credit_card"）
- `value`：匹配到的原始文本
- `start`/`end`：在原文中的字符位置

#### 6.2 PIIDetectionError

```python
class PIIDetectionError(Exception):
    def __init__(self, pii_type: str, matches: Sequence[PIIMatch]) -> None:
        self.pii_type = pii_type
        self.matches = list(matches)
```

- 当策略为 `block` 时抛出
- 携带 PII 类型和所有匹配项

#### 6.3 内置检测器详解

##### detect_email

```python
pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
```

- 使用正则匹配标准邮箱格式
- `\b` 单词边界防止部分匹配

##### detect_credit_card

```python
pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
```

- 匹配 16 位信用卡号（支持空格/短横线分隔）
- **关键**：通过 Luhn 校验过滤误报

##### _passes_luhn 算法

```python
def _passes_luhn(card_number: str) -> bool:
    digits = [int(d) for d in card_number if d.isdigit()]
    if not _CARD_NUMBER_MIN_DIGITS <= len(digits) <= _CARD_NUMBER_MAX_DIGITS:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        value = digit
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0
```

- Luhn 算法步骤：
  1. 提取所有数字
  2. 检查位数在 13-19 之间
  3. 从右向左，偶数位数字乘以 2
  4. 如果乘积大于 9，减去 9
  5. 所有数字求和，模 10 为 0 则有效

##### detect_ip

```python
ipv4_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
```

- 正则匹配后，使用 `ipaddress.ip_address()` 验证
- 自动过滤无效 IP（如 999.999.999.999）

##### detect_url

- 两种模式：
  1. 带协议的 URL：`https?://...`
  2. 裸 URL：`www.example.com` 或 `example.com/path`
- 使用 `urlparse()` 验证
- 裸 URL 只接受有路径或以 `www.` 开头的（减少误报）
- 去重：避免同一 URL 被两种模式重复匹配

#### 6.4 脱敏策略详解

##### redact 策略

```python
replacement = f"[REDACTED_{match['type'].upper()}]"
```

- 完全替换为 `[REDACTED_EMAIL]`、`[REDACTED_CREDIT_CARD]` 等占位符
- **关键**：从后向前替换（`reverse=True`），避免位置偏移

##### mask 策略

按 PII 类型定制遮蔽格式：

| PII 类型 | 遮蔽格式 | 示例 |
|----------|----------|------|
| email | `user@****.domain` | `john@****.com` |
| credit_card | `****-****-****-1234` | 保留最后 4 位 |
| ip | `*.*.*.octet` | `*.*.*.1` |
| mac_address | `**:**:**:**:**:ff` | 保留最后 2 位 |
| url | `[MASKED_URL]` | 完全遮蔽 |
| 其他 | `****5678` | 保留最后 4 位 |

##### hash 策略

```python
digest = hashlib.sha256(match["value"].encode()).hexdigest()[:8]
replacement = f"<{match['type']}_hash:{digest}>"
```

- SHA-256 哈希，取前 8 位十六进制
- 格式：`<email_hash:a1b2c3d4>`
- **保留身份标识**：同一值始终产生相同哈希，适合分析和调试

##### block 策略

```python
raise PIIDetectionError(matches[0]["type"], matches)
```

- 直接抛出异常，终止执行

#### 6.5 resolve_detector()

```python
def resolve_detector(pii_type: str, detector: Detector | str | None) -> Detector:
```

**三种输入的处理**：

1. `detector is None`：从 `BUILTIN_DETECTORS` 查找内置检测器
2. `detector is str`：编译为正则，返回 `regex_detector` 函数
3. `detector is Callable`：包装为 `_normalizing_detector`，统一输出格式
   - 兼容 `{"text": ...}` 和 `{"value": ...}` 两种键名
   - 自动补充 `type` 字段

#### 6.6 RedactionRule 与 ResolvedRedactionRule

```python
@dataclass(frozen=True)
class RedactionRule:
    pii_type: str
    strategy: RedactionStrategy = "redact"
    detector: Detector | str | None = None

    def resolve(self) -> ResolvedRedactionRule:
        resolved_detector = resolve_detector(self.pii_type, self.detector)
        return ResolvedRedactionRule(...)

@dataclass(frozen=True)
class ResolvedRedactionRule:
    pii_type: str
    strategy: RedactionStrategy
    detector: Detector

    def apply(self, content: str) -> tuple[str, list[PIIMatch]]:
        matches = self.detector(content)
        if not matches:
            return content, []
        updated = apply_strategy(content, matches, self.strategy)
        return updated, matches
```

- `RedactionRule`：用户配置（detector 可能是字符串或 None）
- `ResolvedRedactionRule`：运行时配置（detector 已解析为可调用函数）
- `resolve()` 方法完成从配置到运行时的转换
- `apply()` 方法：检测 + 脱敏一步到位

---

### 7. pii.py 文件定位

| 维度 | 说明 |
|------|------|
| **核心职责** | PII 中间件实现，在 Agent 对话的多个层面检测和处理 PII |
| **独特能力** | 不仅修改图状态中的消息，还通过 `_PIIStreamTransformer` 实时脱敏流式输出 |

---

### 8. pii.py 代码结构总览

```
pii.py
├── _PIIStreamTransformer       # 流式输出脱敏转换器
│   ├── __init__()              # 初始化缓冲区和规则
│   ├── process()               # 事件分发入口
│   ├── _process_messages_event()   # 处理 messages 通道事件
│   ├── _process_tools_event()      # 处理 tools 通道事件
│   ├── _process_values_event()     # 处理 values 通道事件
│   ├── _mutate_delta()             # 处理流式文本增量
│   ├── _mutate_string_field_delta() # 带 lookback 的字符串字段脱敏
│   ├── _mutate_tool_call_chunk_delta() # 工具调用参数脱敏
│   ├── _finalize_block()           # 内容块完成时脱敏
│   ├── _redact_value()             # 递归脱敏任意值
│   ├── _redact_base_message()      # 脱敏 BaseMessage
│   └── _redact_tool_call_list()    # 脱敏工具调用列表
└── PIIMiddleware                # PII 中间件
    ├── __init__()              # 初始化规则和流转换器
    ├── _process_content()      # 内容脱敏
    ├── before_model()          # 模型调用前检查输入
    ├── abefore_model()         # 异步版本
    ├── after_model()           # 模型调用后检查输出
    └── aafter_model()          # 异步版本
```

---

### 9. pii.py 逐段深度详解

---

#### 9.1 _PIIStreamTransformer

**核心问题**：流式输出中，PII 可能跨越多个 delta 分片。例如邮箱 `user@example.com` 可能被分成 `user@ex` 和 `ample.com` 两个 delta。如果逐个 delta 检测，会漏掉跨分片的 PII。

**解决方案**：Lookback 缓冲区机制

```
_DEFAULT_STREAM_LOOKBACK = 128
```

- 每个 (run_id, content_block_index) 维护一个缓冲区
- 缓冲区保留最近 128 个字符
- 新 delta 到达时，与缓冲区拼接后检测
- 检测完成后，安全前缀立即发送，尾部 128 字符保留在缓冲区
- 内容块完成时（`content-block-finish`），对完整快照重新检测

##### `__init__` 方法

```python
def __init__(self, scope, *, rule, lookback=128):
    self._rule = rule
    self._lookback = lookback
    self._buffers: dict[tuple[str, int], str] = {}      # 文本缓冲区
    self._tool_buffers: dict[str, str] = {}              # 工具输出缓冲区
```

- `before_builtins = True`：在 LangGraph 内置流转换器之前运行，确保下游消费者看到的是脱敏后的文本
- `required_stream_modes = ("messages", "tools", "values")`：需要监听三个流通道

##### `_mutate_string_field_delta` 核心方法

```python
def _mutate_string_field_delta(self, delta, payload, run_id, field):
```

**内部执行步骤**：

1. 从 delta 中提取文本字段（`text` 或 `reasoning`）
2. 获取缓冲区键 `(run_id, index)`
3. 拼接缓冲区内容和新文本：`combined = held + text`
4. 对拼接后的完整文本运行 PII 检测
5. 如果发现 PII，应用脱敏策略（`block` 策略直接抛出 `PIIDetectionError`）
6. 计算安全发送边界：`emit_end = max(0, len(combined) - self._lookback)`
7. 保留尾部：`self._buffers[key] = combined[emit_end:]`
8. 发送安全前缀：`delta[field] = combined[:emit_end]`

**关键设计**：
- 检测在完整缓冲区上运行（不是仅检测即将发送的前缀），避免跨边界 PII 漏检
- `block` 策略在流式层面直接失败，不需要等到 `after_model` 钩子

##### `_process_tools_event` 方法

处理 `tools` 通道的四种事件：

| 事件 | 处理方式 |
|------|----------|
| `tool-started` | 脱敏 `input` 字段 |
| `tool-output-delta` | 使用 `_tool_buffers` 做 lookback 脱敏 |
| `tool-finished` | 脱敏 `output` 字段，清理缓冲区 |
| `tool-error` | 脱敏 `message` 字段，清理缓冲区 |

##### `_redact_value` 递归脱敏

```python
def _redact_value(self, value: Any) -> Any:
```

- `str` → 检测并脱敏
- `BaseMessage` → 调用 `_redact_base_message`
- `dict` → 递归脱敏每个值
- `list` → 递归脱敏每个元素
- `tuple` → 递归脱敏每个元素
- 其他类型 → 原样返回

##### `_redact_base_message` 方法

脱敏 BaseMessage 的两个 PII 表面：
1. `.content`（字符串或内容块列表）
2. `.tool_calls[*].args` 和 `.invalid_tool_calls[*].args`

使用 `model_copy(update=...)` 创建新副本，原始对象保持不变

---

#### 9.2 PIIMiddleware

##### `__init__` 方法详解

```python
def __init__(
    self,
    pii_type: Literal["email", "credit_card", "ip", "mac_address", "url"] | str,
    *,
    strategy: Literal["block", "redact", "mask", "hash"] = "redact",
    detector: Callable[[str], list[PIIMatch]] | str | None = None,
    apply_to_input: bool = True,
    apply_to_output: bool = False,
    apply_to_tool_results: bool = False,
) -> None:
```

**入参解析**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pii_type` | 必填 | PII 类型名或自定义名称 |
| `strategy` | `"redact"` | 脱敏策略 |
| `detector` | None | 自定义检测器或正则 |
| `apply_to_input` | True | 检查用户输入 |
| `apply_to_output` | False | 检查 AI 输出 |
| `apply_to_tool_results` | False | 检查工具结果 |

**内部执行步骤**：

1. 调用 `RedactionRule(pii_type, strategy, detector).resolve()` 解析规则
2. 存储解析后的 `pii_type`、`strategy`、`detector`
3. **如果 `apply_to_output` 或 `apply_to_tool_results` 为 True**：
   - 创建 `_PIIStreamTransformer` 工厂函数
   - 存储到 `self.transformers`，factory.py 会将其注册为流转换器

##### `name` 属性

```python
@property
def name(self) -> str:
    return f"{self.__class__.__name__}[{self.pii_type}]"
```

- 包含 PII 类型名，便于区分多个 PIIMiddleware 实例

##### `before_model` 钩子详解

```python
@hook_config(can_jump_to=["end"])
def before_model(self, state, runtime):
```

**执行时机**：模型调用前

**内部执行步骤**：

1. 如果 `apply_to_input` 和 `apply_to_tool_results` 都为 False，返回 None
2. **处理用户输入**（`apply_to_input`）：
   - 找到最后一条 HumanMessage
   - 检测并脱敏内容
   - 如果有修改，创建新的 HumanMessage 替换
3. **处理工具结果**（`apply_to_tool_results`）：
   - 找到最后一条 AIMessage
   - 处理其后所有 ToolMessage
   - 检测并脱敏内容
   - 如果有修改，创建新的 ToolMessage 替换
4. 如果有任何修改，返回 `{"messages": new_messages}`
5. `@hook_config(can_jump_to=["end"])`：`block` 策略下可以跳转到结束节点

##### `after_model` 钩子详解

```python
@hook_config(can_jump_to=["end"])
def after_model(self, state, runtime):
```

- 只处理 `apply_to_output`
- 找到最后一条 AIMessage
- 检测并脱敏 `.content`
- 创建新的 AIMessage 替换（保留 `tool_calls`）

---

## 第三部分：_retry.py + tool_retry.py + model_retry.py + model_fallback.py

---

### 10. _retry.py 文件定位

| 维度 | 说明 |
|------|------|
| **核心职责** | 重试工具函数库，被 `tool_retry.py` 和 `model_retry.py` 共享 |
| **解决什么问题** | 避免重试逻辑的代码重复，统一参数验证、异常判断和延迟计算 |

---

### 11. _retry.py 逐段深度详解

#### 11.1 类型别名

```python
RetryOn = tuple[type[Exception], ...] | Callable[[Exception], bool]
OnFailure = Literal["error", "continue"] | Callable[[Exception], str]
```

- `RetryOn`：指定哪些异常触发重试
  - 元组形式：基于 `isinstance` 检查
  - 可调用形式：自定义判断逻辑
- `OnFailure`：重试耗尽后的行为
  - `"error"`：重新抛出异常
  - `"continue"`：注入错误消息，让 Agent 继续
  - 可调用形式：自定义错误消息

#### 11.2 validate_retry_params()

```python
def validate_retry_params(max_retries, initial_delay, max_delay, backoff_factor):
```

- 验证所有参数非负
- `max_retries >= 0`
- `initial_delay >= 0`
- `max_delay >= 0`
- `backoff_factor >= 0`

#### 11.3 should_retry_exception()

```python
def should_retry_exception(exc: Exception, retry_on: RetryOn) -> bool:
```

- 如果 `retry_on` 是可调用的，调用 `retry_on(exc)` 返回布尔值
- 如果 `retry_on` 是元组，返回 `isinstance(exc, retry_on)`

#### 11.4 calculate_delay()

```python
def calculate_delay(retry_number, *, backoff_factor, initial_delay, max_delay, jitter):
```

**指数退避算法**：

1. 如果 `backoff_factor == 0.0`：恒定延迟 `delay = initial_delay`
2. 否则：指数增长 `delay = initial_delay * (backoff_factor ** retry_number)`
3. 上限截断：`delay = min(delay, max_delay)`
4. 抖动（如果启用）：`delay += random.uniform(-0.25*delay, 0.25*delay)`
5. 确保非负：`delay = max(0, delay)`

**示例**：`initial_delay=1.0, backoff_factor=2.0`

| 重试次数 | 延迟 |
|----------|------|
| 0 | 1.0s |
| 1 | 2.0s |
| 2 | 4.0s |
| 3 | 8.0s |

---

### 12. tool_retry.py 文件定位

| 维度 | 说明 |
|------|------|
| **核心职责** | 工具调用重试中间件，在工具执行失败时自动重试 |
| **使用钩子** | `wrap_tool_call` / `awrap_tool_call`（洋葱模型包裹） |

---

### 13. tool_retry.py 逐段深度详解

#### 13.1 ToolRetryMiddleware.__init__()

```python
def __init__(
    self,
    *,
    max_retries: int = 2,
    tools: list[BaseTool | str] | None = None,
    retry_on: RetryOn = (Exception,),
    on_failure: OnFailure = "continue",
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> None:
```

**关键参数**：
- `tools`：可选的工具过滤列表，None 表示应用于所有工具
- `on_failure`：支持向后兼容的 `"return_message"` → `"continue"` 和 `"raise"` → `"error"`

**内部执行步骤**：

1. 调用 `validate_retry_params()` 验证参数
2. 处理废弃的 `on_failure` 值，发出 `DeprecationWarning`
3. 从 `tools` 列表中提取工具名称（支持 BaseTool 实例和字符串）
4. 设置 `self.tools = []`（不注册额外工具）

#### 13.2 wrap_tool_call 核心方法

```python
def wrap_tool_call(self, request, handler):
```

**内部执行步骤**：

1. 提取工具名称
2. 检查是否应该对该工具重试（`_should_retry_tool`）
3. 如果不需要，直接调用 `handler(request)` 透传
4. 进入重试循环 `for attempt in range(self.max_retries + 1)`：
   a. 尝试调用 `handler(request)`
   b. 成功则直接返回
   c. 捕获异常后：
      - 检查是否应该重试该异常（`should_retry_exception`）
      - 如果不应该重试，立即调用 `_handle_failure`
      - 如果还有重试次数，计算延迟并 `time.sleep()`
      - 如果重试耗尽，调用 `_handle_failure`

#### 13.3 _handle_failure 方法

```python
def _handle_failure(self, tool_name, tool_call_id, exc, attempts_made):
```

- `on_failure == "error"`：重新抛出异常
- `on_failure` 是可调用的：调用它获取自定义错误消息
- 否则：使用 `_format_failure_message` 生成默认消息
- 返回 `ToolMessage(content=..., status="error")`

---

### 14. model_retry.py 文件定位

| 维度 | 说明 |
|------|------|
| **核心职责** | 模型调用重试中间件，在模型调用失败时自动重试 |
| **使用钩子** | `wrap_model_call` / `awrap_model_call`（洋葱模型包裹） |
| **与 ToolRetry 的区别** | 失败时返回 `AIMessage`（而非 `ToolMessage`），因为模型调用的上下文不同 |

---

### 15. model_retry.py 逐段深度详解

#### 15.1 ModelRetryMiddleware.__init__()

与 ToolRetryMiddleware 几乎相同，但没有 `tools` 过滤参数（模型调用不需要过滤）

#### 15.2 wrap_model_call 核心方法

与 ToolRetryMiddleware 的 `wrap_tool_call` 逻辑一致，区别在于：
- 使用 `time.sleep()` 而非 `asyncio.sleep()`
- 失败时调用 `_handle_failure` 返回 `ModelResponse(result=[AIMessage(...)])`

#### 15.3 _handle_failure 方法

```python
def _handle_failure(self, exc, attempts_made):
```

- `on_failure == "error"`：重新抛出异常
- `on_failure` 是可调用的：调用它获取自定义消息，包装为 `AIMessage`
- 否则：使用 `_format_failure_message` 生成 `AIMessage`
- 返回 `ModelResponse(result=[ai_msg])`

---

### 16. model_fallback.py 文件定位

| 维度 | 说明 |
|------|------|
| **核心职责** | 模型回退中间件，当主模型失败时依次尝试备用模型 |
| **使用钩子** | `wrap_model_call` / `awrap_model_call` |
| **与 ModelRetry 的区别** | ModelRetry 重试同一模型，ModelFallback 切换到不同模型 |

---

### 17. model_fallback.py 逐段深度详解

#### 17.1 ModelFallbackMiddleware.__init__()

```python
def __init__(self, first_model: str | BaseChatModel, *additional_models: str | BaseChatModel):
```

- 接受一个或多个备用模型
- 字符串形式通过 `init_chat_model()` 初始化
- `BaseChatModel` 实例直接使用

#### 17.2 wrap_model_call 核心方法

```python
def wrap_model_call(self, request, handler):
```

**内部执行步骤**：

1. **尝试主模型**：`handler(request)`
2. 如果失败，保存异常
3. **依次尝试备用模型**：
   - 使用 `request.override(model=fallback_model)` 替换模型
   - 调用 `handler(overridden_request)`
4. 如果所有模型都失败，抛出最后一个异常

**关键设计**：
- `request.override(model=...)` 利用 ModelRequest 的不可变模式，创建新请求而非修改原请求
- 主模型不在 `self.models` 列表中（它由 `create_agent` 的 `model` 参数指定），fallback 列表是额外的

---

## 18. 核心设计模式总结

### 18.1 中断-恢复模式（HITL）

```
模型返回工具调用 → after_model 检测需要审批的工具
    → interrupt(hitl_request) 暂停执行
    → 人类做出决策
    → 恢复执行，处理决策
    → 修订后的工具调用继续执行
```

### 18.2 双层脱敏模式（PII）

```
状态层（before_model / after_model）：修改图状态中的消息
    ↕ 独立运作
流式层（_PIIStreamTransformer）：实时脱敏流式输出
```

- 两层独立运作，互不依赖
- 状态层是"权威"脱敏，流式层是"实时"脱敏
- 流式层使用 lookback 缓冲区解决跨 delta 的 PII 检测问题

### 18.3 洋葱模型重试模式

```
ToolRetryMiddleware.wrap_tool_call():
    for attempt in range(max_retries + 1):
        try:
            return handler(request)  # 调用下一个中间件或实际工具
        except Exception:
            if should_retry and has_retries_left:
                sleep(delay)
                continue
            else:
                return handle_failure()
```

### 18.4 级联回退模式

```
ModelFallbackMiddleware.wrap_model_call():
    try: handler(request)                    # 主模型
    except: handler(request.override(model=fallback1))  # 备用1
    except: handler(request.override(model=fallback2))  # 备用2
    except: raise last_exception             # 全部失败
```
