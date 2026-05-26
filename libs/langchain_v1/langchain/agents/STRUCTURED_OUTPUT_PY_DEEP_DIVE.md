# structured_output.py 深度详解

---

## 1. 文件定位

| 维度 | 说明 |
|------|------|
| **所属模块** | `langchain.agents` |
| **核心职责** | 定义 Agent 结构化输出的类型系统、策略模式与解析基础设施 |
| **解决什么问题** | Agent 不仅要返回自由文本，还需要返回符合特定 Schema 的结构化数据（如 Pydantic Model、dataclass、TypedDict、JSON Schema）。本文件提供了两种策略（Tool 策略 / Provider 策略）来实现结构化输出，并统一了 Schema 解析、验证和错误处理 |
| **与 factory.py 的关系** | `create_agent()` 接收 `response_format` 参数，内部调用本文件的 `ToolStrategy` / `ProviderStrategy` / `AutoStrategy` 来决定结构化输出的实现路径 |

---

## 2. 代码结构总览

```
structured_output.py
├── 类型变量与常量
│   ├── SchemaT                    # Schema 泛型类型变量
│   └── SchemaKind                 # Schema 类型字面量
├── 异常类
│   ├── StructuredOutputError      # 基类（携带 ai_message）
│   ├── MultipleStructuredOutputsError  # 多个结构化输出工具调用
│   └── StructuredOutputValidationError # 解析验证失败
├── 工具函数
│   └── _parse_with_schema()       # 统一解析入口
├── 数据类
│   ├── _SchemaSpec[SchemaT]       # Schema 规格描述
│   ├── ToolStrategy[SchemaT]      # 工具调用策略
│   ├── ProviderStrategy[SchemaT]  # Provider 原生策略
│   ├── OutputToolBinding[SchemaT] # 工具策略绑定信息
│   ├── ProviderStrategyBinding[SchemaT] # Provider 策略绑定信息
│   └── AutoStrategy[SchemaT]      # 自动选择策略
└── 类型别名
    └── ResponseFormat             # 三种策略的联合类型
```

---

## 3. 逐段深度详解

---

### 3.1 导入区

```python
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, is_dataclass
from types import UnionType
from typing import (
    TYPE_CHECKING, Any, Generic, Literal, TypeVar, Union, get_args, get_origin,
)
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, TypeAdapter
from typing_extensions import Self, is_typeddict
```

**逐行解析：**

- `from __future__ import annotations`：启用延迟注解求值（PEP 563），使得类型注解中的前向引用不需要引号包裹
- `import json`：用于 Provider 策略中从 AIMessage 文本内容解析 JSON
- `import uuid`：为没有名称的 Schema 生成唯一标识符
- `from dataclasses import dataclass, is_dataclass`：`dataclass` 装饰器用于定义数据类；`is_dataclass` 用于运行时判断一个类型是否是 dataclass
- `from types import UnionType`：Python 3.10+ 的 `X | Y` 语法产生的类型，用于检测 Union 类型
- `from typing import ...`：标准类型工具——`get_args` 获取泛型参数，`get_origin` 获取泛型原始类型
- `from langchain_core.tools import BaseTool, StructuredTool`：工具基类和结构化工具类
- `from pydantic import BaseModel, TypeAdapter`：`BaseModel` 是 Pydantic 模型基类；`TypeAdapter` 是 Pydantic V2 的通用验证器，可以验证任意类型（不仅限于 BaseModel）
- `from typing_extensions import Self, is_typeddict`：`Self` 用于类方法返回自身类型；`is_typeddict` 运行时判断是否为 TypedDict

---

### 3.2 类型变量与常量

```python
SchemaT = TypeVar("SchemaT")
SchemaKind = Literal["pydantic", "dataclass", "typeddict", "json_schema"]
```

**逐行解析：**

- `SchemaT`：泛型类型变量，代表用户提供的 Schema 类型（可以是 Pydantic Model、dataclass、TypedDict 或 dict）
- `SchemaKind`：字面量类型，用于分类 Schema 的具体类型，四种取值对应四种支持的 Schema 形式

---

### 3.3 异常类体系

#### 3.3.1 StructuredOutputError

```python
class StructuredOutputError(Exception):
    """Base class for structured output errors."""
    ai_message: AIMessage
```

**解析：**

- 这是所有结构化输出错误的基类
- 关键设计：携带 `ai_message` 属性，保存触发错误的原始 AI 消息。这是因为结构化输出的错误发生在模型返回之后，调用方可能需要访问原始 AI 消息来决定重试策略
- 注意：`ai_message` 是类属性声明（没有在 `__init__` 中赋值），子类负责在构造时设置它

#### 3.3.2 MultipleStructuredOutputsError

```python
class MultipleStructuredOutputsError(StructuredOutputError):
    def __init__(self, tool_names: list[str], ai_message: AIMessage) -> None:
        self.tool_names = tool_names
        self.ai_message = ai_message
        super().__init__(
            "Model incorrectly returned multiple structured responses "
            f"({', '.join(tool_names)}) when only one is expected."
        )
```

**逐行解析：**

- **触发场景**：Tool 策略期望模型只返回一个结构化输出工具调用，但模型返回了多个
- `tool_names`：所有触发结构化输出的工具名称列表
- `ai_message`：包含这些工具调用的原始 AI 消息
- 错误消息格式：`"Model incorrectly returned multiple structured responses (tool_a, tool_b) when only one is expected."`

#### 3.3.3 StructuredOutputValidationError

```python
class StructuredOutputValidationError(StructuredOutputError):
    def __init__(self, tool_name: str, source: Exception, ai_message: AIMessage) -> None:
        self.tool_name = tool_name
        self.source = source
        self.ai_message = ai_message
        super().__init__(f"Failed to parse structured output for tool '{tool_name}': {source}.")
```

**逐行解析：**

- **触发场景**：模型返回了结构化输出工具调用，但参数无法按 Schema 解析
- `tool_name`：解析失败的工具名称
- `source`：原始解析异常（如 Pydantic 的 `ValidationError`）
- `ai_message`：包含无效工具调用的原始 AI 消息
- 错误消息格式：`"Failed to parse structured output for tool 'MySchema': ..."`

---

### 3.4 _parse_with_schema()

```python
def _parse_with_schema(
    schema: type[SchemaT] | dict[str, Any], schema_kind: SchemaKind, data: dict[str, Any]
) -> Any:
```

**方法作用**：统一的 Schema 解析入口，根据 Schema 类型选择不同的解析策略

**入参**：
- `schema`：Schema 类型（Pydantic Model 类、dataclass 类、TypedDict 类）或 JSON Schema 字典
- `schema_kind`：Schema 类型分类
- `data`：待解析的字典数据

**出参**：按 Schema 类型解析后的实例

**内部执行步骤**：

1. **JSON Schema 快速返回**：如果 `schema_kind == "json_schema"`，直接返回原始 `data` 字典，不做任何解析——因为 JSON Schema 本身就是字典，不需要实例化
2. **TypeAdapter 统一解析**：对于 Pydantic / dataclass / TypedDict 三种类型，统一使用 `TypeAdapter(schema).validate_python(data)` 进行解析
   - `TypeAdapter` 是 Pydantic V2 的通用验证器，不仅能验证 `BaseModel`，还能验证 dataclass、TypedDict 等任意类型
   - `validate_python(data)` 接收 Python 字典，返回对应类型的实例
3. **异常处理**：如果解析失败，抛出 `ValueError`，包含 Schema 名称和原始错误信息

**设计意图**：将三种不同 Schema 类型的解析逻辑统一到一个函数中，通过 `TypeAdapter` 实现多态解析，避免为每种类型写单独的解析代码

---

### 3.5 _SchemaSpec[SchemaT]

```python
@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
```

**文件定位**：Schema 规格描述数据类，是连接用户提供的 Schema 和内部使用的 JSON Schema 的桥梁

**属性清单**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `schema` | `type[SchemaT] \| dict[str, Any]` | 原始 Schema 类型或 JSON Schema 字典 |
| `name` | `str` | Schema 名称，用于工具调用 |
| `description` | `str` | Schema 描述 |
| `schema_kind` | `SchemaKind` | Schema 类型分类 |
| `json_schema` | `dict[str, Any]` | 转换后的 JSON Schema 字典 |
| `strict` | `bool \| None` | 是否启用严格验证 |

#### `__init__` 方法详解

```python
def __init__(
    self,
    schema: type[SchemaT] | dict[str, Any],
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool | None = None,
) -> None:
```

**入参**：
- `schema`：用户提供的 Schema（必填）
- `name`：可选的自定义名称
- `description`：可选的自定义描述
- `strict`：是否严格验证

**内部执行步骤**：

1. **存储原始 Schema**：`self.schema = schema`

2. **确定名称**：
   - 如果提供了 `name`，直接使用
   - 如果 `schema` 是字典，取 `schema.get("title", ...)` 作为名称
   - 如果 `schema` 是类型，取 `schema.__name__` 作为名称
   - 兜底：生成 `response_format_{uuid4[:4]}` 格式的随机名称

3. **确定描述**：
   - 如果提供了 `description`，直接使用
   - 如果 `schema` 是字典，取 `schema.get("description", "")`
   - 如果 `schema` 是类型，取 `schema.__doc__` 或空字符串

4. **存储 strict 标志**：`self.strict = strict`

5. **分类 Schema 类型并生成 JSON Schema**：
   - **dict** → `schema_kind = "json_schema"`，`json_schema = schema`（直接使用）
   - **BaseModel 子类** → `schema_kind = "pydantic"`，`json_schema = schema.model_json_schema()`
   - **dataclass** → `schema_kind = "dataclass"`，`json_schema = TypeAdapter(schema).json_schema()`
   - **TypedDict** → `schema_kind = "typeddict"`，`json_schema = TypeAdapter(schema).json_schema()`
   - **其他** → 抛出 `ValueError`

**设计意图**：`_SchemaSpec` 是一个"Schema 适配器"，将不同形式的 Schema 统一转换为 JSON Schema + 元数据的形式，供后续的 Tool 策略和 Provider 策略使用

---

### 3.6 ToolStrategy[SchemaT]

```python
@dataclass(init=False)
class ToolStrategy(Generic[SchemaT]):
```

**文件定位**：工具调用策略——将结构化输出伪装成工具调用，让模型通过调用"虚拟工具"来返回结构化数据

**属性清单**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `schema` | `type[SchemaT] \| UnionType \| dict[str, Any]` | 原始 Schema（支持 Union 类型） |
| `schema_specs` | `list[_SchemaSpec[Any]]` | 展开后的 Schema 规格列表 |
| `tool_message_content` | `str \| None` | 返回给模型的人工工具消息内容 |
| `handle_errors` | 多种类型 | 错误处理策略 |

#### `__init__` 方法详解

```python
def __init__(
    self,
    schema: type[SchemaT] | UnionType | dict[str, Any],
    *,
    tool_message_content: str | None = None,
    handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str] = True,
) -> None:
```

**入参**：
- `schema`：支持 Union 类型（如 `ResponseA | ResponseB`），会展开为多个 Schema
- `tool_message_content`：当模型调用结构化输出工具后，返回的 ToolMessage 内容（默认 None 表示由系统自动生成）
- `handle_errors`：错误处理策略，支持 5 种形式

**内部执行步骤**：

1. **存储基本属性**：`self.schema`、`self.tool_message_content`、`self.handle_errors`

2. **定义 `_iter_variants` 内部函数**：递归展开 Union 和 JSON Schema oneOf
   - 如果 `get_origin(schema)` 是 `UnionType` 或 `Union`，递归展开每个参数
   - 如果 `schema` 是字典且包含 `"oneOf"` 键，递归展开每个子 Schema
   - 否则，yield 当前 schema 作为叶子节点

3. **生成 Schema 规格列表**：`self.schema_specs = [_SchemaSpec(s) for s in _iter_variants(schema)]`

**设计意图**：Tool 策略的核心思想是"把结构化输出伪装成工具调用"。模型不需要知道它在返回结构化数据，它只是"调用了一个工具"。这种方式的优点是兼容性极好——任何支持工具调用的模型都能用

---

### 3.7 ProviderStrategy[SchemaT]

```python
@dataclass(init=False)
class ProviderStrategy(Generic[SchemaT]):
```

**文件定位**：Provider 原生策略——利用模型提供商（如 OpenAI）的原生结构化输出能力

**属性清单**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `schema` | `type[SchemaT] \| dict[str, Any]` | 原始 Schema |
| `schema_spec` | `_SchemaSpec[SchemaT]` | Schema 规格描述 |

#### `__init__` 方法详解

```python
def __init__(
    self,
    schema: type[SchemaT] | dict[str, Any],
    *,
    strict: bool | None = None,
) -> None:
```

**入参**：
- `schema`：Schema 类型或 JSON Schema 字典
- `strict`：是否请求 Provider 端严格 Schema 强制

**内部执行步骤**：

1. 存储原始 Schema：`self.schema = schema`
2. 创建 Schema 规格描述：`self.schema_spec = _SchemaSpec(schema, strict=strict)`

#### `to_model_kwargs()` 方法详解

```python
def to_model_kwargs(self) -> dict[str, Any]:
```

**方法作用**：将 Provider 策略转换为绑定到模型的 kwargs

**内部执行步骤**：

1. 构建 `json_schema` 字典：
   - `name`：Schema 名称
   - `schema`：JSON Schema 字典
   - 如果 `strict` 为 True，添加 `"strict": True`
2. 构建 `response_format` 字典：
   - `type: "json_schema"`
   - `json_schema`：上一步构建的字典
3. 返回 `{"response_format": response_format}`

**设计意图**：这个方法生成的 kwargs 会传递给 `model.with_structured_output()` 或 `model.bind()`，让 Provider 在 API 层面强制输出符合 Schema 的 JSON。这是 OpenAI 的 Structured Outputs 功能的对接方式

---

### 3.8 OutputToolBinding[SchemaT]

```python
@dataclass
class OutputToolBinding(Generic[SchemaT]):
```

**文件定位**：工具策略的绑定信息——将 Schema、SchemaKind 和 LangChain Tool 实例关联起来

**属性清单**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `schema` | `type[SchemaT] \| dict[str, Any]` | 原始 Schema |
| `schema_kind` | `SchemaKind` | Schema 类型分类 |
| `tool` | `BaseTool` | 从 Schema 创建的 LangChain 工具实例 |

#### `from_schema_spec` 类方法详解

```python
@classmethod
def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
```

**方法作用**：从 `_SchemaSpec` 创建 `OutputToolBinding` 实例

**内部执行步骤**：

1. 使用 `StructuredTool` 创建工具实例：
   - `args_schema=schema_spec.json_schema`：工具的参数 Schema
   - `name=schema_spec.name`：工具名称
   - `description=schema_spec.description`：工具描述
2. 返回包含 schema、schema_kind、tool 三元组的实例

**设计意图**：`OutputToolBinding` 是 factory.py 中处理 Tool 策略结构化输出的核心数据结构。当 factory.py 检测到用户使用了 ToolStrategy，会为每个 schema_spec 创建一个 OutputToolBinding，将其中的 tool 注册到 Agent 的工具列表中

#### `parse()` 方法详解

```python
def parse(self, tool_args: dict[str, Any]) -> SchemaT:
```

**方法作用**：将工具调用参数按 Schema 解析为对应类型的实例

**内部执行步骤**：直接调用 `_parse_with_schema(self.schema, self.schema_kind, tool_args)`

---

### 3.9 ProviderStrategyBinding[SchemaT]

```python
@dataclass
class ProviderStrategyBinding(Generic[SchemaT]):
```

**文件定位**：Provider 策略的绑定信息——比 OutputToolBinding 更简单，不需要创建工具

**属性清单**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `schema` | `type[SchemaT] \| dict[str, Any]` | 原始 Schema |
| `schema_kind` | `SchemaKind` | Schema 类型分类 |

#### `from_schema_spec` 类方法详解

与 `OutputToolBinding.from_schema_spec` 类似，但不创建工具实例

#### `parse()` 方法详解

```python
def parse(self, response: AIMessage) -> SchemaT:
```

**方法作用**：从 AIMessage 中提取并解析结构化输出

**内部执行步骤**：

1. 调用 `_extract_text_content_from_message(response)` 提取文本内容
2. 使用 `json.loads(raw_text)` 解析 JSON
3. 调用 `_parse_with_schema(self.schema, self.schema_kind, data)` 解析为 Schema 实例

#### `_extract_text_content_from_message()` 静态方法详解

```python
@staticmethod
def _extract_text_content_from_message(message: AIMessage) -> str:
```

**方法作用**：从 AIMessage 中提取纯文本内容

**内部执行步骤**：

1. 如果 `message.content` 是字符串，直接返回
2. 如果 `message.content` 是列表（结构化内容块），遍历每个块：
   - 如果是字典且 `type == "text"`，提取 `text` 字段
   - 如果有 `content` 字符串键，提取 `content`
   - 否则，`str()` 转换
3. 拼接所有文本部分返回

**设计意图**：Provider 策略下，模型直接返回 JSON 文本（不是工具调用），需要从 AIMessage 的 content 中提取。content 可能是纯字符串，也可能是结构化内容块列表，此方法统一处理两种情况

---

### 3.10 AutoStrategy[SchemaT]

```python
class AutoStrategy(Generic[SchemaT]):
```

**文件定位**：自动策略——让 factory.py 根据模型能力自动选择最佳策略

**属性清单**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `schema` | `type[SchemaT] \| dict[str, Any]` | 原始 Schema |

**解析**：

- `AutoStrategy` 本身不包含任何策略逻辑，它只是一个"标记"
- factory.py 中的 `create_agent()` 检测到 `AutoStrategy` 后，会检查模型是否支持原生结构化输出（Provider 策略），如果支持则使用 `ProviderStrategy`，否则退回到 `ToolStrategy`

---

### 3.11 ResponseFormat 类型别名

```python
ResponseFormat = ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]
```

**解析**：三种策略的联合类型，作为 `create_agent()` 的 `response_format` 参数的类型注解

---

## 4. 核心设计模式总结

### 4.1 策略模式（Strategy Pattern）

三种结构化输出策略（Tool / Provider / Auto）是经典的策略模式：

| 策略 | 实现方式 | 适用场景 |
|------|----------|----------|
| `ToolStrategy` | 将 Schema 伪装成工具调用 | 任何支持工具调用的模型 |
| `ProviderStrategy` | 利用 Provider 原生能力 | OpenAI 等支持 Structured Outputs 的 Provider |
| `AutoStrategy` | 自动选择 | 用户不确定模型能力时 |

### 4.2 适配器模式（Adapter Pattern）

`_SchemaSpec` 是适配器模式的体现：将四种不同形式的 Schema（Pydantic / dataclass / TypedDict / JSON Schema dict）统一适配为 `json_schema + name + description + schema_kind` 的标准形式

### 4.3 Union 展开模式

`ToolStrategy` 的 `_iter_variants` 函数递归展开 Union 和 oneOf，支持 `ResponseA | ResponseB` 这样的联合类型。每个变体都会创建独立的 `_SchemaSpec` 和 `OutputToolBinding`，模型可以选择调用其中任意一个"工具"

### 4.4 错误层次体系

```
StructuredOutputError (基类，携带 ai_message)
├── MultipleStructuredOutputsError  # 多个结构化输出
└── StructuredOutputValidationError # 解析验证失败
```

这种设计让调用方可以：
- 统一捕获 `StructuredOutputError` 处理所有结构化输出错误
- 也可以针对特定错误类型做精细化处理
- 始终能通过 `error.ai_message` 访问原始 AI 消息

---

## 5. 完整执行流程

### 5.1 Tool 策略流程

```
用户提供 Schema (Pydantic/dataclass/TypedDict/dict)
    ↓
ToolStrategy.__init__() → _iter_variants() 展开Union → 生成 schema_specs
    ↓
factory.py 中为每个 schema_spec 创建 OutputToolBinding
    ↓
OutputToolBinding.from_schema_spec() → StructuredTool(name, description, args_schema)
    ↓
工具注册到 Agent → 模型"调用"该工具返回结构化数据
    ↓
OutputToolBinding.parse(tool_args) → _parse_with_schema() → 返回 Schema 实例
```

### 5.2 Provider 策略流程

```
用户提供 Schema
    ↓
ProviderStrategy.__init__() → _SchemaSpec(schema, strict)
    ↓
factory.py 中创建 ProviderStrategyBinding
    ↓
ProviderStrategy.to_model_kwargs() → {"response_format": {"type": "json_schema", ...}}
    ↓
model.with_structured_output() 或 model.bind() → 模型直接返回 JSON
    ↓
ProviderStrategyBinding.parse(response) → 提取文本 → json.loads → _parse_with_schema()
```

### 5.3 Auto 策略流程

```
用户提供 AutoStrategy(schema)
    ↓
factory.py 检测模型能力
    ↓
├── 支持 Provider 策略 → 转为 ProviderStrategy
└── 不支持 → 转为 ToolStrategy
```
