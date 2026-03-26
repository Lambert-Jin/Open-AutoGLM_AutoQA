# AutoQA

基于 VLM 的移动端自动化测试框架，用自然语言描述测试用例，自动在真机上执行操作并验证结果。

## feature
**自然语言驱动的测试描述** — 支持用自然语言描述测试场景，LLM 自动分解为操作步骤和断言，生成可执行的 YAML 测试用例；也支持交互式模式实时输入、规划、确认、执行。

**多模型协作架构** — 操作模型（AutoGLM）负责手机操作，VLM（Gemini）负责截图断言和页面理解，LLM 负责自然语言到测试步骤的规划，三个模型各司其职，通过统一 Provider 层（`ModelProvider` Protocol + 工厂函数）实现模型可插拔切换。

**跨页面上下文传递** — 采用步骤级上下文隔离避免历史信息干扰，通过 PageDescriber（VLM）在每步执行前分析截图提取关键信息，再由 ActionOptimizer（LLM）根据累积的全部历史步骤（指令 + 页面描述）以及当前页面描述，智能改写当前操作指令，将模糊指令（如"按照任务要求完成"）补充为具体指令（如"发一条30字以上的微头条"），解决跨页面语义丢失问题。

**Action 语义缓存** — Planner 阶段由 LLM 生成归一化 cache_key（如 `tap:comment_button`），Execution 阶段通过三层查找实现缓存命中：App 包名精确过滤 → 本地 sentence-transformers embedding 语义匹配（<5ms）→ pHash 视觉验证防 UI 变更误命中，重复操作跳过 API 调用，单次命中节省 2-3 秒延迟。



## 环境准备

### 1. 安装依赖

```bash
# Python >= 3.10
pip install -e .
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

```env
# Gemini（视觉断言 + 规划）
GEMINI_API_KEY=your-gemini-api-key

# AutoGLM（手机操作）— 智谱 BigModel API
AUTOGLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
AUTOGLM_API_KEY=your-autoglm-api-key
```

运行前加载环境变量：

```bash
export $(grep -v '^#' .env | xargs)
```

### 3. 配置 ADB（Android）

#### 安装 ADB

```bash
# macOS
brew install android-platform-tools

# Ubuntu / Debian
sudo apt install adb

# Windows
# 下载 SDK Platform Tools：https://developer.android.com/tools/releases/platform-tools
# 解压后将目录添加到系统 PATH
```

#### 手机端设置

1. 进入 **设置 → 关于手机**，连续点击「版本号」7 次，开启开发者模式
2. 进入 **设置 → 开发者选项**，开启：
   - **USB 调试**
   - **USB 调试（安全设置）**（部分手机需要，允许通过 ADB 模拟点击）
3. 用 USB 数据线连接手机和电脑
4. 手机上弹出「允许 USB 调试」对话框，勾选「始终允许」并确认

#### 验证连接

```bash
adb devices
```

正常输出：

```
List of devices attached
XXXXXXXX    device
```

> 如果显示 `unauthorized`，请在手机上确认 USB 调试授权弹窗。
> 如果显示 `offline`，尝试拔插 USB 或 `adb kill-server && adb start-server`。

#### 安装 ADB Keyboard（必需）

AutoQA 通过 [ADB Keyboard](https://github.com/nicnocquee/AdbKeyboard) 实现中文输入，需要在手机上安装：

```bash
# 下载 APK
wget https://github.com/nicnocquee/AdbKeyboard/releases/download/v2.0.0/AdbKeyboard.apk

# 安装到手机
adb install AdbKeyboard.apk
```

安装后在手机上启用：**设置 → 语言和输入法 → ADB Keyboard** → 开启。

> 框架会在输入文字时自动切换到 ADB Keyboard，输入完成后自动恢复原始输入法。

## 模型配置

AutoQA 使用三个独立的模型组件，可分别配置不同的模型：

| 组件 | 用途 | 支持的 Provider |
|---|---|---|
| **action_model** | 手机操作（点击、滑动、输入） | `autoglm` |
| **vlm** | 截图断言（判断期望是否成立） | `gemini`、`qwen`、`openai` |
| **planner** | 自然语言 → 测试步骤规划 | `gemini`、`qwen`、`openai` |

### 全局配置（config.yaml）

项目根目录的 `config.yaml` 是全局默认配置，适用于 interactive 和 generate 模式：

```yaml
config:
  action_model:
    provider: "autoglm"
    base_url: "${AUTOGLM_BASE_URL}"
    api_key: "${AUTOGLM_API_KEY}"
    model: "autoglm-phone"

  vlm:
    provider: "gemini"
    api_key: "${GEMINI_API_KEY}"
    model: "gemini-2.5-flash"

  planner:
    provider: "gemini"
    api_key: "${GEMINI_API_KEY}"
    model: "gemini-2.5-flash"
```

### YAML 测试用例中配置

每个 YAML 测试用例可单独配置模型，覆盖全局配置：

```yaml
name: 我的测试
config:
  action_model:
    provider: "autoglm"
    base_url: "${AUTOGLM_BASE_URL}"
    api_key: "${AUTOGLM_API_KEY}"
    model: "autoglm-phone"
  vlm:
    provider: "qwen"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: "${QWEN_API_KEY}"
    model: "qwen-vl-max"
tasks:
  - name: ...
```

### 配置优先级

```
YAML 测试用例 > config.yaml 全局配置 > 代码默认值
```

### 切换模型示例

**VLM 断言从 Gemini 切换到 Qwen：**

```yaml
vlm:
  provider: "qwen"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key: "${QWEN_API_KEY}"
  model: "qwen-vl-max"
```

**Planner 从 Gemini 切换到 OpenAI：**

```yaml
planner:
  provider: "openai"
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o"
```

### 通用配置字段

所有模型组件支持以下字段：

| 字段 | 说明 | 默认值 |
|---|---|---|
| `provider` | 模型提供商 | 组件各异 |
| `base_url` | API 地址（OpenAI 兼容 provider 需要） | `""` |
| `api_key` | API 密钥，支持 `${ENV_VAR}` 语法 | — |
| `model` | 模型名称 | 组件各异 |
| `temperature` | 生成温度 | 0.1 ~ 0.3 |
| `max_tokens` | 最大输出 token 数 | 1000 ~ 3000 |

> `api_key` 和 `base_url` 支持 `${ENV_VAR}` 环境变量语法，避免在文件中硬编码密钥。

## 使用方式

AutoQA 提供三种运行模式：

### 模式 1：执行 YAML 测试用例

编写 YAML 测试用例，直接执行：

```bash
python main.py run examples/toutiao_comment.yaml -v
```

YAML 格式示例：

```yaml
name: 评论区横幅测试

device:
  type: android

config:
  action_model:
    provider: "autoglm"
    base_url: "${AUTOGLM_BASE_URL}"
    api_key: "${AUTOGLM_API_KEY}"
    model: "autoglm-phone"
  vlm:
    provider: "gemini"
    api_key: "${GEMINI_API_KEY}"
    model: "gemini-2.5-flash"

tasks:
  - name: 横幅验证
    flow:
      - action: "打开今日头条 App"
        cache_key: "launch:toutiao"
        timeout: 15
      - action: "点击推荐页面中的第一篇文章"
        cache_key: "tap:first_article"
      - action: "点击评论图标进入评论区"
        cache_key: "tap:comment_entry"
      - assert: "评论区顶部出现了一个红色横幅"
        severity: critical
```

`cache_key` 为可选字段，格式为 `"动作:目标"`（如 `tap:comment_button`、`launch:toutiao`、`swipe:up`）。相同语义的操作在首次执行后会被缓存，后续执行时自动跳过 API 调用。

### 模式 2：自然语言生成 YAML

用自然语言描述测试场景，LLM 自动分解为操作步骤和断言，生成 YAML 文件：

```bash
# 输出到文件
python main.py generate "打开微信，进入朋友圈，检查第一条是否有点赞按钮" -o examples/wechat.yaml

# 输出到终端预览
python main.py generate "打开设置，检查 WiFi 是否已开启"
```

生成后可通过 `run` 模式执行。

### 模式 3：交互式测试

进入交互模式，输入自然语言 → 自动规划 → 确认后执行 → 循环：

```bash
python main.py interactive --device-type adb
```

```
请描述测试场景: 打开今日头条，找到一篇文章，检查评论区是否有横幅

规划中...

  用例: 今日头条评论区横幅检查
  步骤:
    1. [操作] 打开今日头条App
    2. [操作] 点击推荐页面中的第一篇文章
    3. [操作] 点击评论图标进入评论区
    4. [断言] 评论区顶部显示横幅

是否执行? (Y/n): y
```

## CLI 参数

```
python main.py run <yaml_path> [--device-type adb] [--device-id ID] [--no-cache] [-v]
python main.py generate <description> [-o OUTPUT] [--device-type android|harmony|ios] [-v]
python main.py interactive [--device-type adb] [--device-id ID] [-v]
```

| 参数 | 说明 |
|---|---|
| `--device-type` | 设备类型，默认 adb（Android） |
| `--device-id` | 指定设备 ID，不指定则自动检测 |
| `--no-cache` | 禁用 Action 缓存 |
| `-v, --verbose` | 输出详细调试日志 |
| `-o, --output` | generate 模式的输出文件路径 |

## 项目结构

```
├── main.py              # CLI 入口
├── config.yaml          # 全局默认配置
├── config/              # 配置数据类 + 配置加载器
├── providers/           # 统一模型 Provider 层（Gemini / OpenAI 兼容）
├── planner/             # YAML 解析 + LLM 规划器
├── executor/            # 操作执行器 + 模型适配层
├── asserter/            # VLM 视觉断言
├── describer/           # VLM 页面描述器（截图关键信息提取）
├── optimizer/           # LLM 指令优化器（累积历史上下文改写指令）
├── device/              # 设备抽象层（ADB）
├── cache/               # Action 缓存（embedding 语义匹配 + pHash 视觉验证）
├── screenshot/          # 截图管理
├── runner.py            # 测试编排与容错重试
├── suite.py             # 数据模型（TestCase、TestSuite 等）
├── examples/            # 示例 YAML 用例
└── tests/               # 测试脚本
```

## Action 缓存

AutoQA 内置 Action 缓存机制，对重复的操作自动跳过模型 API 调用，降低延迟（每次命中节省 2-3 秒）和成本。

### 工作原理

缓存采用三层查找：

1. **精确过滤**：按 App 包名筛选候选条目（零成本）
2. **语义匹配**：本地 embedding 余弦相似度匹配 cache_key（`sentence-transformers`，<5ms）
3. **视觉验证**：对比坐标区域 pHash，防止 UI 变更后误命中

首次执行某操作时正常调用 API，成功后自动写入缓存（SQLite 持久化）。后续相同操作直接从缓存执行，跳过 API。

### 配置

`config.yaml` 中的缓存配置：

```yaml
cache:
  enabled: true                    # 是否启用缓存
  similarity_threshold: 0.85       # embedding 相似度阈值
  ttl_days: 30                     # 缓存过期天数
```

运行时可通过 `--no-cache` 禁用：

```bash
python main.py run examples/test.yaml --no-cache
```