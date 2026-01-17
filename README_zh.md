# Open Cowork

> 通用的本地 AI 智能体平台，支持 Vibe 文档、Vibe 编程和通用任务执行。类似 Claude cowork，完全可本地化，支持通用模型。

[**English**](README.md)

## 🚀 项目介绍
**Open Cowork** 是一个通用的智能体平台，可以实现氛围文档撰写（Vibe Doc）、Vibe Coding和基于自然语言的通用任务执行。
类似于 Claude cowork，Open Cowork 是一个通用化的本地智能体操作系统，能够自主操作您的计算机，通过自然语言交互处理复杂任务。平台包含 20+ 内置工具和许多例程文件（skills），适用于广泛的使用场景。Open Cowork 擅长创建带有丰富图表的专业文档，并可以直接在 GUI 中预览和编辑文档。也可以用它编写程序，支持多轮交互、拖放文件等（@files）。
提供 GUI 和 CLI 、嵌入式运行等模式，可以部署在云端、笔记本电脑或嵌入式设备（ARM）上。支持Anthropic/OpenAI大模型接口，支持开源/私有化部署。

### 🤔 这款软件适合您吗？

- **正在寻找开源的 Claude cowork？** Open Cowork 提供类似的协作式 AI 体验，让您能够与智能体协作，智能体可以理解您的需求、操作本地环境并自主执行复杂任务。
- **需要通用化的本地智能体？** 如果您想要一个能够在本地机器上处理多样化任务的智能体系统——从代码编写到文档生成，从数据分析到系统操作——Open Cowork 正是为您设计的。
- **编写复杂的专业文档？** 如果您需要创建带有丰富插图、复杂的专业报告，如学术论文、深度研究或专利，Open Cowork 表现的表现会让你满意;
- **寻求可本地部署的代理？** 如果您想要一个支持本地部署且兼容各种 Anthropic/OpenAI 接口模型的代理系统，这可能是您的解决方案;
- **Vibe 爱好者？** 如果您热衷于 Vibe 工作流程，您会喜欢 Open Cowork。

### 🆚 与 Claude Cowork 的对比

虽然 Open Cowork 提供与 Claude cowork 类似的协作式 AI 体验，但它具有以下关键优势：

- **🏠 完全可本地化**：Open Cowork 可以完全安装在您的本地机器上运行，让您完全控制自己的数据和环境，无需依赖云服务。
- **🔌 通用模型支持**：与 Claude cowork 仅限于 Claude 模型不同，Open Cowork 支持任何主流大语言模型，包括 Claude、GPT-4、DeepSeek V3、Kimi K2、GLM、Qwen 等，通过标准的 Anthropic/OpenAI API 接口接入。
- **💻 跨平台兼容性**：完全支持 Windows、Linux 和 macOS，让您可以在任何您喜欢的操作系统上使用 Open Cowork。
- **📖 100% 开源**：提供完整的源代码，实现透明度、可定制性和社区驱动的改进，无供应商锁定。
- **⚙️ 无需 Claude Code 作为底层**：从零开始构建的独立架构，Open Cowork 不需要 Claude Code 作为底层依赖，提供更大的灵活性和控制权。

## 🚀 快速开始
您可以访问 https://agiagentonline.com/opencoworker 试用！ 
备用地址 http://156.238.255.79:5002
您可以使用空的 API key 以游客身份登录并体验演示功能。
您也可以通过手机号登录，开始新的任务。

## GUI for Vibe Everything

![GUI](./md/images/OpenCowork_GUI.png)

**Open Cowork** 遵循基于计划的 ReAct 模型来执行复杂任务。它采用多轮迭代工作机制，大模型可以在每一轮中调用工具并接收反馈结果。它用于根据用户需求更新工作区中的文件或通过工具改变外部环境。Open Cowork 可以自主调用各种 MCP 工具和操作系统工具，具有多代理协作、多级长期记忆和具身智能感知功能。它强调代理的通用性和自主决策能力。Open Cowork 广泛的操作系统支持、大模型支持和多种操作模式使其适合构建类人通用智能系统，以实现复杂的报告研究和生成、项目级代码编写、自动计算机操作、多代理研究（如竞争、辩论、协作）等应用。



## 🌐 平台兼容性

### 操作系统支持
- ✅ **Linux** - 完全支持
- ✅ **Windows** - 完全支持  
- ✅ **MacOS** - 完全支持

### 运行时接口
- **终端模式**：纯命令行界面，适用于服务器和自动化场景
- **Python 库模式**：作为组件嵌入到其他 Python 应用程序中
- **Web 界面模式**：提供可视化操作体验的现代 Web 界面

### 交互模式
- **全自动模式**：完全自主执行，无需人工干预
- **交互模式**：支持用户确认和指导，提供更多控制

<br/>

### 📦 简易安装

安装非常简单。您可以使用 `install.sh` 进行一键安装。基本功能只需要 Python 3.8+ 环境。对于文档转换和 Mermaid 图像转换，需要 Playwright 和 LaTeX。对于基本功能，您只需要配置大模型 API。您不需要配置 Embedding 模型，因为代码包含内置的向量化代码检索功能。

### 基本使用

### GUI
```bash
python GUI/app.py
# or specify port：
python GUI/app.py --port 5002

# 然后通过浏览器访问 http://localhost:5002
```
Web GUI 显示文件列表。默认列出包含工作区子目录的文件夹，否则不会显示。根目录位置可以在 config/config.txt 中配置。
注意：Web GUI 目前是实验性的，仅提供单用户开发版本（不适合工业部署）。


#### CLI
```bash
#### 新任务
python cowork.py "写一个笑话" 
#### 📁 指定输出目录
python cowork.py "写一个笑话" --dir "my_dir"
#### 🔄 继续任务执行
python cowork.py -c
#### ⚡ 设置执行轮数
python cowork.py --loops 5 -r "需求描述"
#### 🔧 自定义模型配置
python cowork.py --api-key YOUR_KEY --model gpt-4 --api-base https://api.openai.com/v1
```

> **注意**： 
1. 继续执行只会恢复工作目录和最后一个需求提示，不会恢复大模型的上下文。

2. 可以通过命令行直接指定 API 配置，但建议在 `config/config.txt` 中配置以便重复使用。

## 🎯 核心功能

- **🧠 智能任务分解**：AI 自动将复杂需求分解为可执行的子任务
- **🔄 多轮迭代执行**：每个任务支持多轮优化以确保质量（默认 50 轮）
- **🔍 智能代码搜索**：语义搜索 + 关键词搜索，快速定位代码
- **🌐 网络搜索集成**：实时网络搜索获取最新信息和解决方案
- **📚 代码库检索**：高级代码仓库分析和智能代码索引
- **🛠️ 丰富的工具生态系统**：完整的本地工具 + 操作系统命令调用能力，支持完整的开发流程
- **🖼️ 图像输入支持**：使用 `[img=path]` 语法在需求中包含图像，支持 Claude 和 OpenAI 视觉模型
- **🔗 MCP 集成支持**：通过模型上下文协议集成外部工具，包括第三方服务如 AI 搜索
- **🖥️ Web 界面**：直观的 Web 界面，实时执行监控
- **📊 双格式报告**：JSON 详细日志 + Markdown 可读报告
- **⚡ 实时反馈**：详细的执行进度和状态显示
- **🤝 交互式控制**：可选的用户确认模式，逐步控制
- **📁 灵活输出**：自定义输出目录，新项目自动时间戳命名

## 🛠️ 工具集

Open Cowork 提供了丰富的工具集，分为**常规工具集**和**扩展工具集**两大类。常规工具集默认可用，扩展工具集需要手动启用。

### 常规工具集

常规工具集位于 `prompts/tool_prompt.json`，包含以下工具：

| 工具名称 | 作用 | 文件位置 |
|---------|------|---------|
| `workspace_search` | 语义代码搜索，从代码库中查找与查询最相关的代码片段 | `prompts/tool_prompt.json` |
| `read_file` | 读取文件内容，支持读取整个文件或指定行范围 | `prompts/tool_prompt.json` |
| `grep_search` | 基于正则表达式的快速文本搜索，支持文件过滤 | `prompts/tool_prompt.json` |
| `edit_file` | 编辑或创建文件，支持精确替换、追加和完全替换三种模式 | `prompts/tool_prompt.json` |
| `run_terminal_cmd` | 执行终端命令，支持系统命令调用 | `prompts/tool_prompt.json` |
| `web_search` | 网络搜索，获取实时信息，结果保存到 `workspace/web_search_result` | `prompts/tool_prompt.json` |
| `search_img` | 图像搜索，使用多搜索引擎策略（Google/Baidu/Bing）搜索并下载图片 | `prompts/tool_prompt.json` |
| `fetch_webpage_content` | 获取网页内容，用于分析和信息提取 | `prompts/tool_prompt.json` |
| `talk_to_user` | 用户交互工具，显示消息并等待用户输入（支持超时设置） | `prompts/tool_prompt.json` |
| `read_img` | 使用视觉模型分析和理解图像，返回文本描述。**需要配置**：需在 `config.txt` 中配置 `vision_model`、`vision_api_key` 和 `vision_api_base`（可选 `vision_max_tokens`）才能使用 | `prompts/tool_prompt.json` |
| `merge_file` | 合并多个文件，Markdown 文件会自动转换为 Word 和 PDF | `prompts/tool_prompt.json` |
| `convert_docs_to_markdown` | 将各种文档格式转换为 Markdown（支持 .docx, .xlsx, .html, .tex, .rst, .pptx, .pdf） | `prompts/tool_prompt.json` |
| `compress_history` | 压缩对话历史，使用 AI 摘要减少上下文长度 | `prompts/tool_prompt.json` |
| `custom_command` | 执行自定义命令，支持游戏工具和回显工具 | `prompts/tool_prompt.json` |

### 扩展工具集

扩展工具集位于 `prompts/additional_tools.json`，包含以下工具：

| 工具名称 | 作用 | 文件位置 |
|---------|------|---------|
| `tool_help` | 获取特定工具的详细帮助信息，包括参数和使用示例 | `prompts/additional_tools.json` |
| `get_background_update_status` | 获取后台增量更新线程的状态信息和统计，用于代码仓库监控 | `prompts/additional_tools.json` |
| `file_search` | 基于模糊匹配的快速文件搜索，适用于只知道部分文件路径的情况 | `prompts/additional_tools.json` |
| `delete_file` | 删除指定路径的文件，如果文件不存在则优雅失败 | `prompts/additional_tools.json` |
| `get_sensor_data` | 获取物理世界信息，包括图像、视频、音频和传感器数据 | `prompts/additional_tools.json` |
| `run_claude` | 使用 claude_shell.py 作为独立进程运行 Claude 命令 | `prompts/additional_tools.json` |
| `mouse_control` | 控制鼠标操作，包括移动、单击、双击、右键点击和滚轮滚动 | `prompts/additional_tools.json` |
| `read_received_messages` | 从邮箱获取消息，包括广播消息和点对点消息 | `prompts/additional_tools.json` |
| `send_status_update_to_manager` | 向管理器发送代理状态更新，用于多代理协作 | `prompts/additional_tools.json` |
| `idle` | 代理同步工具，用于多代理协作中的等待和同步 | `prompts/additional_tools.json` |
| `recall_memories` | 基于语义相似度从长期记忆中检索相关记忆，实现 RAG 功能 | `prompts/additional_tools.json` |
| `recall_memories_by_time` | 从特定时间段检索记忆，支持"昨天"、"上周"等时间查询 | `prompts/additional_tools.json` |
| `get_memory_summary` | 获取长期记忆系统的状态摘要，包括记忆总数和系统健康信息 | `prompts/additional_tools.json` |

### 启用扩展工具

扩展工具集默认不会加载。如果您需要使用扩展工具集中的工具，需要将相应的工具定义从 `prompts/additional_tools.json` 复制到 `prompts/tool_prompt.json` 中。

**启用步骤**：

1. 打开 `prompts/additional_tools.json` 文件
2. 找到您需要的工具定义（例如 `file_search`）
3. 复制该工具的完整 JSON 定义（包括工具名称和所有内容）
4. 打开 `prompts/tool_prompt.json` 文件
5. 将复制的工具定义添加到 JSON 对象中（注意保持 JSON 格式正确，添加逗号分隔符）
6. 保存文件并重启 Open Cowork

**示例**：启用 `file_search` 工具

```json
// 在 tool_prompt.json 中添加：
"file_search": {
  "description": "Fast file search based on fuzzy matching against file path...",
  "parameters": { ... }
}
```

所有工具都支持通过 MCP（模型上下文协议）进行扩展，可以轻松集成第三方工具和服务。

## 📋 典型 Routine（任务模板）

Open Cowork 在 `routine/` 目录下提供了丰富的任务模板（Routine），这些模板定义了不同类型任务的标准工作流程和质量要求。您可以直接使用这些模板，或根据需求自定义：

### 📝 文档撰写类

- **`academic_journal_paper.txt`** - IEEE 期刊论文撰写标准，包含完整的学术论文结构和格式要求
- **`academic_conference_paper.txt`** - 学术会议论文撰写模板
- **`patent_gen.txt`** - 专业专利说明书生成，符合专利文献标准
- **`doc_report.txt`** - 专业报告撰写，支持丰富的图表和详细讨论
- **`doc_national_proj_apply.txt`** - 国家级项目申请书撰写
- **`HTML_report.txt`** - HTML 格式报告生成
- **`blog.txt`** - 微信公众号和在线平台文章撰写

### 💻 软件开发类

- **`software_dev.txt`** - 软件开发规范，包含代码质量指南、项目结构规范和测试调试要求
- **`code_to_doc.txt`** - 代码文档生成

### 🎨 内容创作类

- **`article_with_rich_images.txt`** - 带丰富图像的文章创作
- **`add_images_to_doc.txt`** - 为文档添加图像
- **`ppt.txt`** - 演示文稿生成

### 🔍 数据采集类

- **`crawl_web_page.txt`** - 网页爬取和数据提取

### 🤝 多代理协作类

- **`multiagent_debate.txt`** - 多代理辩论和协作任务

### 使用方式

在任务描述中引用 Routine 文件，例如：

```bash
python cowork.py "根据 routine/software_dev.txt 的要求开发一个 Python 计算器项目"
```

或者直接在需求中说明使用某个 Routine：

```bash
python cowork.py "按照学术论文标准撰写一篇关于深度学习的论文"
```

Routine 文件定义了任务的标准流程、质量要求和输出格式，帮助 AI 更好地理解任务需求并生成符合标准的输出。

## 🤖 模型选择

Open Cowork 支持各种主流 AI 模型，包括 Claude、GPT-4、DeepSeek V3、Kimi K2 等，满足不同用户需求和预算。支持流式/非流式、工具调用或基于聊天的工具接口、Anthropic/OpenAI API 兼容性。


**🎯 [查看详细模型选择指南 →](md/MODELS.md)**

### 快速推荐

- **🏆 质量优先**：Claude Sonnet 4.5 - 最佳智能和代码质量 
- **💰 性价比**：DeepSeek V3.2 / GLM-4.7 - 出色的性价比
- **🆓 本地部署**：Qwen3-30B-A3B / GLM-4.5-air - 简单任务

> 💡 **提示**：有关详细的模型比较、配置方法和性能优化建议，请参阅 [MODELS.md](md/MODELS.md)

## ⚙️ 配置文件

Open Cowork 使用 `config/config.txt` 和 `config/config_memory.txt` 文件进行系统配置。

### 快速配置
安装后，请配置以下基本选项：

```ini
# 必需配置：API 密钥和模型
api_key=your_api_key
api_base=the_api_base
model=claude-sonnet-4-0

# 语言设置
LANG=zh
```
> 💡 **提示**：有关详细配置选项、使用建议和故障排除，请参阅 [CONFIG.md](md/CONFIG.md)

## 🔧 环境要求和安装

### 系统要求
- **Python 3.8+**
- **网络连接**：用于 API 调用和网络搜索功能

### 安装步骤
我们建议使用 install.sh 进行自动安装。
如果您希望最小化安装，请遵循以下步骤：

```bash
# 从源码安装
pip install -r requirements.txt

# 安装网页抓取工具（如果需要网页抓取）
playwright install-deps
playwright install chromium
```

安装后，不要忘记在 config/config.txt 中配置 api key、api base、model 和语言设置 LANG=en 或 LANG=zh。

## ⚠️ 安全提示

作为通用任务代理，Open Cowork 具有调用系统终端命令的能力。虽然它通常不会在工作目录外操作文件，但大模型可能会执行软件安装命令（如 pip、apt 等）。使用时请注意：
- 仔细审查执行的命令
- 建议在沙箱环境中运行重要任务
- 定期备份重要数据

## 🔗 扩展功能

### 🐍 Python 库接口
Open Cowork 现在支持在代码中直接作为 Python 库调用，提供类似于 OpenAI Chat API 的编程接口。

**📖 [查看 Python 库使用指南 →](md/README_python_lib_zh.md)**

- 🐍 纯 Python 接口，无需命令行
- 💬 OpenAI 风格 API，易于集成
- 🔧 程序化配置，灵活控制
- 📊 详细的返回信息和状态

### 🔌 MCP 协议支持
支持模型上下文协议（MCP）与外部工具服务器通信，大大扩展了系统的工具生态系统。

**📖 [查看 MCP 集成指南 →](md/README_MCP_zh.md)**

- 🌐 标准化工具调用协议
- 🔧 支持官方和第三方 MCP 服务器
- 📁 文件系统、GitHub、Slack 等服务集成
- ⚡ 动态工具发现和注册

## 联系我们
您可以通过提交 Issue 来提交问题或建议。如需进一步沟通，您可以发送邮件至 bitcursor@2925.com
