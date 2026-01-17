# Open Cowork

> A versatile local AI agent platform for Vibe Document, Vibe Coding, and general task execution. Similar to Claude cowork, fully localizable with universal model support.

[**中文**](README_zh.md) | **English**

## 🚀 Project Introduction
**Open Cowork** is a versatile platform for general-purpose tasks, including Vibe Document, Vibe Coding, and Vibe computer execution. Similar to Claude cowork / Cursor, Open Cowork serves as a general-purpose local agent system that can autonomously operate your computer and handle complex tasks through natural language interaction. It offers both GUI and CLI modes, and can be deployed in the cloud, on laptops, or on embedded devices (ARM). The platform includes 20+ built-in tools and many routine files (skills) for a broad range of use cases. Open Cowork excels at creating colorful documents with rich figures, and you can preview and edit your documents directly in the GUI. You can also write programs with multi-round interaction, drag-and-drop file support (or @files), and both agent mode and plan mode. 

### 🤔 Is This Software Right for You?

- **Looking for a Claude cowork-like experience?** Open Cowork provides a similar collaborative AI experience, enabling you to work with an intelligent agent that can understand your needs, operate your local environment, and execute complex tasks autonomously.
- **Need a general-purpose local agent?** If you want an agent system that can handle diverse tasks on your local machine—from code writing to document generation, from data analysis to system operations—Open Cowork is designed for you.
- **Writing complex professional documents?** If you need to create richly illustrated, complex professional reports such as academic papers, in-depth research, or patents, Open Cowork excels at this.
- **Seeking a locally deployable agent?** If you want an agent system that supports local deployment and is compatible with various Anthropic/OpenAI interface models, this could be your solution.
- **Vibe enthusiast?** If you're passionate about the Vibe workflow, you'll love what Open Cowork offers.

### 🆚 Comparison with Claude Cowork

While Open Cowork offers a similar collaborative AI experience to Claude cowork, it provides several key advantages:

- **🏠 Fully Localizable**: Open Cowork can be completely installed and run on your local machine, giving you full control over your data and environment without relying on cloud services.
- **🔌 Universal Model Support**: Unlike Claude cowork which is limited to Claude models, Open Cowork supports any mainstream large language model including Claude, GPT-4, DeepSeek V3, Kimi K2, GLM, Qwen, and more through standard Anthropic/OpenAI API interfaces.
- **💻 Cross-Platform Compatibility**: Full support for Windows, Linux, and macOS, allowing you to use Open Cowork on any operating system you prefer.
- **📖 100% Open Source**: Complete source code available, enabling transparency, customization, and community-driven improvements without vendor lock-in.
- **⚙️ No Dependency on Claude Code**: Built from the ground up with independent architecture, Open Cowork does not require Claude Code as an underlying dependency, providing more flexibility and control.

## Quick start

You can access https://agiagentonline.com/opencoworker to try on Cloud
backup address http://156.238.255.79:5002
You can login in Guest account with empty API key to see demos.
You can also login with your phone number to start a new task.

## GUI for Vibe Everything

![GUI](./md/images/OpenCowork_GUI.png)

**Open Cowork** follows a Plan based ReAct model for complicated task execution. It employs a multi-round iterative working mechanism where the large model can make tool calls and receive feedback results in each round. It is used to update files in the workspace or change the external environment through tools according to user needs. Open Cowork can autonomously call a wide range of MCP tools and operating system tools, featuring multi-agent collaboration, multi-level long-term memory, and embodied intelligence perception. It emphasizes the generality and autonomous decision-making capabilities of the agent. Open Cowork's extensive operating system support, large model support, and multiple operation modes make it suitable for building human-like general intelligence systems to achieve complex report research and generation, project-level code writing, automatic computer operation, multi-agent research (such as competition, debate, collaboration) and other applications.


## 🌐 Platform Compatibility

### Operating System Support
- ✅ **Linux** - Full support
- ✅ **Windows** - Full support  
- ✅ **MacOS** - Full support

### Runtime Interfaces
- **Terminal Mode**: Pure command-line interface, suitable for servers and automation scenarios
- **Python Library Mode**: Embedded as a component in other Python applications
- **Web Interface Mode**: Modern web interface providing visual operation experience

### Interaction Modes
- **Fully Automatic Mode**: Completely autonomous execution without human intervention
- **Interactive Mode**: Supports user confirmation and guidance, providing more control

<br/>

### 📦 Easy Installation

Installation is straightforward. You can use `install.sh` for one-click installation. Basic functionality only requires Python 3.8+ environment. For document conversion and Mermaid image conversion, Playwright and LaTeX are needed. For basic features, you only need to configure the large model API. You don't need to configure an Embedding model, as the code includes built-in vectorized code retrieval functionality.

### Basic Usage

### GUI
```bash
python GUI/app.py
# or specify port：
python GUI/app.py --port 5002

# Then access through browser at http://localhost:5002
```
Web GUI displays file lists. Folders with workspace subdirectories are listed by default, otherwise they won't be shown. The root directory location can be configured in config/config.txt.
Note: Web GUI is currently experimental, providing only a single-user development version (not suitable for industrial deployment).


#### CLI
```bash
#### New task
python cowork.py "Write a joke" 
#### 📁 Specify Output Directory
python cowork.py "Write a joke" --dir "my_dir"
#### 🔄 Continue Task Execution
python cowork.py -c
#### ⚡ Set Execution Rounds
python cowork.py --loops 5 -r "Requirement description"
#### 🔧 Custom Model Configuration
python cowork.py --api-key YOUR_KEY --model gpt-4 --api-base https://api.openai.com/v1
```

> **Note**: 
1. Continue execution only restores the working directory and the last requirement prompt, not the large model's context.

2. Directly specify API configuration through command line, but it's recommended to configure in `config/config.txt` for reuse.

## 🎯 Core Features

- **🧠 Intelligent Task Decomposition**: AI automatically decomposes complex requirements into executable subtasks
- **🔄 Multi-round Iterative Execution**: Each task supports multi-round optimization to ensure quality (default 50 rounds)
- **🔍 Intelligent Code Search**: Semantic search + keyword search for quick code location
- **🌐 Network Search Integration**: Real-time network search for latest information and solutions
- **📚 Codebase Retrieval**: Advanced code repository analysis and intelligent code indexing
- **🛠️ Rich Tool Ecosystem**: Complete local tools + operating system command calling capabilities, supporting full development processes
- **🖼️ Image Input Support**: Use `[img=path]` syntax to include images in requirements, supporting Claude and OpenAI vision models
- **🔗 MCP Integration Support**: Integrate external tools through Model Context Protocol, including third-party services like AI search
- **🖥️ Web Interface**: Intuitive web interface with real-time execution monitoring
- **📊 Dual Format Reports**: JSON detailed logs + Markdown readable reports
- **⚡ Real-time Feedback**: Detailed execution progress and status display
- **🤝 Interactive Control**: Optional user confirmation mode with step-by-step control
- **📁 Flexible Output**: Custom output directory with automatic timestamp naming for new projects

## 🛠️ Tool Sets

Open Cowork provides a rich set of tools divided into **Core Tools** and **Extended Tools** categories. Core tools are available by default, while extended tools need to be manually enabled.

### Core Tools

The core tool set is located in `prompts/tool_prompt.json` and includes the following tools:

| Tool Name | Purpose | File Location |
|-----------|---------|---------------|
| `workspace_search` | Semantic code search to find code snippets most relevant to the query from the codebase | `prompts/tool_prompt.json` |
| `read_file` | Read file contents, supports reading entire file or specified line ranges | `prompts/tool_prompt.json` |
| `grep_search` | Fast regex-based text search with file filtering support | `prompts/tool_prompt.json` |
| `edit_file` | Edit or create files, supports three modes: precise replacement, append, and full replace | `prompts/tool_prompt.json` |
| `run_terminal_cmd` | Execute terminal commands, supports system command calls | `prompts/tool_prompt.json` |
| `web_search` | Web search to get real-time information, results saved to `workspace/web_search_result` | `prompts/tool_prompt.json` |
| `search_img` | Image search using multi-search engine strategy (Google/Baidu/Bing) to search and download images | `prompts/tool_prompt.json` |
| `fetch_webpage_content` | Fetch webpage content for analysis and information extraction | `prompts/tool_prompt.json` |
| `talk_to_user` | User interaction tool to display messages and wait for user input (supports timeout) | `prompts/tool_prompt.json` |
| `read_img` | Analyze and understand images using vision models, returns text description. **Requirement**: Must configure `vision_model`, `vision_api_key`, and `vision_api_base` (optionally `vision_max_tokens`) in `config.txt` before use | `prompts/tool_prompt.json` |
| `merge_file` | Merge multiple files, Markdown files are automatically converted to Word and PDF | `prompts/tool_prompt.json` |
| `convert_docs_to_markdown` | Convert various document formats to Markdown (supports .docx, .xlsx, .html, .tex, .rst, .pptx, .pdf) | `prompts/tool_prompt.json` |
| `compress_history` | Compress conversation history using AI summarization to reduce context length | `prompts/tool_prompt.json` |
| `custom_command` | Execute custom commands, supports game tools and echo tools | `prompts/tool_prompt.json` |

### Extended Tools

The extended tool set is located in `prompts/additional_tools.json` and includes the following tools:

| Tool Name | Purpose | File Location |
|-----------|---------|---------------|
| `tool_help` | Get detailed help information for a specific tool, including parameters and usage examples | `prompts/additional_tools.json` |
| `get_background_update_status` | Get status information and statistics of background incremental update thread for code repository monitoring | `prompts/additional_tools.json` |
| `file_search` | Fast file search based on fuzzy matching against file path, useful when you know part of the path | `prompts/additional_tools.json` |
| `delete_file` | Delete a file at the specified path, fails gracefully if file doesn't exist | `prompts/additional_tools.json` |
| `get_sensor_data` | Acquire physical world information including images, videos, audio, and sensor data | `prompts/additional_tools.json` |
| `run_claude` | Run Claude command using claude_shell.py as a separate process | `prompts/additional_tools.json` |
| `mouse_control` | Control mouse operations including move, click, double click, right click, and scroll | `prompts/additional_tools.json` |
| `read_received_messages` | Fetch messages from mailbox, including broadcast messages and P2P messages | `prompts/additional_tools.json` |
| `send_status_update_to_manager` | Send agent status update to manager for multi-agent collaboration | `prompts/additional_tools.json` |
| `idle` | Agent synchronization tool for waiting and synchronization in multi-agent collaboration | `prompts/additional_tools.json` |
| `recall_memories` | Search and recall relevant memories from long-term memory based on semantic similarity, implements RAG functionality | `prompts/additional_tools.json` |
| `recall_memories_by_time` | Retrieve memories from a specific time period, supports queries like "yesterday", "last week" | `prompts/additional_tools.json` |
| `get_memory_summary` | Get summary of long-term memory system status, including total memories and system health | `prompts/additional_tools.json` |

### Enabling Extended Tools

Extended tools are not loaded by default. To use tools from the extended tool set, you need to copy the corresponding tool definitions from `prompts/additional_tools.json` to `prompts/tool_prompt.json`.

**Steps to Enable**:

1. Open the `prompts/additional_tools.json` file
2. Find the tool definition you need (e.g., `file_search`)
3. Copy the complete JSON definition of that tool (including the tool name and all content)
4. Open the `prompts/tool_prompt.json` file
5. Add the copied tool definition to the JSON object (make sure to maintain correct JSON format and add comma separators)
6. Save the file and restart Open Cowork

**Example**: Enabling the `file_search` tool

```json
// Add to tool_prompt.json:
"file_search": {
  "description": "Fast file search based on fuzzy matching against file path...",
  "parameters": { ... }
}
```

All tools support extension through MCP (Model Context Protocol), allowing easy integration of third-party tools and services.

## 📋 Typical Routines (Task Templates)

Open Cowork provides rich task templates (Routines) in the `routine/` directory. These templates define standard workflows and quality requirements for different types of tasks. You can use these templates directly or customize them according to your needs:

### 📝 Document Writing

- **`academic_journal_paper.txt`** - IEEE journal paper writing standards, including complete academic paper structure and format requirements
- **`academic_conference_paper.txt`** - Academic conference paper writing template
- **`patent_gen.txt`** - Professional patent disclosure generation, conforming to patent literature standards
- **`doc_report.txt`** - Professional report writing with rich charts and detailed discussions
- **`doc_national_proj_apply.txt`** - National project application writing
- **`HTML_report.txt`** - HTML format report generation
- **`blog.txt`** - WeChat public account and online platform article writing

### 💻 Software Development

- **`software_dev.txt`** - Software development standards, including code quality guidelines, project structure specifications, and testing requirements
- **`code_to_doc.txt`** - Code documentation generation

### 🎨 Content Creation

- **`article_with_rich_images.txt`** - Article creation with rich images
- **`add_images_to_doc.txt`** - Add images to documents
- **`ppt.txt`** - Presentation generation

### 🔍 Data Collection

- **`crawl_web_page.txt`** - Web page crawling and data extraction

### 🤝 Multi-Agent Collaboration

- **`multiagent_debate.txt`** - Multi-agent debate and collaboration tasks

### Usage

Reference Routine files in task descriptions, for example:

```bash
python cowork.py "Develop a Python calculator project according to the requirements in routine/software_dev.txt"
```

Or directly specify using a Routine in the requirement:

```bash
python cowork.py "Write an academic paper about deep learning following academic paper standards"
```

Routine files define standard workflows, quality requirements, and output formats for tasks, helping AI better understand task requirements and generate outputs that meet standards.

## 🤖 Model Selection

Open Cowork supports various mainstream AI models including Claude, GPT-4, DeepSeek V3, Kimi K2, etc., meeting different user needs and budgets. With streaming / non-streaming support, tool-call or chat-based tool interface, Anthropic / OpenAI API compatibility.

**🎯 [View Detailed Model Selection Guide →](md/MODELS.md)**

### Quick Recommendations

- **🏆 Quality First**: Claude Sonnet 4.5 - Best intelligence and code quality 
- **💰 Cost-Effective**: DeepSeek V3.2 / GLM-4.7 - Excellent cost-effectiveness ratio
- **🆓 Local deployment**: Qwen3-30B-A3B / GLM-4.5-air - Simple tasks

> 💡 **Tip**: For detailed model comparisons, configuration methods, and performance optimization suggestions, please refer to [MODELS.md](md/MODELS.md)

## ⚙️ Configuration Files

Open Cowork uses `config/config.txt` and `config/config_memory.txt` files for system configuration.

### Quick Configuration
After installation, please configure the following basic options:

```ini
# Required configuration: API key and model
api_key=your_api_key
api_base=the_api_base
model=claude-sonnet-4-0

# Language setting
LANG=en
```
> 💡 **Tip**: For detailed configuration options, usage suggestions, and troubleshooting, please refer to [CONFIG.md](md/CONFIG.md)

## 🔧 Environment Requirements and Installation

### System Requirements
- **Python 3.8+**
- **Network Connection**: For API calls and network search functionality

### Installation Steps
We recommend to use install.sh for automatic install.
If you wish a minimum installation, following:

```bash
# Install from source
pip install -r requirements.txt

# Install web scraping tools (if web scraping is needed)
playwright install-deps
playwright install chromium
```

After installation, don't forget to configure api key, api base, model, and language setting LANG=en or LANG=zh in config/config.txt. 

## ⚠️ Security Notice

As a general-purpose task agent, Open Cowork has the capability to call system terminal commands. Although it usually does not operate files outside the working directory, the large model may execute software installation commands (such as pip, apt, etc.). Please pay attention when using:
- Carefully review executed commands
- Recommend running important tasks in a sandbox environment
- Regularly backup important data

## 🔗 Extended Features

### 🐍 Python Library Interface
Open Cowork now supports being called directly as a Python library in code, providing a programming interface similar to the OpenAI Chat API.

**📖 [View Python Library Usage Guide →](md/README_python_lib.md)**

- 🐍 Pure Python interface, no command line needed
- 💬 OpenAI-style API, easy to integrate
- 🔧 Programmatic configuration, flexible control
- 📊 Detailed return information and status

### 🔌 MCP Protocol Support
Supports Model Context Protocol (MCP) for communication with external tool servers, greatly expanding the system's tool ecosystem.

**📖 [View MCP Integration Guide →](md/README_MCP.md)**

- 🌐 Standardized tool calling protocol
- 🔧 Support for official and third-party MCP servers
- 📁 File system, GitHub, Slack, and other service integrations
- ⚡ Dynamic tool discovery and registration


## Contact Us
You can submit questions or suggestions by opening an Issue. For further communication, you can email us at bitcursor@2925.com

