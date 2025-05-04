# DeepCore - Web3 智能代理平台

<p align="center">
  <img src="http://deepcore.top/deepcore.png" alt="DeepCore Logo" width="200"/>
</p>

<p align="center">
  <b>构建、部署和管理专为 Web3 生态系统设计的高级智能代理</b>
</p>

<p align="center">
  <a href="#功能特性">功能特性</a> •
  <a href="#项目介绍">项目介绍</a> •
  <a href="#技术架构">技术架构</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#应用场景">应用场景</a> •
  <a href="#未来规划">未来规划</a> •
  <a href="README.md">English</a>
</p>

## 功能特性

DeepCore 提供了一系列强大的功能，使开发者能够轻松创建和部署智能代理：

- **多模型支持** - 无缝集成多种 AI 模型，包括 GPT-4、Claude、本地模型等
- **Web3 原生工具** - 丰富的区块链交互工具，支持智能合约调用和链上数据分析
- **MCP 架构** - 创新的 Model-Context-Protocol 架构，将代理逻辑与实现分离
- **钱包认证** - 支持多种 Web3 钱包登录和认证机制
- **跨链支持** - 内置支持多种区块链网络，包括 Ethereum、Polygon、Arbitrum 等
- **可视化监控** - 实时监控和分析代理执行过程和性能
- **工具市场** - 可扩展的工具生态系统，允许开发者共享和复用工具
- **多代理协作** - 支持多个代理之间的通信和协作，以处理复杂任务

## 项目介绍

DeepCore 是一个革命性的平台，一个专为 Web3 应用设计的 AI 代理系统。我们的平台将先进的 AI 技术与区块链集成相结合，为开发者提供无缝且强大的体验，使他们能够创建、部署和管理可以与去中心化应用和区块链网络交互的智能代理。

DeepCore 的架构基于三个核心原则：

1. **去中心化** - 代理在无信任环境中运行，数据和执行由用户控制
2. **可扩展性** - 模块化设计允许轻松集成任何 Web3 协议或传统 API
3. **智能化** - 利用最先进的 AI 模型提供上下文感知和自适应的代理行为

DeepCore 相比其他解决方案的优势在于为 Web3 生态系统中的智能代理构建、部署和商业化提供了完整的解决方案，具有出色的性能和无缝的区块链集成。

## 技术架构

DeepCore 基于创新的 MCP（Model-Context-Protocol）架构构建，这是一种设计模式，使我们能够构建高度灵活和强大的智能代理系统。

<p>
  <img src="./docs/deepcore_architecture.jpg" alt="DeepCore Architecture" width="800"/>
</p>

DeepCore 的架构由三个主要层次组成：

### 1. Web3 代理商店层

顶层提供针对不同用途的专业代理市场：
- **分析代理** - 用于数据分析和洞察生成
- **交易代理** - 用于在各种平台上执行交易策略
- **媒体代理** - 用于内容创建和媒体交互
- **深度研究代理** - 用于深入研究和知识发现
- **其他专业代理** - 可扩展用于各种特定领域任务

### 2. DeepCore 代理协议层

核心协议层是主要代理智能和编排发生的地方：

#### 服务组件
- **MCP 服务** - 实现 Model-Context-Protocol 模式
- **SSE（服务器发送事件）** - 提供实时通信
- **CMD** - 代理控制的命令接口
- **HTTP 服务** - 用于集成的 RESTful API 端点
- **OpenAPIs** - 用于外部连接的标准 API 接口
- **SDKs** - 各种编程语言的软件开发工具包

#### 代理编排
- **规划代理** - 分解复杂任务的中央协调器
- **任务代理** - 执行特定子任务的专业代理
- **工具集成** - 代理可用的各种工具类别：
  - **CodeAct** - 用于代码生成和执行
  - **浏览器** - 用于网页浏览和信息检索
  - **初始工具** - 基本内置工具
  - **搜索** - 跨各种来源的搜索能力
  - **自定义工具** - 用户定义或特定领域的工具

#### 客户端集成
- **工具中心** - 工具发现和管理的中央注册表
- **授权** - 安全性和权限管理
- **客户端 MCP 服务** - 面向各种平台（APP | WEB | 桌面）的客户端接口

### 3. 链基础层

底层提供区块链和数据基础设施：
- **多链支持** - 与主要区块链（BASE、BTC、ETH、BNB、SOL、APT、SUI 等）集成
- **社交媒体集成** - 与 X 和 Telegram 等平台的连接
- **DeFi 集成** - 支持 DEX 和 CEX 交互
- **第三方平台支持** - 与外部平台的可扩展集成

### 核心组件

#### 代理系统

DeepCore 的代理系统由以下主要组件组成：

- **代理核心** - 核心代理逻辑实现，管理推理过程和工具调用
- **记忆系统** - 短期和长期记忆管理，支持上下文感知和历史查询
- **工具管理器** - 工具注册、验证和执行管理
- **提示引擎** - 高级提示模板和提示优化
- **LLM 连接器** - 多模型接口，支持模型混合和回退策略

#### 工具集成

DeepCore 支持多种工具类型：

- **OpenAPI 工具** - 通过 OpenAPI 规范自动集成 RESTful API
- **区块链工具** - 用于与各种区块链网络交互的专业工具
- **数据分析工具** - 用于处理和分析大量数据的工具
- **自定义工具** - 支持开发者创建和注册自定义工具

#### 安全机制

DeepCore 实现了多层安全机制：

- **权限控制** - 细粒度的 API 访问权限管理
- **资源限制** - 监控和限制代理资源使用
- **审计日志** - 全面的操作日志记录
- **漏洞防护** - 防止常见安全漏洞的机制

## 快速开始

### 环境要求

* Python 3.11+
* Poetry (dependency management)
* Docker (optional)
* Git

### 本地开发设置

1. 克隆仓库：

```bash
git clone https://github.com/0xdevpro/deepcore.git
cd deepcore
```

2. 安装依赖：

```bash
poetry install
```

3. 设置环境变量：

```bash
cp .env.example .env
```

4. 配置 `.env` 文件：

```
HOST=localhost
PORT=8000
DEBUG=true
JWT_SECRET=your_jwt_secret
DATABASE_URL=postgresql://user:password@localhost:5432/deepcore
```

5. 启动开发服务器：

```bash
poetry run python api.py
```

## 项目结构

```
deepcore/
├── agents/               # 核心代理实现
│   ├── agent/            # 代理核心逻辑
│   │   ├── mcp/          # MCP 实现
│   │   └── executor/     # 代理执行逻辑
│   ├── api/              # API 端点
│   ├── common/           # 共享工具
│   ├── middleware/       # 中间件组件
│   ├── models/           # 数据模型
│   ├── protocol/         # 协议定义
│   ├── services/         # 业务逻辑
│   └── utils/            # 工具函数
├── sql/                  # 数据库迁移
├── api.py                # 主应用入口
├── pyproject.toml        # 项目依赖
└── README.md             # 项目文档
```

## DeepCore API

DeepCore 提供了一个全面的 RESTful API，使开发者能够无缝地与我们的智能代理平台交互。专为 Web3 生态系统设计，API 支持代理管理、工具集成和安全区块链通信等功能。

### API 概述

DeepCore API 包含以下核心模块：

- **认证模块** - JWT 和 Web3 钱包认证接口
- **代理管理模块** - 创建、更新、删除和查询代理
- **会话管理模块** - 创建会话、发送消息和查询历史
- **工具集成模块** - 工具注册、更新和管理
- **模型管理模块** - 添加和配置 AI 模型
- **文件管理模块** - 上传、下载和管理文件
- **数据分析模块** - 代理性能和用量数据分析

所有 API 都遵循 RESTful 设计原则，支持 JSON 格式数据交换，并提供详细的错误信息。

### 代理端点

DeepCore API 支持完整的代理管理，包括创建、列出、更新和删除代理的端点。

#### 创建代理

**端点：** POST /api/agent/create

```javascript
// 请求
{
    "name": "DeFi 分析师",
    "description": "用于 DeFi 协议分析的代理",
    "mode": "ReAct",
    "tools": ["tool_id_1", "tool_id_2"],
    "model_id": 1
}

// 响应
{
    "agent_id": "agt_12345",
    "name": "DeFi 分析师",
    "status": "created"
}
```

#### 列出代理

**端点：** GET /api/agent/list?skip=0&limit=10

```javascript
// 响应
{
    "total": 25,
    "agents": [
        {
            "agent_id": "agt_12345",
            "name": "DeFi 分析师",
            "description": "用于 DeFi 协议分析的代理",
            "created_at": "2024-03-15T10:30:00Z"
        },
        // ...更多代理
    ]
}
```

#### 更新代理

**端点：** PATCH /api/agent/{agent_id}/update

```javascript
// 请求
{
    "description": "专注于 DeFi 流动性分析的高级代理",
    "tools": ["tool_id_1", "tool_id_2", "tool_id_3"]
}

// 响应
{
    "agent_id": "agt_12345",
    "status": "updated"
}
```

#### 删除代理

**端点：** DELETE /api/agent/{agent_id}

```javascript
// 响应
{
    "agent_id": "agt_12345",
    "status": "deleted"
}
```

### 认证

DeepCore 支持多种强大的认证机制以确保安全访问：

#### JWT 认证

**端点：** POST /api/auth/login

```javascript
// 请求
{
    "username": "example@email.com",
    "password": "your_password"
}

// 响应
{
    "token": "eyJhbGciOiJIUzI1NiIs...",
    "user": {
        "id": "user_id",
        "username": "example@email.com"
    }
}
```

#### Web3 钱包认证

**步骤 1：请求 Nonce**

**端点：** GET /api/auth/wallet/nonce/{wallet_address}

```javascript
// 响应
{
    "nonce": "123456",
    "message": "使用您的钱包签署此消息以在 DeepCore 中进行认证。"
}
```

**步骤 2：使用签名认证**

**端点：** POST /api/auth/wallet/login

```javascript
// 请求
{
    "wallet_type": "metamask",
    "wallet_address": "0xABC...",
    "signature": "signed_message"
}

// 响应
{
    "token": "eyJhbGciOiJIUzI1NiIs...",
    "user": {
        "id": "user_id",
        "wallet_address": "0xABC..."
    }
}
```

### 工具集成

注册和管理代理工具：

**端点：** POST /api/tool/register

```javascript
// 请求
{
    "name": "blockchain_reader",
    "type": "openapi",
    "content": "OpenAPI 规范...",
    "auth_config": {
        "location": "header",
        "key": "Authorization",
        "value": "Bearer ${TOKEN}"
    }
}

// 响应
{
    "tool_id": "tool_789xyz",
    "status": "registered"
}
```

### 会话管理

**创建会话**

**端点：** POST /api/agent/{agent_id}/sessions

```javascript
// 请求
{
    "context": "defi_analysis",
    "metadata": {
        "protocols": ["uniswap", "aave"],
        "chains": ["ethereum"]
    }
}

// 响应
{
    "session_id": "sess_abcde12345",
    "created_at": "2024-03-15T10:30:00Z"
}
```

**发送消息**

**端点：** POST /api/agent/{agent_id}/sessions/{session_id}/messages

```javascript
// 请求
{
    "content": "分析 Uniswap 在 Ethereum 上的最新交易量"
}

// 响应
{
    "message_id": "msg_12345",
    "status": "processing"
}
```

### 模型管理

添加和管理为智能代理提供支持的 AI 模型：

**端点：** POST /api/model/create

```javascript
// 请求
{
    "name": "gpt-4",
    "endpoint": "https://api.openai.com/v1/chat/completions",
    "api_key": "your_api_key",
    "config": {
        "max_tokens": 4096,
        "temperature": 0.7
    }
}

// 响应
{
    "model_id": 1,
    "status": "active"
}
```

### 错误处理和速率限制

DeepCore API 提供一致的错误响应格式：

```javascript
{
    "code": "RESOURCE_NOT_FOUND",
    "message": "请求的代理未找到",
    "details": {
        "agent_id": "agt_12345"
    }
}
```

所有响应都包含速率限制头：

* **X-RateLimit-Limit:** 允许的最大请求数
* **X-RateLimit-Remaining:** 剩余请求数
* **X-RateLimit-Reset:** 速率限制重置时间

## 预构建工具集

DeepCore 提供了一系列可以立即与您的代理集成的预构建工具：

### 区块链工具

- **区块链读取器** - 从各种区块链网络读取数据
- **合约调用器** - 调用智能合约函数
- **交易监控器** - 监控交易活动
- **钱包余额检查器** - 检查钱包余额
- **代币价格追踪器** - 追踪代币价格变动
- **NFT 元数据读取器** - 读取 NFT 元数据

### DeFi 工具

- **DEX 价格聚合器** - 从多个交易所聚合价格
- **流动性池分析器** - 分析流动性池数据
- **收益耕作扫描器** - 扫描最佳收益率
- **Gas 价格估算器** - 估算 gas 价格
- **协议健康监控器** - 监控协议健康指标

### 数据分析工具

- **图表生成器** - 生成图表和可视化
- **市场趋势分析器** - 分析市场趋势
- **历史数据获取器** - 获取历史数据
- **情绪分析器** - 分析社区情绪
- **相关性发现器** - 发现资产之间的相关性

## 部署指南

### Docker 部署

1. 构建镜像：

```bash
docker build -t deepcore .
```

2. 运行容器：

```bash
docker run -p 8000:8000 \
    -e DATABASE_URL=postgresql://user:password@host:5432/deepcore \
    -e JWT_SECRET=your_secret \
    deepcore
```

## 应用场景

DeepCore 的独特架构使其能够支持各种专为 Web3 生态系统定制的应用：

### AI 驱动的 Web3 分析

DeepCore 代理可以分析链上数据，为各种区块链指标提供实时洞察和预测：

- **市场分析**：追踪代币在交易所间的流动并预测价格变动
- **协议健康监控**：分析协议指标和用户活动以评估健康状况和增长
- **鲸鱼活动追踪**：监控大额交易和钱包行为以预测市场影响

### 去中心化金融代理

DeepCore 擅长创建可以与 DeFi 协议交互的代理：

- **投资组合管理**：根据市场条件自动重新平衡投资组合
- **收益优化**：在协议间寻找最高收益机会
- **风险评估**：分析智能合约风险和协议漏洞

### NFT 市场情报

DeepCore 代理可以监控和分析 NFT 市场：

- **收藏品估值**：追踪地板价、销售量和稀有度分布
- **趋势预测**：识别新兴收藏品和创作者
- **机会检测**：基于元数据分析找到被低估的资产

### 跨链自动化

DeepCore 的协议层实现了与多个区块链的无缝交互：

- **跨链套利**：识别和执行跨链价格差异
- **流动性管理**：优化跨多个 DEX 的流动性提供
- **资产桥接**：自动化资产在区块链间的转移过程

## 开发者指南

### 创建自定义代理

以下是创建自定义代理的基本流程：

1. **定义代理配置**

```python
agent_config = {
    "name": "DeFi 资产管理代理",
    "description": "用于自动化 DeFi 资产管理的智能代理",
    "mode": "ReAct",  # 支持 ReAct、Reflection、Structured 模式
    "tools": ["blockchain_reader", "dex_trader", "yield_finder"],
    "model_id": 1,  # 使用 GPT-4 或其他配置的模型
    "memory_config": {
        "short_term_capacity": 10,  # 短期记忆容量
        "long_term_enabled": True   # 启用长期记忆
    }
}
```

2. **调用 API 创建代理**

```python
import requests

response = requests.post(
    "https://your-deepcore-instance.com/api/agent/create",
    json=agent_config,
    headers={"Authorization": f"Bearer {your_token}"}
)

agent_id = response.json()["agent_id"]
print(f"创建的代理 ID: {agent_id}")
```

3. **创建会话并发送指令**

```python
# 创建会话
session_response = requests.post(
    f"https://your-deepcore-instance.com/api/agent/{agent_id}/sessions",
    json={"context": "portfolio_management"},
    headers={"Authorization": f"Bearer {your_token}"}
)

session_id = session_response.json()["session_id"]

# 发送指令
message_response = requests.post(
    f"https://your-deepcore-instance.com/api/agent/{agent_id}/sessions/{session_id}/messages",
    json={"content": "分析我的 ETH 地址 0x123... 在 Uniswap 和 Aave 上的资产，并提供优化建议"},
    headers={"Authorization": f"Bearer {your_token}"}
)

# 获取响应流
import json
import sseclient

url = f"https://your-deepcore-instance.com/api/agent/{agent_id}/sessions/{session_id}/stream"
headers = {"Authorization": f"Bearer {your_token}"}
client = sseclient.SSEClient(url, headers=headers)

for event in client.events():
    data = json.loads(event.data)
    if data["type"] == "thinking":
        print(f"思考过程: {data['content']}")
    elif data["type"] == "action":
        print(f"执行动作: {data['tool']} 参数: {data['parameters']}")
    elif data["type"] == "final":
        print(f"最终答案: {data['content']}")
        break
```

### 创建自定义工具

要为 DeepCore 创建自定义工具，可以使用以下方法：

1. **通过 OpenAPI 规范注册工具**

最简单的方法是通过 OpenAPI（Swagger）规范注册现有 API 作为工具：

```python
openapi_spec = """
openapi: 3.0.0
info:
  title: Token Price API
  version: 1.0.0
paths:
  /price/{token}:
    get:
      summary: Get token price
      parameters:
        - name: token
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Successfully returned price
          content:
            application/json:
              schema:
                type: object
                properties:
                  price:
                    type: number
                  timestamp:
                    type: integer
"""

response = requests.post(
    "https://your-deepcore-instance.com/api/tool/register",
    json={
        "name": "token_price_tool",
        "description": "获取各种代币的最新价格",
        "type": "openapi",
        "content": openapi_spec,
        "base_url": "https://api.tokenprices.example",
        "auth_config": {
            "location": "header",
            "key": "X-API-Key",
            "value": "your_api_key"
        }
    },
    headers={"Authorization": f"Bearer {your_token}"}
)

tool_id = response.json()["tool_id"]
```

2. **以编程方式创建工具**

对于更复杂的工具，可以以编程方式创建自定义工具：

```python
from agents.agent.tools import BaseTool

class TokenPriceTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="token_price_tool",
            description="获取各种代币的最新价格"
        )
        
    async def _run(self, token: str) -> dict:
        # 实现价格获取逻辑
        price = await self._fetch_price(token)
        return {"price": price, "timestamp": int(time.time())}
        
    async def _fetch_price(self, token: str) -> float:
        # 实现具体的价格获取逻辑
        # ...
        return price
```

## 性能和部署建议

### 资源需求

根据您的使用规模，以下是推荐的资源配置：

| 规模 | CPU | 内存 | 存储 | 代理并发数 |
|------|-----|------|------|------------|
| 小型 | 2 核 | 4GB  | 20GB | ~10        |
| 中型 | 4 核 | 8GB  | 50GB | ~50        |
| 大型 | 8 核 | 16GB | 100GB| ~200       |
| 企业级 | 16+ 核 | 32GB+ | 500GB+ | 500+ |

### 扩展建议

对于高负载场景，我们推荐以下扩展策略：

1. **水平扩展** - 使用负载均衡器部署多个 DeepCore 实例以分发请求
2. **代理池** - 预创建代理实例池以减少冷启动时间
3. **模型缓存** - 为常用模型启用响应缓存
4. **分布式工具执行** - 使用专用工具执行集群

### 监控

DeepCore 提供内置监控功能，可以与 Prometheus 集成：

```bash
# Prometheus 配置示例
scrape_configs:
  - job_name: 'deepcore'
    scrape_interval: 15s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['deepcore:8000']
```

关键监控指标包括：

- **agent_creation_time** - 代理创建时间
- **agent_execution_time** - 代理执行时间
- **tool_call_count** - 工具调用次数
- **model_token_usage** - 模型 token 使用量
- **api_request_count** - API 请求次数
- **error_rate** - 错误率

## 未来规划

在未来的开发中，DeepCore 计划：

1. **扩展模型支持** - 集成更多本地和云端 AI 模型，提供更广泛的选择
2. **增强去中心化能力** - 探索去中心化代理执行，使代理能够在去中心化环境中运行
3. **跨链生态系统** - 扩展对更多区块链网络的支持，实现真正的跨链智能代理
4. **社区工具集** - 构建社区驱动的工具库，允许开发者共享和复用代理工具
5. **企业解决方案** - 开发专门针对企业级 Web3 应用的解决方案和部署选项

## 愿景

DeepCore 的愿景是成为 Web3 智能代理的行业标准平台，实现人工智能和区块链技术的无缝集成。我们相信，通过提供强大、灵活和用户友好的工具，我们可以大大加速 Web3 创新，使开发者能够构建更智能、更高效的去中心化应用。

随着 AI 技术和区块链生态系统的不断发展，DeepCore 将继续发展，提供前沿解决方案，使开发者能够构建不仅能理解区块链数据，还能做出智能决策并自动执行复杂操作的智能代理，为最终用户带来更大价值。

## 贡献

我们欢迎社区贡献！如果您有兴趣参与 DeepCore 的开发，请查看我们的贡献指南。

