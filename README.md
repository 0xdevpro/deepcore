# DeepCore - Web3 Intelligent Agent Platform

<p align="center">
  <img src="http://deepcore.top/deepcore.png" alt="DeepCore Logo" width="200"/>
</p>

<p align="center">
  <b>Build, Deploy, and Manage Advanced Intelligent Agents Designed for the Web3 Ecosystem</b>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#introduction">Introduction</a> •
  <a href="#technical-architecture">Technical Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#use-cases">Use Cases</a> •
  <a href="#future-plans">Future Plans</a>
</p>

## Features

DeepCore provides a series of powerful features that enable developers to easily create and deploy intelligent agents:

- **Multi-model Support** - Seamless integration with various AI models, including GPT-4, Claude, local models, etc.
- **Web3 Native Tools** - Rich blockchain interaction tools supporting smart contract calls and on-chain data analysis
- **MCP Architecture** - Innovative Model-Context-Protocol architecture, separating agent logic from implementation
- **Wallet Authentication** - Support for various Web3 wallet login and authentication mechanisms
- **Cross-chain Support** - Built-in support for multiple blockchain networks, including Ethereum, Polygon, Arbitrum, etc.
- **Visual Monitoring** - Real-time monitoring and analysis of agent execution processes and performance
- **Tool Marketplace** - Extensible tool ecosystem allowing developers to share and reuse tools
- **Multi-agent Collaboration** - Support for communication and collaboration between multiple agents to handle complex tasks

## Introduction

DeepCore is a revolutionary platform, an AI agent system designed specifically for Web3 applications. Our platform combines advanced AI technology with blockchain integration to provide developers with a seamless and powerful experience, enabling them to create, deploy, and manage intelligent agents that can interact with decentralized applications and blockchain networks.

DeepCore's architecture is built on three core principles:

1. **Decentralization** - Agents operate in a trustless environment, with data and execution controlled by the user
2. **Extensibility** - Modular design allows for easy integration with any Web3 protocol or traditional API
3. **Intelligence** - Leveraging state-of-the-art AI models to provide context-aware and adaptive agent behaviors

DeepCore's advantage over other solutions lies in providing a complete solution for building, deploying, and commercializing intelligent agents in the Web3 ecosystem, with outstanding performance and seamless blockchain integration.

## Technical Architecture

DeepCore is built on the innovative MCP (Model-Context-Protocol) architecture, a design pattern that enables us to build highly flexible and powerful intelligent agent systems.

<p>
  <img src="./docs/deepcore_architecture.jpg" alt="DeepCore Architecture" width="800"/>
</p>

DeepCore's architecture consists of three main layers:

### 1. Web3 Agent Store Layer

The top layer provides a marketplace of specialized agents for different purposes:
- **Analysis Agent** - For data analytics and insights generation
- **Trade Agent** - For executing trading strategies on various platforms
- **Media Agent** - For content creation and media interaction
- **DeepResearch Agent** - For in-depth research and knowledge discovery
- **Additional specialized agents** - Extensible for various domain-specific tasks

### 2. DeepCore Agent Protocol Layer

The core protocol layer is where the main agent intelligence and orchestration happens:

#### Service Components
- **MCP Service** - Implements the Model-Context-Protocol pattern
- **SSE (Server-Sent Events)** - Provides real-time communication
- **CMD** - Command interface for agent control
- **HTTP Service** - RESTful API endpoints for integrations
- **OpenAPIs** - Standardized API interfaces for external connectivity
- **SDKs** - Software Development Kits for various programming languages

#### Agent Orchestration
- **Planner Agent** - Central coordinator that breaks down complex tasks
- **TaskAgents** - Specialized agents that execute specific subtasks
- **Tools Integration** - Various tool categories available to agents:
  - **CodeAct** - For code generation and execution
  - **Browser** - For web browsing and information retrieval
  - **Initial Tools** - Basic built-in tooling
  - **Search** - Search capabilities across various sources
  - **Custom Tools** - User-defined or domain-specific tools

#### Client Integration
- **Tools Center** - Central registry for tool discovery and management
- **Authorization** - Security and permissions management
- **MCP Service for Client** - Client-facing interfaces for various platforms (APP | WEB | Desktop)

### 3. Chain Foundation Layer

The bottom layer provides blockchain and data infrastructure:
- **Multi-chain Support** - Integration with major blockchains (BASE, BTC, ETH, BNB, SOL, APT, SUI, etc.)
- **Social Media Integration** - Connections to platforms like X and Telegram
- **DeFi Integration** - Support for DEX and CEX interactions
- **Third-party Platform Support** - Extensible integration with external platforms

### Core Components

#### Agent System

DeepCore's agent system consists of the following main components:

- **Agent Core** - Core agent logic implementation, managing reasoning processes and tool invocation
- **Memory System** - Short-term and long-term memory management, supporting context awareness and historical queries
- **Tool Manager** - Tool registration, validation, and execution management
- **Prompt Engine** - Advanced prompt templates and prompt optimization
- **LLM Connector** - Multi-model interface, supporting model mixing and fallback strategies

#### Tool Integration

DeepCore supports multiple tool types:

- **OpenAPI Tools** - Automatically integrate RESTful APIs through OpenAPI specifications
- **Blockchain Tools** - Specialized tools for interacting with various blockchain networks
- **Data Analysis Tools** - Tools for processing and analyzing large amounts of data
- **Custom Tools** - Support for developers to create and register custom tools

#### Security Mechanisms

DeepCore implements multi-layered security mechanisms:

- **Permission Control** - Fine-grained API access permission management
- **Resource Limits** - Monitoring and limiting agent resource usage
- **Audit Logs** - Comprehensive operation logging
- **Vulnerability Protection** - Mechanisms to prevent common security vulnerabilities

## Quick Start

### Requirements

* Python 3.11+
* Poetry (dependency management)
* Docker (optional)
* Git

### Local Development Setup

1. Clone the repository:

```bash
git clone https://github.com/0xdevpro/deepcore.git
cd deepcore
```

2. Install dependencies:

```bash
poetry install
```

3. Set up environment variables:

```bash
cp .env.example .env
```

4. Configure your `.env` file:

```
HOST=localhost
PORT=8000
DEBUG=true
JWT_SECRET=your_jwt_secret
DATABASE_URL=postgresql://user:password@localhost:5432/deepcore
```

5. Start the development server:

```bash
poetry run python api.py
```

## Project Structure

```
deepcore/
├── agents/               # Core agent implementation
│   ├── agent/            # Agent core logic
│   │   ├── mcp/          # MCP implementation
│   │   └── executor/     # Agent execution logic
│   ├── api/              # API endpoints
│   ├── common/           # Shared utilities
│   ├── middleware/       # Middleware components
│   ├── models/           # Data models
│   ├── protocol/         # Protocol definitions
│   ├── services/         # Business logic
│   └── utils/            # Utility functions
├── sql/                  # Database migrations
├── api.py                # Main application entry
├── pyproject.toml        # Project dependencies
└── README.md             # Project documentation
```

## DeepCore API

DeepCore provides a comprehensive RESTful API that enables developers to interact seamlessly with our intelligent agent platform. Designed specifically for the Web3 ecosystem, the API supports functionalities such as agent management, tool integration, and secure blockchain-based communications.

### API Overview

The DeepCore API includes the following core modules:

- **Authentication Module** - JWT and Web3 wallet authentication interfaces
- **Agent Management Module** - Create, update, delete, and query agents
- **Session Management Module** - Create sessions, send messages, and query history
- **Tool Integration Module** - Tool registration, updating, and management
- **Model Management Module** - Add and configure AI models
- **File Management Module** - Upload, download, and manage files
- **Data Analysis Module** - Agent performance and usage data analysis

All APIs follow RESTful design principles, support JSON format data exchange, and provide detailed error information.

### Agent Endpoints

DeepCore API supports complete agent management, including endpoints for creating, listing, updating, and deleting agents.

#### Create Agent

**Endpoint:** POST /api/agent/create

```json
// Request
{
    "name": "DeFi Analyst",
    "description": "Agent for DeFi protocol analysis",
    "mode": "ReAct",
    "tools": ["tool_id_1", "tool_id_2"],
    "model_id": 1
}

// Response
{
    "agent_id": "agt_12345",
    "name": "DeFi Analyst",
    "status": "created"
}
```

#### List Agents

**Endpoint:** GET /api/agent/list?skip=0&limit=10

```json
// Response
{
    "total": 25,
    "agents": [
        {
            "agent_id": "agt_12345",
            "name": "DeFi Analyst",
            "description": "Agent for DeFi protocol analysis",
            "created_at": "2024-03-15T10:30:00Z"
        },
        // ...more agents
    ]
}
```

#### Update Agent

**Endpoint:** PATCH /api/agent/{agent_id}/update

```json
// Request
{
    "description": "Advanced agent specialized in DeFi liquidity analysis",
    "tools": ["tool_id_1", "tool_id_2", "tool_id_3"]
}

// Response
{
    "agent_id": "agt_12345",
    "status": "updated"
}
```

#### Delete Agent

**Endpoint:** DELETE /api/agent/{agent_id}

```json
// Response
{
    "agent_id": "agt_12345",
    "status": "deleted"
}
```

### Authentication

DeepCore supports multiple robust authentication mechanisms to ensure secure access:

#### JWT Authentication

**Endpoint:** POST /api/auth/login

```json
// Request
{
    "username": "example@email.com",
    "password": "your_password"
}

// Response
{
    "token": "eyJhbGciOiJIUzI1NiIs...",
    "user": {
        "id": "user_id",
        "username": "example@email.com"
    }
}
```

#### Web3 Wallet Authentication

**Step 1: Request Nonce**

**Endpoint:** GET /api/auth/wallet/nonce/{wallet_address}

```json
// Response
{
    "nonce": "123456",
    "message": "Sign this message with your wallet to authenticate with DeepCore."
}
```

**Step 2: Authenticate with Signature**

**Endpoint:** POST /api/auth/wallet/login

```json
// Request
{
    "wallet_type": "metamask",
    "wallet_address": "0xABC...",
    "signature": "signed_message"
}

// Response
{
    "token": "eyJhbGciOiJIUzI1NiIs...",
    "user": {
        "id": "user_id",
        "wallet_address": "0xABC..."
    }
}
```

### Tool Integration

Register and manage tools for agents:

**Endpoint:** POST /api/tool/register

```json
// Request
{
    "name": "blockchain_reader",
    "type": "openapi",
    "content": "OpenAPI specification...",
    "auth_config": {
        "location": "header",
        "key": "Authorization",
        "value": "Bearer ${TOKEN}"
    }
}

// Response
{
    "tool_id": "tool_789xyz",
    "status": "registered"
}
```

### Session Management

**Create Session**

**Endpoint:** POST /api/agent/{agent_id}/sessions

```json
// Request
{
    "context": "defi_analysis",
    "metadata": {
        "protocols": ["uniswap", "aave"],
        "chains": ["ethereum"]
    }
}

// Response
{
    "session_id": "sess_abcde12345",
    "created_at": "2024-03-15T10:30:00Z"
}
```

**Send Message**

**Endpoint:** POST /api/agent/{agent_id}/sessions/{session_id}/messages

```json
// Request
{
    "content": "Analyze the latest trading volume on Uniswap on Ethereum"
}

// Response
{
    "message_id": "msg_12345",
    "status": "processing"
}
```

### Model Management

Add and manage AI models that power intelligent agents:

**Endpoint:** POST /api/model/create

```json
// Request
{
    "name": "gpt-4",
    "endpoint": "https://api.openai.com/v1/chat/completions",
    "api_key": "your_api_key",
    "config": {
        "max_tokens": 4096,
        "temperature": 0.7
    }
}

// Response
{
    "model_id": 1,
    "status": "active"
}
```

### Error Handling and Rate Limiting

DeepCore API provides a consistent error response format:

```json
{
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested agent was not found",
    "details": {
        "agent_id": "agt_12345"
    }
}
```

All responses include rate limit headers:

* **X-RateLimit-Limit:** Maximum number of requests allowed
* **X-RateLimit-Remaining:** Number of requests remaining
* **X-RateLimit-Reset:** Time until rate limit resets

## Pre-built Tool Set

DeepCore provides a series of pre-built tools that can be immediately integrated with your agents:

### Blockchain Tools

- **Blockchain Reader** - Read data from various blockchain networks
- **Contract Caller** - Call smart contract functions
- **Transaction Monitor** - Monitor transaction activity
- **Wallet Balance Checker** - Check wallet balances
- **Token Price Tracker** - Track token price movements
- **NFT Metadata Reader** - Read NFT metadata

### DeFi Tools

- **DEX Price Aggregator** - Aggregate prices from multiple exchanges
- **Liquidity Pool Analyzer** - Analyze liquidity pool data
- **Yield Farming Scanner** - Scan for best yield rates
- **Gas Price Estimator** - Estimate gas prices
- **Protocol Health Monitor** - Monitor protocol health metrics

### Data Analysis Tools

- **Chart Generator** - Generate charts and visualizations
- **Market Trend Analyzer** - Analyze market trends
- **Historical Data Fetcher** - Retrieve historical data
- **Sentiment Analyzer** - Analyze community sentiment
- **Correlation Finder** - Discover correlations between assets

## Deployment Guide

### Docker Deployment

1. Build the image:

```bash
docker build -t deepcore .
```

2. Run the container:

```bash
docker run -p 8000:8000 \
    -e DATABASE_URL=postgresql://user:password@host:5432/deepcore \
    -e JWT_SECRET=your_secret \
    deepcore
```

## Use Cases

DeepCore's unique architecture enables it to support a variety of applications specifically tailored for the Web3 ecosystem:

### AI-Powered Web3 Analytics

DeepCore agents can analyze on-chain data to provide real-time insights and predictions for various blockchain metrics:

- **Market Analysis**: Track token movements across exchanges and predict price movements
- **Protocol Health Monitoring**: Analyze protocol metrics and user activity to assess health and growth
- **Whale Activity Tracking**: Monitor large transactions and wallet behaviors to predict market impacts

### Decentralized Finance Agents

DeepCore excels at creating agents that can interact with DeFi protocols:

- **Portfolio Management**: Automatically rebalance portfolios based on market conditions
- **Yield Optimization**: Find the highest yield opportunities across protocols
- **Risk Assessment**: Analyze smart contract risks and protocol vulnerabilities

### NFT Market Intelligence

DeepCore agents can monitor and analyze the NFT market:

- **Collection Valuation**: Track floor prices, sales volume, and rarity distributions
- **Trend Prediction**: Identify emerging collections and creators
- **Opportunity Detection**: Find undervalued assets based on metadata analysis

### Cross-Chain Automation

DeepCore's protocol layer enables seamless interaction with multiple blockchains:

- **Cross-Chain Arbitrage**: Identify and execute price differences across chains
- **Liquidity Management**: Optimize liquidity provision across multiple DEXs
- **Asset Bridging**: Automate the process of moving assets between blockchains

## Developer Guide

### Creating Custom Agents

Here is the basic process for creating custom agents:

1. **Define Agent Configuration**

```python
agent_config = {
    "name": "DeFi Asset Management Agent",
    "description": "Intelligent agent for automated DeFi asset management",
    "mode": "ReAct",  # Supports ReAct, Reflection, Structured modes
    "tools": ["blockchain_reader", "dex_trader", "yield_finder"],
    "model_id": 1,  # Use GPT-4 or other configured model
    "memory_config": {
        "short_term_capacity": 10,  # Short-term memory capacity
        "long_term_enabled": True   # Enable long-term memory
    }
}
```

2. **Call API to Create Agent**

```python
import requests

response = requests.post(
    "https://your-deepcore-instance.com/api/agent/create",
    json=agent_config,
    headers={"Authorization": f"Bearer {your_token}"}
)

agent_id = response.json()["agent_id"]
print(f"Created agent ID: {agent_id}")
```

3. **Create Session and Send Instructions**

```python
# Create session
session_response = requests.post(
    f"https://your-deepcore-instance.com/api/agent/{agent_id}/sessions",
    json={"context": "portfolio_management"},
    headers={"Authorization": f"Bearer {your_token}"}
)

session_id = session_response.json()["session_id"]

# Send instruction
message_response = requests.post(
    f"https://your-deepcore-instance.com/api/agent/{agent_id}/sessions/{session_id}/messages",
    json={"content": "Analyze my ETH address 0x123... assets on Uniswap and Aave, and provide optimization suggestions"},
    headers={"Authorization": f"Bearer {your_token}"}
)

# Get response stream
import json
import sseclient

url = f"https://your-deepcore-instance.com/api/agent/{agent_id}/sessions/{session_id}/stream"
headers = {"Authorization": f"Bearer {your_token}"}
client = sseclient.SSEClient(url, headers=headers)

for event in client.events():
    data = json.loads(event.data)
    if data["type"] == "thinking":
        print(f"Thinking process: {data['content']}")
    elif data["type"] == "action":
        print(f"Executing action: {data['tool']} Parameters: {data['parameters']}")
    elif data["type"] == "final":
        print(f"Final answer: {data['content']}")
        break
```

### Creating Custom Tools

To create custom tools for DeepCore, you can use the following methods:

1. **Register Tools via OpenAPI Specification**

The simplest way is to register existing APIs as tools through OpenAPI (Swagger) specifications:

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
        "description": "Get the latest prices for various tokens",
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

2. **Create Tools Programmatically**

For more complex tools, you can create custom tools programmatically:

```python
from agents.agent.tools import BaseTool

class TokenPriceTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="token_price_tool",
            description="Get the latest prices for various tokens"
        )
        
    async def _run(self, token: str) -> dict:
        # Implement price retrieval logic
        price = await self._fetch_price(token)
        return {"price": price, "timestamp": int(time.time())}
        
    async def _fetch_price(self, token: str) -> float:
        # Implement specific price retrieval logic
        # ...
        return price
```

## Performance and Deployment Recommendations

### Resource Requirements

Based on your usage scale, the following are recommended resource configurations:

| Scale | CPU | Memory | Storage | Agent Concurrency |
|------|-----|------|------|------------|
| Small | 2 cores | 4GB  | 20GB | ~10        |
| Medium | 4 cores | 8GB  | 50GB | ~50        |
| Large | 8 cores | 16GB | 100GB| ~200       |
| Enterprise | 16+ cores | 32GB+ | 500GB+ | 500+ |

### Scaling Recommendations

For high-load scenarios, we recommend the following scaling strategies:

1. **Horizontal Scaling** - Deploy multiple DeepCore instances using a load balancer to distribute requests
2. **Agent Pool** - Pre-create agent instance pools to reduce cold start time
3. **Model Caching** - Enable response caching for commonly used models
4. **Distributed Tool Execution** - Use a dedicated tool execution cluster

### Monitoring

DeepCore provides built-in monitoring capabilities that can be integrated with Prometheus:

```bash
# Prometheus configuration example
scrape_configs:
  - job_name: 'deepcore'
    scrape_interval: 15s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['deepcore:8000']
```

Key monitoring metrics include:

- **agent_creation_time** - Agent creation time
- **agent_execution_time** - Agent execution time
- **tool_call_count** - Tool call count
- **model_token_usage** - Model token usage
- **api_request_count** - API request count
- **error_rate** - Error rate

## Future Plans

In future development, DeepCore plans to:

1. **Expand Model Support** - Integrate more local and cloud-based AI models, providing a wider range of choices
2. **Enhance Decentralization Capabilities** - Explore decentralized agent execution, enabling agents to run in decentralized environments
3. **Cross-Chain Ecosystem** - Extend support for more blockchain networks, achieving truly cross-chain intelligent agents
4. **Community Tool Set** - Build a community-driven tool library, allowing developers to share and reuse agent tools
5. **Enterprise Solutions** - Develop solutions and deployment options specifically for enterprise-level Web3 applications

## Vision

DeepCore's vision is to become the industry standard platform for Web3 intelligent agents, achieving seamless integration of artificial intelligence and blockchain technology. We believe that by providing powerful, flexible, and user-friendly tools, we can greatly accelerate Web3 innovation, enabling developers to build smarter and more efficient decentralized applications.

As AI technology and the blockchain ecosystem continue to evolve, DeepCore will continue to evolve, providing cutting-edge solutions that enable developers to build intelligent agents that not only understand blockchain data but can also make intelligent decisions and automatically execute complex operations, bringing greater value to end users.

## Contributions

We welcome community contributions! If you're interested in participating in DeepCore's development, please check our contribution guidelines.

