# Stashly Financial Assistant

A conversational AI assistant that provides financial analysis, portfolio management guidance, and market insights.

## Recent Updates

- 🔄 Complete refactoring of research agent structure for better modularity
- 📊 Improved chart generation using real market data from market_data.xlsx
- 🔍 Enhanced portfolio analysis with support for common portfolio aliases (70/30, 60/40, etc.)
- 🚀 Better error handling and response formatting
- ⏱️ Optimized agent performance with proper timeout handling
- 👤 Added personalized agent personas in Knowledge_base

## Installation

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized deployment)

### Setup
1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

### Docker Deployment
Make sure Docker Desktop is running, then build and start the container:

## Features

- **AI Financial Analyst**: Powered by OpenAI's GPT-4 model
- **Interactive Avatar**: Integration with Simli API for a personalized avatar experience
- **Market Analysis**: Provides detailed market insights and financial data analysis
- **Streaming Responses**: Real-time streaming chat responses for better user experience
- **Research Agent System**: Modular research system with specialized agents:
  - Analysis Agent: Market and financial data analysis
  - Editor Agent: Content refinement and formatting
  - Expert Agent: Domain-specific expertise
  - Research Supervisor: Coordination and task management
- **Fund Analysis System**:
  - Fund Loader: Parses and processes fund XML data from Swedish financial institutions
  - Fund Position Agent: Analyzes fund holdings, exposures, and sector allocations
  - Support for common fund aliases (e.g., "LF Global", "LF USA")
  - Exposure calculation across multiple funds
  - Sector and industry analysis
- **Stock Price Analysis**:
  - Real-time stock data using Yahoo Finance
  - Support for common company aliases (e.g., "Apple", "Tesla", "Google")
  - Current price and market data
  - Fundamental analysis (PE ratio, EPS, revenue, etc.)
  - Dividend history
  - Historical price data and charts
- **Web Research System**:
  - Web search for recent market news and events
  - Wikipedia integration for background information
  - Smart date range handling for time-sensitive queries
  - Fallback mechanisms for comprehensive coverage
  - Context-aware search results
- **Conversational Agent**:
  - Personalized responses using Ashley's persona
  - Context-aware conversations with memory
  - Handles greetings, introductions, and personal questions
  - Maintains conversation history and user context
- **Multiple Endpoints**:
  - Streaming chat completions (OpenAI compatible)
  - Regular chat interface
  - Avatar connection and management
  - Market data analysis
  - Conversation history management
  - Research agent interactions
  - Fund analysis and exposure calculations
  - Stock price queries
  - Web research results

## Prerequisites

- Python 3.11
- Docker
- Azure CLI (for deployment)
- OpenAI API key
- Simli API credentials (managed in frontend)
- Market data file (market_data.xlsx)

## Local Development

1. **Set Up Environment**
   ```bash
   # Clone repository and install dependencies
   pip install -r requirements.txt

   # Create .env file with required API keys
   OPENAI_API_KEY=your_openai_key
   ```

2. **Run Locally**
   ```bash
   # Test the React agent directly
   python react_agent.py

   # Or run the full FastAPI application
   uvicorn app:app --reload
   ```

## Docker Setup

1. **Build the Image**
   ```bash
   docker build -t stashly-financial-coach .
   ```

2. **Run the Container**
   ```bash
   docker run -p 8000:8000 -v ${PWD}/.env:/app/.env stashly-financial-coach
   ```

## API Endpoints

- `/chat/completions`: Streaming chat interface (OpenAI compatible)
- `/chat`: Regular chat interface
- `/conversations/{thread_id}`: Get conversation history
- `/conversations/{thread_id}`: Clear conversation history (DELETE)
- `/test`: Simple test endpoint
- `/simli/connect`: Connect to Simli avatar
- `/proxy/simli/startE2ESession`: Proxy endpoint for Simli API
- `/test/simli-api`: Test Simli API connection

## Azure Deployment

For detailed deployment instructions, see [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md)

## Environment Variables

- `OPENAI_API_KEY`: Required for AI functionality
- `TAVILY_API_KEY`: For additional research capabilities
- `GOOGLE_API_KEY`: For extended functionality
- `SIMLI_API_KEY`: For avatar integration
- `SIMLI_API_SECRET`: For avatar integration
- `OPENFIGI_API_KEY`: For fund and security information lookup
- `MARKET_DATA_PATH`: Path to market data files (default: /app/Market_data)

Note: Simli API credentials are managed by the frontend application.

## Testing

- Use the `/test` endpoint to verify API functionality
- Test avatar connection using `/test/simli-api`
- Monitor logs for detailed debugging information

## Error Handling

The application includes comprehensive error handling for:
- API connection issues
- Rate limiting
- Invalid requests
- Timeout scenarios

## API Usage

### Stock Price Analysis
`POST /chat`

Request:
```json
{
  "message": "What is Apple's current stock price and PE ratio?",
  "thread_id": "default"
}
```

Response:
```json
{
  "response": "The current price of Apple Inc. (AAPL) is $175.50 USD.\n\nKey fundamentals:\nMarket Cap: 2.8T\nPE Ratio: 28.5\nEPS: 6.15\nRevenue: 383.3B\nProfit Margin: 25.3%\nReturn on Equity: 147.2%",
  "thread_id": "default",
  "stock_output": {
    "price": 175.50,
    "currency": "USD",
    "fundamentals": {
      "market_cap": "2.8T",
      "pe_ratio": 28.5,
      "eps": 6.15,
      "revenue": "383.3B",
      "profit_margin": "25.3%",
      "roe": "147.2%"
    }
  }
}
```

### Web Research
`POST /chat`

Request:
```json
{
  "message": "What are the latest developments in AI regulation?",
  "thread_id": "default"
}
```

Response:
```json
{
  "response": "Recent developments in AI regulation include...",
  "thread_id": "default",
  "websearch_output": {
    "content": "Detailed research results...",
    "sources": ["source1", "source2"],
    "timestamp": "2024-03-21T10:00:00Z"
  }
}
```

### Fund Analysis Endpoints
`POST /chat`

Request:
```json
{
  "message": "What is my exposure to Tesla across my funds? I have 25% in LF Global and 25% in LF USA",
  "thread_id": "default"
}
```

Response:
```json
{
  "response": "Your total exposure to 'Tesla' is approximately 0.1234%...",
  "thread_id": "default",
  "fund_output": {
    "exposure_table": "| Fund | Allocation (%) | Exposure in Fund (%) | Total Exposure (%) |\n|------|-----------------|------------------------|---------------------|\n| LF Global | 25.00 | 0.1234 | 0.0309 |\n| LF USA | 25.00 | 0.3700 | 0.0925 |"
  }
}
```

### Chat Endpoint
`POST /chat`

Request:
```json
{
  "message": "Show me the performance of the S&P 500 for 2023",
  "thread_id": "default"
}
```

Response:
```json
{
  "response": "Here's the S&P 500 performance for 2023...",
  "thread_id": "default",
  "vega_lite_spec": {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "description": "S&P 500 (2023-01-01 to 2023-12-31)",
    "mark": "line",
    "width": 600,
    "height": 400,
    "data": {
      "values": [
        {"date": "2023-01-31", "value": 4076.6},
        // Additional data points
      ]
    },
    "encoding": {
      // Encoding specifications
    }
  }
}
```

## Features

### Portfolio Analysis
- Performance metrics calculation
- Portfolio comparison
- Asset allocation suggestions
- Return analysis by timeframe

### Chart Generation
The system will generate charts based on real market data stored in the market_data.xlsx file. You can refer to indices using their full names or common aliases:

- "70/30" → "70/30 Portfolio"
- "60/40" → "60/40 Portfolio"
- "stocks" → "MSCI World" 
- "bonds" → "US 10 Year Treasury"

### Market Insights
- Economic data analysis
- Market trend identification
- Sector performance comparison

## Troubleshooting

### Chart Data Issues
If charts show incomplete data or have gaps:
- Verify that market_data.xlsx contains data for the requested timeframe
- Use shorter date ranges for more detailed visualization

### Agent Timeout
For complex analyses, the agent might time out. In these cases:
- Break down complex questions into smaller parts
- Specify more precise timeframes or metrics

### Docker Connection Issues
If you see "Docker Desktop connection errors":
- Ensure Docker Desktop is running
- Check Windows Services for "Docker Desktop Service"
- Try restarting the Docker service
