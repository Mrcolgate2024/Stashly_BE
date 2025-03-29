# Stashly Financial Assistant

A conversational AI assistant that provides financial analysis, portfolio management guidance, and market insights.

## Recent Updates

- 📊 Improved chart generation using real market data from market_data.xlsx
- 🔍 Enhanced portfolio analysis with support for common portfolio aliases (70/30, 60/40, etc.)
- 🚀 Better error handling and response formatting
- ⏱️ Optimized agent performance with proper timeout handling

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
- **Multiple Endpoints**:
  - Streaming chat completions (OpenAI compatible)
  - Regular chat interface
  - Avatar connection and management
  - Market data analysis
  - Conversation history management

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
