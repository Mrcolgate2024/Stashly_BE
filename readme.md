# Stashly Financial Assistant

A FastAPI-based backend application that powers Stashly's Financial Assistant, featuring an AI-powered financial analyst that provides market insights and analysis through both chat and avatar interfaces.

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
