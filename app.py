from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import uvicorn
from Agents.portfolio_agent import agent_executor, process_user_input, filter_messages
from fastapi.middleware.cors import CORSMiddleware
import traceback
import logging
import json
import time
import asyncio
import httpx
from langchain.agents import AgentExecutor, create_react_agent
from supervisor_agent import process_user_input
# Update imports to use new file names
# from supervisor_agent import agent_supervisor, process_user_input
# from portfolio_agent import PortfolioAgent, memory  # was ReactAgent
# from market_report_agent import MarketReportAgent  # was MarketSummary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stashly's Financial Coach API",
    description="API for interacting with Stashly's Financial Coach. Use the /chat/completions endpoint for streaming or /chat for regular responses.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stashly.lovable.app", "http://localhost:3000", "http://localhost:5173",  "http://localhost:8080", "http://localhost:8081"],  # frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add this constant at the top with other imports
MAX_CONVERSATION_TURNS = 5  # Adjust this number based on your needs

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None
    thread_id: Optional[str] = "default"

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"
    user_name: Optional[str] = None
    message_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    vega_lite_spec: Optional[Dict[str, Any]] = None

async def stream_response(response_text: str):
    """
    Stream the response text in small chunks with proper formatting for OpenAI compatibility.
    """
    # Split into sentences first, then into smaller chunks
    sentences = response_text.replace('. ', '.|').split('|')
    
    for sentence_idx, sentence in enumerate(sentences):
        # Split sentence into smaller chunks (2-3 words each)
        words = sentence.split()
        chunks = [' '.join(words[i:i+2]) for i in range(0, len(words), 2)]
        
        for chunk_idx, chunk in enumerate(chunks):
            # Add proper punctuation and spacing
            if chunk_idx == len(chunks) - 1 and sentence.endswith('.'):
                chunk = chunk.rstrip() + '. '
            else:
                chunk = chunk + ' '

            data = {
                "id": f"chatcmpl-{sentence_idx}-{chunk_idx}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.15)  # Slightly longer delay for better speech
    
    # Send the final completion message
    final_data = {
        "id": f"chatcmpl-final",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_data)}\n\n"
    
    # Send the DONE marker
    yield "data: [DONE]\n\n"

@app.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        request_data = await request.json()
        messages = request_data.get("messages", [])
        thread_id = request_data.get("thread_id", "default")
        
        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Process the message (memory is handled in react_agent.py)
        response = await process_user_input(user_message)
        
        # Stream the response
        async for chunk in stream_response(response):
            yield chunk
        
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        # Process the message through the supervisor agent
        result = await process_user_input(
            message=request.message,
            thread_id=request.thread_id
        )
        
        # Debug the result
        print(f"API response: {json.dumps({k: v is not None for k, v in result.items()}, indent=2)}")
        
        # Ensure vega_lite_spec is included in the response
        if 'vega_lite_spec' not in result:
            result['vega_lite_spec'] = None
            
        return ChatResponse(
            response=result["response"],
            thread_id=result["thread_id"],
            vega_lite_spec=result.get("vega_lite_spec")
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{thread_id}")
async def get_conversation_history(thread_id: str):
    """Get the conversation history for a specific thread."""
    messages = memory.load_memory_variables({}).get("history", [])
    return {"history": messages}

@app.delete("/conversations/{thread_id}")
async def clear_conversation_history(thread_id: str):
    """Clear the conversation history for a specific thread."""
    memory.clear()
    return {"message": f"Conversation history cleared for thread {thread_id}"}

@app.post("/proxy/simli/startE2ESession")
async def proxy_simli_session(request: Request):
    """
    Proxy endpoint to forward requests to Simli API to avoid CORS issues
    """
    try:
        # Get the request body
        body = await request.json()
        logger.info(f"Proxying request to Simli API with body: {json.dumps(body)}")
        
        # Validate required fields for Simli API
        required_fields = ["apiKey", "faceId", "voiceId"]
        missing_fields = [field for field in required_fields if field not in body or not body[field]]
        
        if missing_fields:
            error_msg = f"Missing required fields for Simli API: {', '.join(missing_fields)}"
            logger.error(error_msg)
            return JSONResponse(
                content={"error": "Missing required fields", "details": error_msg},
                status_code=400
            )
        
        # Ensure speech parameters are set
        if "enableSpeech" not in body:
            body["enableSpeech"] = True
            logger.info("Added enableSpeech parameter to request")
        
        if "disableDefaultResponses" not in body:
            body["disableDefaultResponses"] = False
            logger.info("Added disableDefaultResponses parameter to request")
            
        if "enableAutoResponse" not in body:
            body["enableAutoResponse"] = True
            logger.info("Added enableAutoResponse parameter to request")
            
        if "firstMessage" not in body or not body["firstMessage"]:
            body["firstMessage"] = "Hello, I'm your financial analyst from Stashly. How can I assist you today?"
            logger.info("Added firstMessage parameter to request")
        
        # Forward the request to Simli API
        simli_url = "https://api.simli.ai/startE2ESession"
        logger.info(f"Attempting to connect to Simli API at: {simli_url}")
        
        # Set up client with more detailed settings
        async with httpx.AsyncClient(timeout=30.0, verify=True) as client:
            try:
                # Proceed with the actual request
                logger.info("Sending request to Simli API...")
                response = await client.post(
                    simli_url,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "Stashly-Backend/1.0"
                    },
                    timeout=30.0
                )
                
                # Log the response status and content
                logger.info(f"Simli API response status: {response.status_code}")
                
                # Handle 400 Bad Request specifically
                if response.status_code == 400:
                    try:
                        error_content = response.json()
                        logger.error(f"Simli API returned 400 Bad Request: {json.dumps(error_content)}")
                        
                        # Return the error response with additional debugging info
                        return JSONResponse(
                            content={
                                "error": "Bad Request to Simli API",
                                "simli_response": error_content,
                                "request_body": body,
                                "message": "The Simli API rejected the request. Please check the API key, faceId, and voiceId values."
                            },
                            status_code=400
                        )
                    except Exception as json_error:
                        # If response is not JSON, return the raw text
                        error_text = response.text
                        logger.error(f"Simli API returned 400 Bad Request with non-JSON response: {error_text}")
                        return JSONResponse(
                            content={
                                "error": "Bad Request to Simli API",
                                "details": error_text,
                                "request_body": body,
                                "message": "The Simli API rejected the request with a non-JSON response."
                            },
                            status_code=400
                        )
                
                try:
                    content = response.json()
                    logger.info(f"Simli API response content: {json.dumps(content)}")
                    return JSONResponse(
                        content=content,
                        status_code=response.status_code
                    )
                except Exception as json_error:
                    # If response is not JSON, return the raw text
                    logger.error(f"Error parsing Simli API response as JSON: {str(json_error)}")
                    logger.info(f"Raw response text: {response.text}")
                    return JSONResponse(
                        content={"error": "Invalid JSON response from Simli API", "details": response.text},
                        status_code=500
                    )
                    
            except httpx.ConnectError as connect_error:
                # Handle connection errors specifically
                error_msg = f"Connection error when calling Simli API: {str(connect_error)}"
                logger.error(error_msg)
                # Try to get more network diagnostic information
                try:
                    import socket
                    host = "api.simli.ai"
                    logger.info(f"Attempting to resolve DNS for {host}...")
                    ip_address = socket.gethostbyname(host)
                    logger.info(f"DNS resolution successful: {host} -> {ip_address}")
                except Exception as dns_error:
                    logger.error(f"DNS resolution failed: {str(dns_error)}")
                
                return JSONResponse(
                    content={"error": "Connection error", "details": str(connect_error)},
                    status_code=500
                )
                
            except httpx.TimeoutException as timeout_error:
                # Handle timeout errors specifically
                error_msg = f"Timeout when calling Simli API: {str(timeout_error)}"
                logger.error(error_msg)
                return JSONResponse(
                    content={"error": "Request timed out", "details": str(timeout_error)},
                    status_code=504  # Gateway Timeout
                )
                
            except httpx.RequestError as req_error:
                # Handle other request errors
                error_msg = f"Request error when calling Simli API: {str(req_error)}"
                logger.error(error_msg)
                return JSONResponse(
                    content={"error": "Request error", "details": str(req_error)},
                    status_code=500
                )
                
    except Exception as e:
        error_detail = f"Error proxying request to Simli API: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        return JSONResponse(
            content={"error": str(e), "traceback": traceback.format_exc()},
            status_code=500
        )

@app.post("/simli/connect")
async def simli_connect(request: Request):
    """
    Endpoint specifically for connecting to Simli avatar
    """
    try:
        # Get the request body
        body = await request.json()
        logger.info(f"Simli connect request with body: {json.dumps(body)}")
        
        # Extract API key, face ID, and voice ID from request or use defaults
        api_key = body.get("apiKey", "")
        face_id = body.get("faceId", "")
        voice_id = body.get("voiceId", "")
        
        # Log the connection attempt
        logger.info(f"Attempting to connect to Simli with API key: {api_key[:5]}..., face ID: {face_id}, voice ID: {voice_id}")
        
        # Prepare the request to Simli API with additional parameters for speech
        simli_request = {
            "apiKey": api_key,
            "faceId": face_id,
            "voiceId": voice_id,
            "firstMessage": "Hello, I'm your financial analyst from Stashly. How can I assist you today?",
            "systemPrompt": "You are a helpful financial analyst assistant. Respond briefly and clearly.",
            "disableDefaultResponses": False,
            "enableAutoResponse": True,
            "enableSpeech": True
        }
        
        # Forward the request to Simli API
        simli_url = "https://api.simli.ai/startE2ESession"
        logger.info(f"Connecting to Simli API at: {simli_url}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Make the request to Simli API
                response = await client.post(
                    simli_url,
                    json=simli_request,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "Stashly-Backend/1.0"
                    },
                    timeout=30.0
                )
                
                # Log the response
                logger.info(f"Simli API response status: {response.status_code}")
                
                try:
                    # Parse the response as JSON
                    content = response.json()
                    logger.info(f"Simli API response content: {json.dumps(content)}")
                    
                    # Return the response from Simli API
                    return JSONResponse(
                        content={
                            "status": "success",
                            "simli_response": content,
                            "message": "Successfully connected to Simli API"
                        },
                        status_code=200
                    )
                except Exception as json_error:
                    # If the response is not JSON, return the error
                    logger.error(f"Error parsing Simli API response: {str(json_error)}")
                    return JSONResponse(
                        content={
                            "status": "error",
                            "message": "Failed to parse Simli API response",
                            "error": str(json_error),
                            "raw_response": response.text
                        },
                        status_code=500
                    )
            except Exception as request_error:
                # If the request fails, return the error
                logger.error(f"Error connecting to Simli API: {str(request_error)}")
                return JSONResponse(
                    content={
                        "status": "error",
                        "message": "Failed to connect to Simli API",
                        "error": str(request_error)
                    },
                    status_code=500
                )
    except Exception as e:
        # If there's an error processing the request, return the error
        logger.error(f"Error processing Simli connect request: {str(e)}")
        return JSONResponse(
            content={
                "status": "error",
                "message": "Failed to process Simli connect request",
                "error": str(e)
            },
            status_code=500
        )

# Add a simple test endpoint
@app.get("/test")
async def test_endpoint():
    """
    Simple test endpoint to check if the API is running
    """
    return {
        "status": "success",
        "message": "API is running correctly",
        "timestamp": time.time()
    }

# Add a test endpoint to check Simli API with sample data
@app.get("/test/simli-api")
async def test_simli_api():
    """
    Test endpoint to check Simli API with sample data
    """
    try:
        logger.info("Testing Simli API with sample data...")
        
        # Sample data for testing - replace with valid values for your account
        test_data = {
            "apiKey": "sample_key_replace_with_valid",  # Replace with your actual API key
            "faceId": "sample_face_id_replace_with_valid",  # Replace with a valid face ID
            "voiceId": "sample_voice_id_replace_with_valid",  # Replace with a valid voice ID
            "firstMessage": "",
            "systemPrompt": "",
            "disableDefaultResponses": True
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # Try to connect to the Simli API
                response = await client.post(
                    "https://api.simli.ai/startE2ESession",
                    json=test_data,
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )
                
                try:
                    content = response.json()
                    return {
                        "status": "response_received",
                        "status_code": response.status_code,
                        "content": content,
                        "message": f"Received response from Simli API with status code {response.status_code}"
                    }
                except Exception as json_error:
                    return {
                        "status": "non_json_response",
                        "status_code": response.status_code,
                        "raw_content": response.text,
                        "message": f"Received non-JSON response from Simli API with status code {response.status_code}"
                    }
            except Exception as e:
                # If connection fails, return error details
                return {
                    "status": "error",
                    "message": f"Failed to connect to Simli API: {str(e)}",
                    "error_type": type(e).__name__
                }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error during API test: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.get("/test-architecture")
def test_architecture():
    import platform
    return {
        "architecture": platform.architecture(),
        "machine": platform.machine(),
        "system": platform.system()
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to Stashly's Financial Coach API",
        "description": "API for interacting with Stashly's Financial Coach. Use the /chat/completions endpoint for streaming or /chat for regular responses.",
        "version": "1.0.0",
        "endpoints": {
            "/chat/completions": "Streaming chat interface (OpenAI compatible)",
            "/chat": "Main chat interface",
            "/conversations/{thread_id}": "Get conversation history",
            "/conversations/{thread_id} (DELETE)": "Clear conversation history",
            "/test": "Simple test endpoint to check if the API is running",
            "/simli/connect": "Connect to Simli avatar",
            "/proxy/simli/startE2ESession": "Proxy endpoint for Simli API",
            "/test/simli-api": "Test Simli API connection with sample data",
            "/test-architecture": "Test architecture information"
        },
        "agents": {
            "market_report_agent": "Generates comprehensive weekly market reports",
            "portfolio_agent": "Analyzes portfolio data and performance",
            "supervisor_agent": "Routes queries to appropriate specialized agents"
        }
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)