from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from react_agent import agent, HumanMessage
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Yoda's Galactic Feast API",
    description="API for interacting with Yoda's restaurant chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat with Yoda's restaurant chatbot.
    
    Parameters:
    - message: The user's message
    - thread_id: Optional thread ID for maintaining conversation context (defaults to "default")
    
    Returns:
    - response: Yoda's response
    - thread_id: The thread ID used for the conversation
    """
    try:
        # Process the message through the agent
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.thread_id}}
        )
        
        # Extract the last message from the result
        if result.get("messages"):
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                return ChatResponse(
                    response=last_message.content,
                    thread_id=request.thread_id
                )
        
        raise HTTPException(status_code=500, detail="No response generated")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Welcome endpoint that returns basic information about the API."""
    return {
        "message": "Welcome to Yoda's Galactic Feast API",
        "description": "Use the /chat endpoint to talk to Yoda",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
