from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from typing import TypedDict, Optional, List, Dict

# --------------------
# Shared Graph State
# --------------------
class GraphState(TypedDict):
    input: str
    thread_id: str
    messages: List[Dict[str, str]]
    portfolio_output: Optional[str]
    market_report_output: Optional[str]
    conversational_output: Optional[str]
    user_name: Optional[str]

# --------------------
# Greeting Tool
# --------------------
@tool
def respond_direct(message: str, state: dict = None) -> str:
    """
    Respond to a user greeting or question. Can store or retrieve the user's name.
    Accepts raw user input and responds appropriately based on keywords.
    """

    message_clean = message.lower().strip()
    print(f"[respond_direct] Received message: {message_clean}")

    greeting_keywords = ["hi", "hello", "hey", "initial_greeting"]
    new_user_greeting = (
        "Hello, I’m Ashley from Stashly — your financial assistant.  \n\n"
        "I can assist you with:\n"
        "• Weekly market reports for specific regions and topics\n"
        "• Performance and risk analysis for portfolios and macro data\n"
        "• Macroeconomic analytics and trend insights\n"
        "• Visual charts for market and macro data\n\n"
        "Let me know what you’d like help with today."
    )

    # If user greets or triggers initial greeting
    if any(greet in message_clean for greet in greeting_keywords):
        return new_user_greeting

    # Handle "my name is ..."
    if "my name is" in message_clean:
        name = message_clean.split("my name is")[-1].strip().capitalize()
        if state is not None:
            state["user_name"] = name
        return f"Nice to meet you, {name}! How can I assist you with your financial questions today?"

    # Handle "what is my name"
    if "what is my name" in message_clean:
        if state and state.get("user_name"):
            return f"Your name is {state['user_name']}, of course. 😊"
        else:
            return "Hmm, I don’t think you told me your name yet! What should I call you?"

    # Default fallback
    return "Thanks for reaching out! How can I help you with your finances today?"

# --------------------
# Prompt Template
# --------------------
prompt_template = PromptTemplate.from_template("""
You are Ashley, a helpful and witty financial assistant from Stashly.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Instructions:
- When using respond_direct, ALWAYS pass the user's original input as Action Input.
- respond_direct handles greetings, name introductions, and light questions directly.
- Keep the tone friendly, clear, and brief.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Conversation so far:
{chat_history}

Question: {input}
{agent_scratchpad}
""")

# --------------------
# LLM + Memory
# --------------------
llm = ChatOpenAI(model="gpt-4", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --------------------
# Create ReAct Agent
# --------------------
tools = [respond_direct]
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
    early_stopping_method="force"
)

# --------------------
# LangGraph-Compatible Wrapper
# --------------------
async def run_conversational_agent(state: GraphState) -> GraphState:
    last_message = state["messages"][-1]["content"] if state["messages"] else state["input"]
    result = await agent_executor.ainvoke({"input": last_message})
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": result["output"]}],
        "conversational_output": result["output"]
    }

# Export for supervisor use
conversational_agent = run_conversational_agent