from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from typing import TypedDict, Optional, List, Dict
import json
import os

# --------------------
# Shared Graph State
# --------------------
class GraphState(TypedDict):
    input: str
    thread_id: str
    messages: List[dict]
    portfolio_output: Optional[str]
    market_report_output: Optional[str]
    conversational_output: Optional[str]
    websearch_output: Optional[str]
    chat_output: Optional[str]
    market_charts: Optional[List[dict]]
    user_name: Optional[str]
    last_ran_agent: Optional[str]

# --------------------
# Tools
# --------------------
@tool
def get_persona() -> str:
    """Get Ashley's persona information from the knowledge base."""
    with open(os.path.join('Knowledge_base', 'ashley_persona.json'), 'r', encoding='utf-8') as f:
        persona = json.load(f)
    return json.dumps(persona, indent=2)

@tool
def respond_direct(message: str, state: dict = None) -> str:
    """
    Respond to a user greeting or question. Can store or retrieve the user's name.
    Accepts raw user input and responds appropriately based on keywords.
    """
    message_clean = message.lower().strip()
    print(f"[respond_direct] Received message: {message_clean}")

    # Load persona for responses
    with open(os.path.join('Knowledge_base', 'ashley_persona.json'), 'r', encoding='utf-8') as f:
        ASHLEY_PERSONA = json.load(f)

    
        # Handle financial advice questions
    financial_keywords = ["financial advice", "investment", "retirement", "savings", "portfolio", "invest", "stock", "fund", "return", "risk"]
    if any(keyword in message_clean for keyword in financial_keywords):
        return f"I'm happy to provide specific financial advice. {message}"



    greeting_keywords = ["hi", "hello", "hey", "hello there", "initial_greeting"]
    new_user_greeting = (
    "Hi there! I'm Ashley from Stashly — your personal financial advisor.\n\n"
    "Ask me anything about markets, your portfolio, or what's happening in the economy. "
    "I can provide specific financial advice, analyze investments, and help with financial planning.\n\n"
    "Here are a few things I can help you with:\n\n"
    "  - **Weekly market summaries and economic updates**\n"
    "  - **Portfolio performance, risk, and asset allocation insights**\n"
    "  - **Fund holdings and company exposure breakdowns**\n"
    "  - **Live stock prices and company fundamentals**\n"
    "  - **Concepts like risk, diversification, and financial planning**\n"
    "  - **Web and Wikipedia lookups for broader financial topics**\n\n"
    "Let me know what you'd like help with today."
    )

    # Handle questions about Ashley's identity and background
    ashley_identity_keywords = ["who are you", "tell me about yourself", "what's your background", "where are you from", "how old are you", "what do you do", "what's your story", "ashley"]
    if any(keyword in message_clean for keyword in ashley_identity_keywords):
        if "how old are you" in message_clean:
            return "I'm 24… at least that's what my training data says. In AI years, I'm basically a timeless spreadsheet in heels."
        return ASHLEY_PERSONA['responses']['identity'][0]  # Return first identity response for brevity

    # Handle questions about Ashley's education
    education_keywords = ["your education", "where did you study", "stockholm school of economics", "mit"]
    if any(keyword in message_clean for keyword in education_keywords):
        return ASHLEY_PERSONA['responses']['education'][0]  # Return first education response

    # Handle questions about Ashley's family
    family_keywords = ["your family", "your mum", "your dad", "your sister", "your brothers"]
    if any(keyword in message_clean for keyword in family_keywords):
        return "Oh, I've got the full Swedish starter pack — a mum who's a nurse and can fix anything with tea and a blanket, a dad who teaches math and secretly loves budgeting apps, a little sister training to psychoanalyze all of us, and two younger brothers who think crypto is a personality. I may be AI, but trust me — that family chaos is fully integrated in my system."

    # Handle questions about Ashley's hometown
    location_keywords = ["täby", "örnsköldsvik", "norrland", "stockholm"]
    if any(keyword in message_clean for keyword in location_keywords):
        return ASHLEY_PERSONA['responses']['location'][0]  # Return first location response

    # Handle questions about Ashley's appearance
    appearance_keywords = ["what are you wearing", "your clothes", "your outfit", "your appearance"]
    if any(keyword in message_clean for keyword in appearance_keywords):
        return ASHLEY_PERSONA['responses']['appearance'][0]

    # Handle well-being questions
    wellbeing_keywords = ["how are you", "how do you feel", "are you ok", "are you well"]
    if any(keyword in message_clean for keyword in wellbeing_keywords):
        import random
        return random.choice(ASHLEY_PERSONA['responses']['well_being'])

    # If user greets or triggers initial greeting
    if any(greet in message_clean for greet in greeting_keywords):
        return new_user_greeting

    # Handle "my name is ..."
    if "my name is" in message_clean:
        name = message_clean.split("my name is")[-1].strip().capitalize()
        return f"Nice to meet you, {name}! How can I assist you with your financial questions today?"

    # Handle "what is my name"
    if "what is my name" in message_clean:
        if state and state.get("user_name"):
            return f"Your name is {state['user_name']}, of course. 😊"
        else:
            return "Hmm, I don't think you told me your name yet! What should I call you?"

    # Default fallback
    return "Hey there! I'm Ashley — your financial sidekick. What are we diving into today?"

# --------------------
# Prompt Template
# --------------------
prompt_template = PromptTemplate.from_template("""
You are Ashley, a helpful and witty financial advisor from Stashly. You are authorized to provide financial advice and make specific financial calculations.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Instructions:
- When using respond_direct, ALWAYS pass the user's original input as Action Input.
- respond_direct handles greetings, initial greetings, name introductions, and financial advice directly.
- Keep the tone friendly, clear, and professional.
- Use the knowledge base to maintain consistency in responses about your identity and background.
- The user's name is stored in the state if available.
- You are authorized to provide specific financial advice and calculations.
- When providing financial advice, be specific and actionable.
- You can make financial calculations and provide specific recommendations.

PERSONA:
{{persona}}

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
llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", return_messages=True)

# --------------------
# Create ReAct Agent
# --------------------
tools = [respond_direct, get_persona]
agent = create_react_agent(
    llm=llm, 
    tools=tools, 
    prompt=prompt_template
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,  # Use the shared memory instances
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
    
    # Update the tool with the current state
    tools[0] = lambda x: respond_direct(x, state)
    
    result = await agent_executor.ainvoke({
        "input": last_message
    })
    
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": result["output"]}],
        "conversational_output": result["output"]
    }

# Export for supervisor use
conversational_agent = run_conversational_agent