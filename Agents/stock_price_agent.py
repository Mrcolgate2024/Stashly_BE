import yfinance as yf
from typing import List, Optional, Dict, Any, TypedDict
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from datetime import datetime
import traceback
import pandas as pd

# -------------------------
# Full Shared Graph State
# -------------------------

class GraphState(TypedDict):
    input: str
    thread_id: str
    messages: List[dict]
    portfolio_output: Optional[str]
    market_report_output: Optional[str]
    conversational_output: Optional[str]
    websearch_output: Optional[str]
    stock_output: Optional[str]
    chat_output: Optional[str]
    fund_output: Optional[str]
    market_charts: Optional[List[dict]]
    user_name: Optional[str]
    last_ran_agent: Optional[str]

# -------------------------
# Common Ticker Lookup
# -------------------------

COMMON_COMPANY_MAP = {
    "apple": "AAPL",
    "tesla": "TSLA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "berkshire": "BRK-B"
}

def resolve_ticker(company: str) -> Optional[str]:
    company_lower = company.lower().strip()
    return COMMON_COMPANY_MAP.get(company_lower)

# -------------------------
# Tool: Flexible Stock Query
# -------------------------

@tool
def query_stock_info(user_input: str) -> str:
    """Query stock info like price, fundamentals, dividends, or history for well-known companies."""
    try:
        # Fuzzy name matching
        company_name = None
        for name in COMMON_COMPANY_MAP.keys():
            if name in user_input.lower():
                company_name = name
                break

        if not company_name:
            return "Please specify a well-known company."

        ticker = resolve_ticker(company_name)
        if not ticker:
            return f"Couldn't find stock ticker for '{company_name}'."

        stock = yf.Ticker(ticker)
        info = stock.info
        lower = user_input.lower()

        responses = []

        # 1. Current Price
        if "price" in lower or "current" in lower:
            price = info.get("regularMarketPrice")
            currency = info.get("currency", "USD")
            responses.append(f"The current price of {info.get('shortName')} ({ticker}) is {price} {currency}.")

        # 2. Fundamentals
        if "fundamentals" in lower or "pe" in lower or "eps" in lower or "valuation" in lower:
            fundamentals = {
                "Market Cap": info.get("marketCap"),
                "PE Ratio": info.get("trailingPE"),
                "EPS": info.get("trailingEps"),
                "Revenue": info.get("totalRevenue"),
                "Profit Margin": info.get("profitMargins"),
                "Return on Equity": info.get("returnOnEquity")
            }
            responses.append("Key fundamentals:\n" + "\n".join([f"{k}: {v}" for k, v in fundamentals.items() if v is not None]))

        # 3. Dividend History
        if "dividend" in lower:
            dividends = stock.dividends
            if dividends.empty:
                responses.append(f"{ticker} has no dividend history.")
            else:
                last_div = dividends.iloc[-1]
                last_date = dividends.index[-1].date()
                responses.append(f"Last dividend for {ticker} was {last_div:.2f} on {last_date}")

        # 4. Time Series (basic)
        if "history" in lower or "chart" in lower or "time series" in lower:
            period = "6mo"
            if "1 year" in lower or "year" in lower:
                period = "1y"
            elif "3 month" in lower or "3mo" in lower:
                period = "3mo"
            hist = stock.history(period=period)
            if hist.empty:
                responses.append(f"No historical data for {ticker} over {period}.")
            else:
                last = hist.tail(5)[["Open", "Close"]].round(2)
                responses.append(f"Recent data for {ticker}:\n{last.to_string()}")


        # 5. Default Fallback
        if not responses:
            return f"What financial data about {info.get('shortName')} ({ticker}) would you like?"

        return "\n\n".join(responses)

    except Exception as e:
        return f"Error fetching stock data: {str(e)}"

# -------------------------
# LangChain Agent Setup
# -------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = PromptTemplate.from_template("""
You are a helpful financial assistant that can find stock data like price, fundamentals, time series, or dividends.

Use the tool below if needed.

You must always reason step-by-step using this format:

Thought: your reasoning
Action: query_stock_info
Action Input: the tool input

When you have all the information you need, conclude with:
Final Answer: your full answer to the user

{tool_names}

{tools}

Previous conversation:
{chat_history}

User question: {input}

{agent_scratchpad}
""")

tools: List = [query_stock_info]

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    max_execution_time=120
)

# -------------------------
# LangGraph Entrypoint
# -------------------------

async def run_stock_price_agent(state: dict) -> dict:
    last_message = state["messages"][-1]["content"]
    try:
        result = await agent_executor.ainvoke({"input": last_message})
        print("🔍 RAW AGENT RESULT:", result)
        output = result.get("output", "No output.")
    except Exception as e:
        output = f"Error in stock agent: {str(e)}"

    return {
        **state,
        "stock_output": output,
        "last_ran_agent": "stock",
        "messages": state["messages"] + [{"role": "assistant", "content": output}]
    }

# -------------------------
# Standalone test function
# -------------------------

async def process_user_input(message: str, thread_id: str = "default") -> Dict[str, Any]:
    try:
        state: GraphState = {
            "input": message,
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": message}],
            "portfolio_output": None,
            "market_report_output": None,
            "conversational_output": None,
            "websearch_output": None,
            "chat_output": None,
            "market_charts": None,
            "stock_output": None,
            "user_name": None,
            "last_ran_agent": None
        }

        result = await run_stock_price_agent(state)
        print(result["stock_output"])
        return {"response": result["stock_output"]}

    except Exception as e:
        print(traceback.format_exc())
        return {"response": f"Error: {str(e)}"}

# -------------------------
# Export for graph usage
# -------------------------

__all__ = ["run_stock_price_agent", "agent_executor", "memory"]