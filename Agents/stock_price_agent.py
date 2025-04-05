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
import time

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

# Quick lookup map for common companies
COMPANY_MAP = {
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
    "berkshire": "BRK-B",
    "swedbank": "SWED-A.ST",
    "ericsson": "ERIC-B.ST",
    "volvo": "VOLV-B.ST",
    "hm": "HM-B.ST",
    "nordea": "NDA-FI.HE"
}

def resolve_ticker(company: str) -> Optional[str]:
    """Resolve company name to ticker symbol with exchange suffix if needed."""
    company_lower = company.lower().strip()
    
    # Check common company map first
    if company_lower in COMPANY_MAP:
        return COMPANY_MAP[company_lower]
    
    # Handle Swedish stocks with .ST suffix
    if company_lower.endswith('.st'):
        return company_lower.upper()
    elif company_lower.endswith(' ab'):
        return f"{company_lower[:-3].strip().upper()}.ST"
    elif company_lower.endswith(' a'):
        return f"{company_lower[:-2].strip().upper()}-A.ST"
    elif company_lower.endswith(' b'):
        return f"{company_lower[:-2].strip().upper()}-B.ST"
    
    return None

def lookup_ticker(company_name: str) -> Optional[str]:
    """
    Look up a company's ticker symbol using yfinance's search functionality.
    """
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Try direct search with the company name
            try:
                # First try the exact ticker if it looks like one
                if any(suffix in company_name.upper() for suffix in ['.ST', '.HE', '.OL', '.CO']):
                    ticker = company_name.upper()
                    search_results = yf.Ticker(ticker)
                    info = search_results.info
                    if info and 'symbol' in info:
                        return info['symbol']
                
                # Try with common exchange suffixes
                exchanges = ['', '.ST', '-A.ST', '-B.ST', '.HE', '.OL', '.CO']
                for suffix in exchanges:
                    try:
                        ticker = f"{company_name.upper()}{suffix}"
                        search_results = yf.Ticker(ticker)
                        info = search_results.info
                        if info and 'symbol' in info:
                            return info['symbol']
                    except Exception as e:
                        print(f"Exchange suffix search failed for {ticker}: {str(e)}")
                        continue
                        
                return None
            except Exception as e:
                print(f"Direct search failed: {str(e)}")
                return None
                
        except Exception as e:
            if "Invalid Crumb" in str(e) and attempt < max_retries - 1:
                print(f"Yahoo Finance API error (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(retry_delay)
                continue
            print(f"Error in lookup_ticker: {str(e)}")
            return None
    return None

# -------------------------
# Tool: Flexible Stock Query
# -------------------------

@tool
def query_stock_info(user_input: str) -> str:
    """Query stock info like price, fundamentals, dividends, or history for any company."""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Get current local time
            local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # First check if the input contains a known ticker
            ticker = None
            for known_ticker in COMPANY_MAP.values():
                if known_ticker.lower() in user_input.lower():
                    ticker = known_ticker
                    break
            
            # If no known ticker found, try to extract company name
            if not ticker:
                words = user_input.lower().split()
                company_words = [w for w in words if w not in ['price', 'stock', 'share', 'of', 'the', 'what', 'is', 'current', 'latest']]
                company_name = ' '.join(company_words)
                
                if company_name:
                    # First check the map
                    if company_name in COMPANY_MAP:
                        ticker = COMPANY_MAP[company_name]
                    else:
                        # If not in map, try dynamic lookup
                        ticker = lookup_ticker(company_name)
            
            if not ticker:
                return "Please specify a company name or ticker symbol. For Swedish stocks, include the exchange suffix (e.g., SWED-A.ST)."

            stock = yf.Ticker(ticker)
            info = stock.info
            lower = user_input.lower()

            responses = []

            # 1. Current Price
            if "price" in lower or "current" in lower or not responses:
                price = info.get("regularMarketPrice")
                currency = info.get("currency", "USD")
                market_time = info.get("regularMarketTime")
                
                if market_time:
                    market_time = datetime.fromtimestamp(market_time).strftime("%Y-%m-%d %H:%M:%S")
                    responses.append(f"The price of {info.get('shortName', ticker)} ({ticker}) is {price} {currency} as of market time {market_time} (your local time: {local_time}).")
                else:
                    responses.append(f"The price of {info.get('shortName', ticker)} ({ticker}) is {price} {currency} (latest available data, your local time: {local_time}).")

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
                responses.append(f"Key fundamentals (as of your local time {local_time}):\n" + "\n".join([f"{k}: {v}" for k, v in fundamentals.items() if v is not None]))

            # 3. Dividend History
            if "dividend" in lower:
                dividends = stock.dividends
                if dividends.empty:
                    responses.append(f"{ticker} has no dividend history.")
                else:
                    last_div = dividends.iloc[-1]
                    last_date = dividends.index[-1].date()
                    responses.append(f"Last dividend for {ticker} was {last_div:.2f} on {last_date} (checked at your local time: {local_time})")

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
                    responses.append(f"Recent data for {ticker} (checked at your local time: {local_time}):\n{last.to_string()}")

            return "\n\n".join(responses)
            
        except Exception as e:
            if "Invalid Crumb" in str(e) and attempt < max_retries - 1:
                print(f"Yahoo Finance API error (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(retry_delay)
                continue
            return f"Error fetching stock data: {str(e)}"

# -------------------------
# LangChain Agent Setup
# -------------------------

# Create a persistent memory instance
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = PromptTemplate.from_template("""
You are a helpful financial assistant that can find stock data like price, fundamentals, time series, or dividends.

Use the tool below if needed.

You must always reason step-by-step using this format:

Thought: your reasoning
Action: query_stock_info
Action Input: the tool input

When you have all the information you need, conclude with:
Final Answer: your full answer to the user

IMPORTANT: Always include both the market time and local time in your responses when providing stock prices or other time-sensitive information.

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
    memory=memory,  # Use the shared memory instance
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    max_execution_time=120
)

# -------------------------
# LangGraph Entrypoint
# -------------------------

async def run_stock_price_agent(state: dict) -> dict:
    try:
        # Get the last message and chat history
        last_message = state["messages"][-1]["content"]
        
        # Convert messages to the format expected by the agent
        chat_history = []
        for msg in state.get("messages", []):
            if msg["role"] == "user":
                chat_history.append({"type": "human", "data": {"content": msg["content"]}})
            elif msg["role"] == "assistant":
                chat_history.append({"type": "ai", "data": {"content": msg["content"]}})

        # Run the agent with chat history
        result = await agent_executor.ainvoke({
            "input": last_message,
            "chat_history": chat_history
        })
        
        print("🔍 RAW AGENT RESULT:", result)
        output = result.get("output", "No output was returned.")

        # Update the state with all necessary fields
        updated_state = {
            **state,  # Preserve existing state
            "stock_output": output,
            "last_ran_agent": "stock",
            "messages": state["messages"] + [{"role": "assistant", "content": output}]
        }

        # Make sure we're not returning any keys that aren't in the GraphState schema
        return {k: v for k, v in updated_state.items() if k in GraphState.__annotations__}

    except Exception as e:
        print(f"Error in stock agent: {str(e)}")
        traceback.print_exc()
        return {
            **state,
            "stock_output": f"Error in stock agent: {str(e)}",
            "last_ran_agent": "stock"
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