from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph
from datetime import datetime
import traceback
import json
import re

# Agent handlers
from Agents.portfolio_agent import run_portfolio_agent
from Agents.market_report_agent import run_market_report_agent
from Agents.conversational_agent import run_conversational_agent
from Agents.websearch_agent import run_websearch_agent
from Agents.stock_price_agent import run_stock_price_agent  # ✅ New stock agent
from Agents.fund_position_agent import run_fund_position_agent  # ✅ Fund position agent

# ---- Shared State ----

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
    fund_output: Optional[str]  # ✅ Added fund output
    market_charts: Optional[List[dict]]
    user_name: Optional[str]
    last_ran_agent: Optional[str]

# ---- Routing Logic ----

def route_message_node(state: GraphState) -> GraphState:
    return state

def route_decision(state: GraphState) -> str:
    message = state["input"].lower()
    print(f"Routing message: {message}")

    # # Check if this is a follow-up question
    # follow_up_keywords = [
    #     ""

    # ]
    
    # # Get the last message from the conversation
    # last_message = state.get("messages", [{}])[-1].get("content", "").lower()
    
    # # Check if this is a follow-up by looking at both the current message and context
    # is_follow_up = (
    #     any(keyword in message for keyword in follow_up_keywords) or
    #     (message in ["when is this for?", "what was the price again?"]) or
    #     (last_message and any(word in last_message for word in ["price", "stock", "market", "fund", "portfolio"]))
    # )
    
    # last_agent = state.get("last_ran_agent")

    # # If this is a follow-up question and we have a last agent, use that agent
    # if is_follow_up and last_agent:
    #     print(f"Routing to {last_agent} agent (follow-up question)")
    #     return last_agent

    # Keyword categories
    market_keywords = ["market", "report", "weekly", "analysis", "economy", "trends", "economic"]
    portfolio_keywords = ["portfolio", "70/30", "60/40", "50/50", "return", "performance", "time series", "index", "ppm", "chart", "visualize", "graph"]
    fund_keywords = ["exposure", "owns", "fund risk", "risk level", "fund volatility", "holding", "fund owns", "who holds", "do i have exposure", "fund position", "fund", "allocation", "holdings", "funds", "isin", "sector", "industry", "risk", "volatility", "standard deviation", "tracking error", "active risk", "risk profile", "risk assessment", "risk analysis", "risk metrics"]
    stock_keywords = ["stock", "price", "fundamentals", "ticker", "dividend", "pe ratio", "eps", "valuation", "quote", "history"]
    websearch_keywords = ["search", "look up", "find", "web", "online", "google", "weather", "news", "retirement", "financial advice", "investment", "portfolio", "invest", "stock", "fund", "return", "risk", "savings", "spending"]
    greeting_keywords = ["hello", "hello there", "hi there", "hi", "hey", "bjorn", "name is", "greetings", "initial_greeting"]
    portfolio_names = ["70/30", "60/40", "50/50"]
    ashley_keywords = ["who are you", "tell me about yourself", "what's your background", "where are you from", "how old are you", "what do you do", "what's your story", "ashley", "your background", "your education", "your family", "how are you", "how do you feel", "are you ok", "are you well", "what are you wearing", "your clothes", "your outfit", "your appearance"]

    # Personal questions about Ashley (highest priority) 
    if any(keyword in message for keyword in greeting_keywords):
        print("Routing to conversational agent (greeting)")
        return "chat"

    # Greetings (moved to second priority)
    if any(keyword in message for keyword in ashley_keywords):
        print("Routing to conversational agent (personal questions)")
        return "chat"

    # Stock agent routing
    if any(keyword in message for keyword in stock_keywords):
        print("Routing to stock agent (keyword match)")
        return "stock"

    # Fund position agent routing
    if any(keyword in message for keyword in fund_keywords):
        print("Routing to fund position agent (keyword match)")
        return "fund"

    # Market report
    if any(keyword in message for keyword in market_keywords):
        print("Routing to market agent (keyword match)")
        return "market"

    # Portfolio
    if any(keyword in message for keyword in portfolio_keywords + portfolio_names):
        print("Routing to portfolio agent (keyword or name match)")
        return "portfolio"

    # Previous context fallback
    if state.get("stock_output"):
        return "stock"
    if state.get("fund_output"):
        return "fund"
    if state.get("portfolio_output"):
        return "portfolio"
    if state.get("market_report_output"):
        return "market"
    if state.get("websearch_output"):
        return "websearch"
    if state.get("conversational_output"):
        return "chat"

    # Web search fallback
    if any(keyword in message for keyword in websearch_keywords):
        print("Routing to websearch agent (explicit search terms)")
        return "websearch"

    # Default fallback
    print("Default routing to conversational agent")
    return "chat"

# ---- Create LangGraph ----

def create_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route", route_message_node)
    graph.add_node("portfolio", run_portfolio_agent)
    graph.add_node("market", run_market_report_agent)
    graph.add_node("chat", run_conversational_agent)
    graph.add_node("websearch", run_websearch_agent)
    graph.add_node("stock", run_stock_price_agent)
    graph.add_node("fund", run_fund_position_agent)  # ✅ New node
    graph.add_node("end", lambda state: state)

    graph.add_conditional_edges(
        "route",
        route_decision,
        {
            "portfolio": "portfolio",
            "market": "market",
            "chat": "chat",
            "websearch": "websearch",
            "stock": "stock",
            "fund": "fund"
        }
    )

    def should_continue(state: GraphState) -> bool:
        return False

    for node in ["portfolio", "market", "chat", "websearch", "stock", "fund"]:
        graph.add_conditional_edges(
            node,
            should_continue,
            {False: "end"}
        )

    graph.set_entry_point("route")
    return graph.compile()

# ---- Entrypoint Function ----

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
            "stock_output": None,
            "chat_output": None,
            "fund_output": None,
            "market_charts": None,
            "user_name": None,
            "last_ran_agent": None
        }

        graph = create_graph()
        result = await graph.ainvoke(state)

        print("LangGraph result:", json.dumps(result, indent=2))

        # Handle the response text
        if result.get("market_report_output"):
            text = result["market_report_output"]
            last_agent = "market"
        elif result.get("portfolio_output"):
            text = result["portfolio_output"]
            last_agent = "portfolio"
        elif result.get("stock_output"):
            text = result["stock_output"]
            last_agent = "stock"
        elif result.get("fund_output"):
            text = result["fund_output"]
            last_agent = "fund"
        elif result.get("conversational_output"):
            # For conversational output, ensure we get the full message
            text = result["conversational_output"]
            if message == "initial_greeting":
                text = (
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
            last_agent = "chat"
        elif result.get("websearch_output"):
            text = result["websearch_output"]
            last_agent = "websearch"
        else:
            text = "I couldn't process your request. Try rephrasing?"
            last_agent = None

        chart = result.get("market_charts", [None])[0] if result.get("market_charts") else None

        final_response = {
            "response": text,
            "thread_id": thread_id,
            "vega_lite_spec": chart
        }

        result["last_ran_agent"] = last_agent
        print(f"Final response structure: {json.dumps({k: v is not None for k, v in final_response.items()}, indent=2)}")
        print(f"Vega spec present: {chart is not None}")
        print(f"Last ran agent: {last_agent}")

        return final_response

    except Exception as e:
        print("Error processing message:\n", traceback.format_exc())
        return {
            "response": f"Sorry, something went wrong: {str(e)}",
            "thread_id": thread_id,
            "vega_lite_spec": None
        }