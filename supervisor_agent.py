from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph
from datetime import datetime
import traceback
import json

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

    # Keyword categories
    market_keywords = ["market", "report", "weekly", "analysis", "economy", "trends", "economic"]
    portfolio_keywords = ["portfolio", "70/30", "60/40", "50/50", "return", "performance", "time series", "index", "ppm", "chart", "visualize", "graph"]
    fund_keywords = ["exposure", "owns", "holding", "fund owns", "who holds", "do i have exposure", "fund position", "fund", "allocation", "holdings", "funds"]
    stock_keywords = ["stock", "price", "fundamentals", "ticker", "dividend", "pe ratio", "eps", "valuation", "quote", "history"]
    websearch_keywords = ["search", "look up", "find", "web", "online", "google", "weather", "news"]
    greeting_keywords = ["hello", "hi", "hey", "bjorn", "name is", "greetings", "initial_greeting"]
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
            text = result["conversational_output"]
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
