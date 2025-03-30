from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph
from datetime import datetime
import traceback
import json

# Import correct callable agent handlers
from Agents.portfolio_agent import run_portfolio_agent
from Agents.market_report_agent import run_market_report_agent
from Agents.conversational_agent import run_conversational_agent  # ✅ Make sure this is a callable async def

# Define shared state for LangGraph
class GraphState(TypedDict):
    input: str
    thread_id: str
    messages: List[dict]
    portfolio_output: Optional[str]
    market_report_output: Optional[str]
    conversational_output: Optional[str]
    user_name: Optional[str]
    chat_output: Optional[str]
    market_charts: Optional[List[dict]]

# --- ROUTING NODE (supervisor logic) ---

def route_message_node(state: GraphState) -> GraphState:
    return state  # No change, just a placeholder node

def route_decision(state: GraphState) -> str:
    message = state["input"].lower()
    print(f"Routing message: {message}")

    market_keywords = ["market", "report", "weekly", "analysis", "economy", "trends", "economic"]
    portfolio_keywords = ["portfolio", "return", "allocation", "holdings", "fund", "ppm", "chart", "visualize", "graph"]
    portfolio_names = ["70/30", "60/40", "50/50"]

    # Check for explicit keywords
    if any(keyword in message for keyword in market_keywords):
        print("Routing to market agent (keyword match)")
        return "market"

    if any(keyword in message for keyword in portfolio_keywords + portfolio_names):
        print("Routing to portfolio agent (keyword or name match)")
        return "portfolio"

    # Check previous context to route follow-up messages correctly
    last_output_keys = ["portfolio_output", "market_report_output"]
    if state.get("portfolio_output"):
        print("Routing to portfolio agent (based on previous context)")
        return "portfolio"
    if state.get("market_report_output"):
        print("Routing to market agent (based on previous context)")
        return "market"

    # Greetings or default
    greeting_keywords = ["hello", "hi", "hey", "bjorn", "name is", "greetings"]
    if any(keyword in message for keyword in greeting_keywords):
        print("Routing to conversational agent (greeting)")
        return "chat"

    print("Default routing to conversational agent")
    return "chat"

# --- CREATE LANGGRAPH ---

def create_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route", route_message_node)
    graph.add_node("portfolio", run_portfolio_agent)
    graph.add_node("market", run_market_report_agent)
    graph.add_node("chat", run_conversational_agent)
    graph.add_node("end", lambda state: state)

    graph.add_conditional_edges(
        "route",
        route_decision,
        {
            "portfolio": "portfolio",
            "market": "market",
            "chat": "chat"
        }
    )

    def should_continue(state: GraphState) -> bool:
        return False

    for node in ["portfolio", "market", "chat"]:
        graph.add_conditional_edges(
            node,
            should_continue,
            {False: "end"}
        )

    graph.set_entry_point("route")
    return graph.compile()

# --- ENTRYPOINT FOR API / FRONTEND ---

async def process_user_input(message: str, thread_id: str = "default") -> Dict[str, Any]:
    try:
        # Initialize state
        state: GraphState = {
            "input": message,
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": message}],
            "portfolio_output": None,
            "market_report_output": None,
            "conversational_output": None,
            "chat_output": None,
            "market_charts": None
        }

        # Run the graph
        graph = create_graph()
        result = await graph.ainvoke(state)
        
        # Debug the result
        print("LangGraph result:", json.dumps(result, indent=2))

        # Prioritize output selection
        if result.get("market_report_output"):
            text = result["market_report_output"]
        elif result.get("portfolio_output"):
            text = result["portfolio_output"]
        elif result.get("conversational_output"):
            text = result["conversational_output"]
        else:
            text = "I couldn't process your request. Try rephrasing?"

        # 🔥 Add chart if available
        chart = result.get("market_charts", [None])[0] if result.get("market_charts") else None

        # Create final response
        final_response = {
            "response": text,
            "thread_id": thread_id,
            "vega_lite_spec": chart
        }

        # Debug output
        print(f"Final response structure: {json.dumps({k: v is not None for k, v in final_response.items()}, indent=2)}")
        print(f"Vega spec present: {chart is not None}")

        return final_response
        
    except Exception as e:
        print("Error processing message:\n", traceback.format_exc())
        return {
            "response": f"Sorry, something went wrong: {str(e)}",
            "thread_id": thread_id,
            "vega_lite_spec": None
        }