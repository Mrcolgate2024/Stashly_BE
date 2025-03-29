import asyncio
from Agents.market_report_agent import run_market_report_agent, GraphState

# async def run():
#     print("\n=== Testing Market Report with Chart ===")
#     state: GraphState = {
#         "input": "Sweden market performance",
#         "thread_id": "test-thread-graph",
#         "messages": [],
#         "portfolio_output": None,
#         "market_report_output": None,
#         "conversational_output": None,
#         "user_name": "Tester",
#         "market_charts": []
#     }
#     state = await run_market_report_agent(state)
#     print("\n\n--- Final Market Report Output ---")
#     print(state.get("market_report_output"))
#     if state.get("market_charts"):
#         print("\n--- Charts Generated ---")
#         for chart in state["market_charts"]:
#             print(chart)


import asyncio
from Agents.portfolio_agent import run_portfolio_agent, GraphState

async def run():
    state: GraphState = {
        "input": "Show me a chart of the OMX30 from 2022-01-01 to 2023-01-01",
        "thread_id": "test-portfolio-chart",
        "messages": [],
        "portfolio_output": None,
        "market_report_output": None,
        "conversational_output": None,
        "user_name": None,
        "market_charts": []
    }

    final_state = await run_portfolio_agent(state)
    print("Portfolio Output:", final_state.get("portfolio_output"))
    if final_state.get("market_charts"):
        print("Vega-Lite Specs Found:")
        for chart in final_state["market_charts"]:
            print(chart)



if __name__ == "__main__":
    asyncio.run(run())
