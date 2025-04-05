import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.portfolio_agent import run_portfolio_agent

TEST_QUESTIONS = [
    # "Can you create a chart for the values for OMX30 for the monthly values for the year 2024?",
    "Can you calculate the return for '60/40 Portfolio' between 2020-01-01 and 2024-12-31?",
    # "Can you calculate the performance of OMX30 between 2020-01-01 and 2023-12-31?",
    # "Give me a table of OMX30 and MSCI World from 2021-01-01 to 2022-12-31, monthly",
    # "What is the Sharpe ratio for OMX30 for 2023?"

]

async def run_all_tests():
    for i, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n--- Test {i}: {question} ---")

        state: GraphState = {
            "input": question,
            "thread_id": f"test-thread-{i}",
            "messages": [{"role": "user", "content": question}],
            "portfolio_output": None,
            "market_report_output": None,
            "conversational_output": None,
            "websearch_output": None,
            "stock_output": None,
            "chat_output": None,
            "fund_output": None,
            "market_charts": None,
            "user_name": None,
            "last_ran_agent": None,
        }

        result = await run_portfolio_agent(state)
        print("✅ Portfolio Agent Response:")
        print(result["portfolio_output"])

if __name__ == "__main__":
    asyncio.run(run_all_tests())
