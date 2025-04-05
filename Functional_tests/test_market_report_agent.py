# test_market_report_agent.py

import asyncio
from Agents.market_report_agent import run_market_report_agent

def test_market_report_question():
    user_input = "Can you create a weekly market report for the past week for Sweden?"

    # This is the correct format expected by run_market_report_agent
    initial_state = {
        "input": user_input,
        "thread_id": "test-thread",
        "messages": [],
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

    print(f"\n🧪 Running market report test with input: {user_input}\n")

    # Run async agent
    final_state = asyncio.run(run_market_report_agent(initial_state))

    print("\n✅ Agent Response:\n")
    print(final_state.get("market_report_output", "[No output returned]"))
    print("\n🔚 End of test\n")

if __name__ == "__main__":
    test_market_report_question()
