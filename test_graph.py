import asyncio
from supervisor_agent import process_user_input

async def test_graph_with_inputs():
    test_cases = [
        "Can you show me the market report for last week?",
        # "What's the annual return on Equity Sweden from 2020 to 2024?",
        # "Tell me about my portfolio allocation",
        # "Give me a performance table for S&P 500 over 2021",
        # "initial_greeting"
    ]

    print("=== TESTING GRAPH ROUTING + AGENTS ===\n")
    
    for i, message in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"User Input: {message}\n")
        
        result = await process_user_input(message, thread_id=f"test-thread-{i}")
        print(f"Agent Response:\n{result}\n")

if __name__ == "__main__":
    asyncio.run(test_graph_with_inputs())