import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supervisor_agent import process_user_input

TEST_CASES = [
    # "Imagine you are a financial advisor, what advice would you give me?",
    # "I am 60, spending 100 000 kr per month, how much savings do I need to retire at 65?",
    # "So how much in savings?",
    # "Can you calculate the return for Portfolio 60/40 between 2020-01-01 and 2024-12-31?",
    # "Can you create a chart for the values for OMX30 for the monthly values for the year 2024?",
    # "Do I have exposure to Microsoft in my fund Länsförsäkringar Sverige Index?",
    # "What is the allocation to Microsoft in my fund Länsförsäkringar Global Index?",
    # "What’s the latest price for Tesla?",
    # "What about Microsoft?"
    # "Top Holdings for 'Länsförsäkringar Global Index'?",
    # "Can you create a weekly market report for the past week for Sweden",
    # "Can you calculate the return for OMX30 between 2020-01-01 and 2023-12-31?",
    # "Check the stock price for Tesla.",
    # "Hello there!",
    # "Search the web for inflation data in Sweden.",
    # "Can you create a chart for the values for omx30 for the monthly values for the year 2024?"
    
    # 🔹 Conversational Agent
    "Hi",
    "Hello there",
    # "Can you tell me about yourself?",

    # 🔹 Market Summary Agent
    # "Can you create a weekly market report for the past week for Sweden?",
    # "Can you give me a weekly market report on USA and Donald Trump?",

    # 🔹 Web Search Agent
    # "What is diversification?",
    # "Explain compound interest in simple terms.",
    # "I would like to retire in 5 years, can you help me?",
    # "Imagine you are a financial advisor, what advice would you give me?",
    # "I am 60, spending 100 000 kr per month, how much savings do I need to retire at 65?",


    # 🔹 Portfolio Agent
    # "Can you calculate the return for S&P500 for 2023-03-31 to 2024-03-31?",
    "Can you calculate the return for Portfolio 60/40 between 2020-01-01 and 2024-12-31?",
    # "Can you create a chart for the values for OMX30 for the monthly values for the year 2024?",

    # 🔹 Fund Position Agent
    # "Do I have exposure to Microsoft in my fund Länsförsäkringar Sverige Index?",
    # "What are the top holdings in Swedbank Robur Access Global?",
    # "Can you give me a list of the holdings for 'Länsförsäkringar Global Index'?",
    # "What funds hold Tesla?",
    # "What is the risk level of Länsförsäkringar USA Index?",

    # # 🔹 Stock Price Agent
    # "What’s the latest price for Tesla?",


]



async def run_tests():
    for message in TEST_CASES:
        print(f"\n💬 User Input: {message}")
        result = await process_user_input(message, thread_id="test-thread")
        print("🧠 Agent Response:")
        print(result["response"])
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(run_tests())