import asyncio
from Agents.stock_price_agent import run_stock_price_agent

async def test_stock_agent():
    message = "What’s the stock price for Swedbank?"
    state = {
        "input": message,
        "thread_id": "test-stock",
        "messages": [{"role": "user", "content": message}]
    }

    result = await run_stock_price_agent(state)
    print("\n💹 Stock Price Agent Output:\n")
    print(result.get("stock_output", "No result"))

if __name__ == "__main__":
    asyncio.run(test_stock_agent())
