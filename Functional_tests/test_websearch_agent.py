import asyncio
from Agents.websearch_agent import run_websearch_agent, agent_executor

test_questions = [
    "Latest news on Nvidia's stock performance",
    "Who is the current president of France?",
    "What is the weather in Stockholm today?",
    # "Explain GPT-4 in simple terms",
    # "What happened in the financial markets last week?",
    # "Economic impact of the US inflation rate in 2024"
]

async def run_tests():
    for idx, question in enumerate(test_questions, 1):
        print(f"\n🧪 TEST {idx}: {question}\n")

        # ✅ Reset memory to avoid context bleed
        agent_executor.memory.clear()

        # Create state input for the agent
        result = await run_websearch_agent({
            "input": question,
            "thread_id": f"test-{idx}",
            "messages": [{"role": "user", "content": question}],
            "websearch_output": None,
            "used_fallback": None
        })

        print(f"🔹 Response: {result.get('websearch_output')}")
        print(f"🔁 Used Fallback: {result.get('used_fallback')}")

if __name__ == "__main__":
    asyncio.run(run_tests())
