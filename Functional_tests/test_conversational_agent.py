import asyncio
from Agents.conversational_agent import run_conversational_agent, agent_executor

test_questions = [
    "Hi",
    "Initial greeting",
    "Who are you?",
    "What are you wearing?",
    "How are you?",
    "Where are you from?",
    "Tell me about your family",
    "My name is John",
    "What is my name?",
    "Tell me about your education",
    "What's your background?",
    "How old are you?"
]

async def run_tests():
    for idx, question in enumerate(test_questions, 1):
        print(f"\n🧪 TEST {idx}: {question}\n")

        # ✅ Reset memory to avoid context bleed
        agent_executor.memory.clear()

        # Create state input for the agent
        result = await run_conversational_agent({
            "input": question,
            "thread_id": f"test-{idx}",
            "messages": [{"role": "user", "content": question}],
            "portfolio_output": None,
            "market_report_output": None,
            "conversational_output": None,
            "websearch_output": None,
            "chat_output": None,
            "market_charts": None,
            "user_name": None,
            "last_ran_agent": None
        })

        print(f"🔹 Response: {result.get('conversational_output')}")
        print(f"👤 User Name: {result.get('user_name')}")

if __name__ == "__main__":
    asyncio.run(run_tests())
