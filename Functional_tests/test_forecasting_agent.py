# test_forecasting_agent.py

import sys
import os
import asyncio
from pathlib import Path
from Agents.forecasting_agent_final import run_forecasting_agent, GraphState
from dotenv import load_dotenv

test_questions = [
    "What will the MSCI World index be in 1 month?",
    "Can you show me a chart of the forecast for all horizons?",
    "Explain what drives the 1Y forecast.",
    "What market regime are we in?",
    "How does today compare to 2008?"
]

async def run_tests():
    for idx, question in enumerate(test_questions, 1):
        print(f"\n🧪 TEST {idx}: {question}\n")
        
        result = await run_forecasting_agent({
            "input": question,
            "thread_id": f"forecast-{idx}",
            "messages": [{"role": "user", "content": question}],
            "forecast_output": None,
            "used_fallback": None,
            "chat_history": []
        })
        
        print(f"🔹 Response: {result.get('forecast_output')}")
        print(f"🔁 Used Fallback: {result.get('used_fallback')}")

if __name__ == "__main__":
    asyncio.run(run_tests())