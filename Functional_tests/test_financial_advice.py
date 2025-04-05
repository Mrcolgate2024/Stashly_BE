import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supervisor_agent import process_user_input

test_questions = [
    # Current market data questions
    "What's the current state of the tech stock market?",
    "What are today's market trends?",
    
    # General financial principles
    "What is a 60/40 portfolio?",
    "How does compound interest work?",
    
    # Historical calculations
    "What were the returns from 2024 to 2025?",
    "How did the market perform in 2024?",
    
    # Specific financial advice
    "How much should I save for retirement?",
    "What's the best way to allocate my portfolio?"
]

async def run_tests():
    for idx, question in enumerate(test_questions, 1):
        print(f"\n🧪 TEST {idx}: {question}\n")
        
        result = await process_user_input(question, thread_id=f"test-{idx}")
        print(f"🔹 Response: {result.get('response')}")
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(run_tests()) 