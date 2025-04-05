import asyncio
import sys
import os
import traceback

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supervisor_agent import process_user_input
import json

async def test_fund_agent():
    test_cases = [
        # Test single fund exposure
        "What is the allocation to Tesla in my fund Länsförsäkringar Global Index?",
        
        # # Test ownership-based exposure in single fund
        "I own 2% in Länsförsäkringar Global Index, what is my exposure to Tesla?",
        
        # # Test multiple fund exposure
        "What is my exposure to Tesla if I own 50% in Länsförsäkringar Sverige Index, and 50% in Länsförsäkringar USA Index?",
        
        # # Old test cases (commented out)
        # "What is the largest position across all funds?",
        # "What is the sector exposure of my funds?",
        # "What are the top 5 holdings in Länsförsäkringar Global Index?",
        # "How much exposure do I have to the technology sector?",
        # "What is the total value of my fund holdings?",
        # "Which funds have the highest exposure to Microsoft?",
        # "What is the geographic distribution of my fund holdings?",
        # "How has my fund allocation changed over the last quarter?",
        # "What is the risk level of my current fund portfolio?",
        # "Which funds have the highest concentration in a single stock?"
        "What is the risk level of Länsförsäkringar USA Index?"
        # "What is the risk level of XACT Nordic High Dividend Low Volatility ETF?",
        # "What is the risk level of my portfolio consisting of XACT Nordic High Dividend Low Volatility ETF, XACT Nordic High Dividend ETF, and XACT Nordic High Dividend ESG ETF?",
    #    "Which of my funds have the highest risk? Compare XACT Nordic High Dividend Low Volatility ETF and XACT Nordic High Dividend ETF",
    # 
    # 
    ]
    
    print("\n🧪 Starting Fund Position Agent Tests\n")
    
    for idx, question in enumerate(test_cases, 1):
        print(f"\n🧪 TEST {idx}: {question}\n")
        
        try:
            result = await process_user_input(question, thread_id=f"test-{idx}")
            print(f"🔹 Response: {result.get('response')}")
            print(f"🔹 Last Ran Agent: {result.get('state', {}).get('last_ran_agent')}")
            print(f"🔹 Fund Output Present: {'fund_output' in result.get('state', {})}")
            
            # Print intermediate steps if available
            if 'state' in result and 'intermediate_steps' in result['state']:
                print("\n🔹 Intermediate Steps:")
                for step in result['state']['intermediate_steps']:
                    print(f"Action: {step[0]}")
                    print(f"Observation: {step[1]}\n")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print(f"❌ Traceback: {traceback.format_exc()}")
        
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(test_fund_agent())