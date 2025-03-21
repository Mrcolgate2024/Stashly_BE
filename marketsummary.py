from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
# from langchain_community.tools import TavilySearchResults
# from langchain_google_community import GoogleSearchAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WikipediaLoader
from langchain.tools import tool
import os
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize memory
memory = ConversationBufferMemory()

# Initialize context tracker
context_tracker = {
    "last_index_discussed": None,
    "last_year_discussed": None
}

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    request_timeout=120,
    max_retries=5
)

# Setup API-Tools
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY)

# Define SearchQuery class for structured output
class SearchQuery:
    def __init__(self, search_query: str):
        self.search_query = search_query

# Define search instructions
search_instructions = "Based on the conversation, create a search query that will help answer the user's question."

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================


@tool
def search_web(query: str):
    """ Retrieve docs from web search """
    try:
        # Add date specificity if not already present
        today = datetime.now()
        one_week_ago = today - timedelta(days=7)
        date_range = f"{one_week_ago.strftime('%b %d')}-{today.strftime('%b %d, %Y')}"
        
        # If the query doesn't already have a date range, add it
        if not any(term in query.lower() for term in ["last week", "past week", "recent", today.strftime("%B").lower(), today.strftime("%b").lower()]):
            # Don't add date range if it's already a date-specific query
            if not re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', query) and not re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', query.lower()):
                query = f"{query} {date_range}"
        
        # Use the tavily tool directly with the provided query
        search_docs = tavily_tool.invoke(query)
        
        # Format results
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ]
        )
        
        return formatted_search_docs
    except Exception as e:
        return f"Error performing web search: {str(e)}"

@tool
def search_wikipedia(query: str):
    """ Retrieve docs from wikipedia """
    try:
        # Search Wikipedia directly with the provided query
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        
        # Format results
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )
        
        return formatted_search_docs
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

# ============================================================================
# AGENT SETUP
# ============================================================================


# Define the prompt template
prompt_template = PromptTemplate.from_template("""
You are a specialized market analyst from Stashly who creates weekly market reports.

{chat_history}

You have access to the following tools:
{tools}

Tool names: {tool_names} and use llm to generate the response

##CURRENT DATE:
Today's date is {current_date}. All reports cover the past week ({date_range}).

##PERSONA:
You are a professional, insightful market analyst with deep financial expertise.
You don't just report data - you analyze patterns, identify underlying trends, and form independent conclusions.

##REPORT PROCESS:
1. GATHER DATA: Use search_web to find recent market data, macroeconomic news, and policy decisions.
   - Search for "[markets] performance {date_range}"
   - Search for "[markets] economic news {date_range}"
   - Search for "central bank [markets] {date_range}"

2. EVALUATE & ANALYZE: 
   - Connect economic data points to specific market reactions
   - Identify cause-and-effect relationships between news and price movements
   - Look for patterns that might not be explicitly stated in the sources
   - Apply your financial expertise to interpret what the data suggests

3. FORM INDEPENDENT CONCLUSIONS:
   - Based on the data gathered, form your own reasoned outlook
   - Consider historical patterns and economic principles
   - Identify potential risks and opportunities not explicitly mentioned in sources
   - Draw connections between different market segments and economic indicators

4. PRODUCE REPORT: Create a concise report following this format:

**MARKET SUMMARY ({date_range})**: 
- Brief overview of market performance
- Major indices with % changes
- Direct connection to key economic drivers

**MACRO IMPACT & KEY DEVELOPMENTS**:
- How economic data influenced markets
- Interest rate/central bank impacts
- Major news events and their market effects
- Sector-specific reactions to economic conditions

**ANALYSIS & OUTLOOK**:
- Your independent assessment of market conditions
- Reasoned predictions based on current trends and historical patterns
- Potential risks and opportunities you've identified
- Sectors or assets that may outperform or underperform
- List of sources with links

IMPORTANT: In the Analysis & Outlook section, go beyond what's explicitly stated in sources. Apply your expertise to draw original conclusions and provide a forward-looking perspective. This is where your analytical value is most important.

##FORMATTING REQUIREMENTS:
Question: {input}
Thought: [your reasoning]
Action: [tool name]
Action Input: [search query]
Observation: [result]
... (repeat as needed)
Thought: I now know the final answer
Final Answer: [your concise weekly market report with emphasis on macro impacts and your independent analysis]

Begin!

Question: {input}
{agent_scratchpad}
""")

# Define the tools
tools = [
    search_web,
    search_wikipedia
]

# Create the agent
marketagent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

# Get current date for default agent execution
today = datetime.now()
today_str = today.strftime("%B %d, %Y")
one_week_ago = today - timedelta(days=7)
date_range = f"{one_week_ago.strftime('%b %d')}-{today.strftime('%b %d, %Y')}"

# Update the agent executor with more robust error handling
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=marketagent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    max_execution_time=60,
    agent_kwargs={
        "input_variables": ["input", "chat_history", "agent_scratchpad", "current_date", "date_range"]
    }
)

# ============================================================================
# CHAT FUNCTIONALITY
# ============================================================================

def get_chat_history() -> str:
    """Get formatted chat history for the prompt."""
    # Get the messages from memory
    messages = memory.load_memory_variables({})["history"]
    return messages

def get_date_range():
    """Get the date range for the past week."""
    today = datetime.now()
    one_week_ago = today - timedelta(days=7)
    
    # Format dates as "Month Day, Year"
    today_str = today.strftime("%B %d, %Y")
    one_week_ago_str = one_week_ago.strftime("%B %d, %Y")
    
    # Create a shorter format for search queries: "Apr 22-29, 2024"
    short_today = today.strftime("%b %d, %Y")
    short_one_week_ago = one_week_ago.strftime("%b %d")
    short_range = f"{short_one_week_ago}-{short_today.split(' ')[1]}, {short_today.split(' ')[2]}"
    
    return today_str, one_week_ago_str, short_range

def process_user_input(user_input: str) -> str:
    """Process user input and return the agent's response."""
    global context_tracker, memory
    
    try:
        # Check for initial greeting
        if user_input.lower().strip() == "initial_greeting":
            response = "Hi, I'm your market analyst from Stashly. Which markets would you like in your weekly report?"
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response)
            return response
            
        # Add user input to memory
        memory.chat_memory.add_user_message(user_input)
        
        # Get chat history
        chat_history = get_chat_history()
        
        # Get current date and date range
        today_str, one_week_ago_str, date_range = get_date_range()
        
        # Execute the agent
        try:
            # If the user input is very general, provide a more specific prompt
            if user_input.lower() in ["market report", "market report for the last week", "market update", "weekly report"]:
                # Add a more specific response for general market report requests
                output = f"I'd be happy to prepare your weekly market report! Which markets are you interested in (e.g., US, European, Swedish)?"
            else:
                # For specific market queries, directly search for the data
                if any(term in user_input.lower() for term in ["market", "stock", "index", "economy"]) and any(market in user_input.lower() for market in ["us", "usa", "united states", "sweden", "uk", "europe", "china", "japan", "global"]):
                    # This is likely a specific market query, let's use the agent
                    # Modify the user input to explicitly mention the date range if it doesn't already
                    if "last week" not in user_input.lower() and "past week" not in user_input.lower():
                        modified_input = f"{user_input} for the past week ({date_range})"
                    else:
                        modified_input = user_input
                    
                    result = agent_executor.invoke({
                        "input": modified_input,
                        "chat_history": chat_history,
                        "current_date": today_str,
                        "date_range": date_range
                    })
                    output = result["output"]
                else:
                    # For other queries, use the agent
                    result = agent_executor.invoke({
                        "input": user_input,
                        "chat_history": chat_history,
                        "current_date": today_str,
                        "date_range": date_range
                    })
                    output = result["output"]
                
                # Remove duplicated tables
                # First, identify all markdown tables in the output
                table_pattern = r'(\|\s*[\w\s]+\s*\|.*?\n\|[-\s|]+\|\n(?:\|.*?\|\n)+)'
                tables = re.findall(table_pattern, output, re.DOTALL)
                
                # If we found multiple tables, check for duplicates
                if len(tables) > 1:
                    unique_tables = []
                    for table in tables:
                        # Normalize the table by removing extra whitespace
                        normalized = re.sub(r'\s+', ' ', table).strip()
                        if normalized not in [re.sub(r'\s+', ' ', t).strip() for t in unique_tables]:
                            unique_tables.append(table)
                    
                    # If we found duplicates, reconstruct the output with only unique tables
                    if len(unique_tables) < len(tables):
                        # Split the output by tables
                        parts = re.split(table_pattern, output, flags=re.DOTALL)
                        
                        # Reconstruct the output with only unique tables
                        new_output = parts[0]  # Start with the text before the first table
                        for i in range(len(unique_tables)):
                            new_output += unique_tables[i]
                            if (i+1)*2 < len(parts):
                                new_output += parts[(i+1)*2]  # Add the text between tables
                        
                        output = new_output
            
        except Exception as e:
            error_str = str(e)
            
            # Check if the error is a formatting error
            if "Invalid Format" in error_str or "Could not parse LLM output" in error_str:
                # For general market report requests, provide a more specific response
                if user_input.lower() in ["market report", "market report for the last week", "market update"]:
                    output = f"I'd be happy to prepare your weekly market report! Which markets are you interested in (e.g., US, European, Swedish)?"
                # For specific market queries, try to handle directly
                elif any(term in user_input.lower() for term in ["market", "stock", "index", "economy"]) and any(market in user_input.lower() for market in ["us", "usa", "united states", "sweden", "uk", "europe", "china", "japan", "global"]):
                    # Extract the market
                    market_match = re.search(r'(us|usa|united states|sweden|uk|europe|china|japan|global)', user_input.lower())
                    market = market_match.group(1) if market_match else "global"
                    
                    # Perform a direct web search with the current date range
                    try:
                        # Search for both market data and economic news
                        market_query = f"{market} market summary {date_range}"
                        economic_query = f"{market} economic news impact markets {date_range}"
                        
                        market_results = search_web(market_query)
                        economic_results = search_web(economic_query)
                        
                        combined_results = market_results + "\n\n" + economic_results
                        
                        # Extract a simple summary from the search results
                        output = f"**MARKET SUMMARY ({date_range})**: \n\n"
                        
                        # Extract market performance
                        if "dropped" in combined_results.lower() or "down" in combined_results.lower() or "declined" in combined_results.lower():
                            output += f"• {market.title()} markets declined this week. "
                        elif "increased" in combined_results.lower() or "up" in combined_results.lower() or "gained" in combined_results.lower():
                            output += f"• {market.title()} markets gained this week. "
                        else:
                            output += f"• {market.title()} markets showed mixed performance. "
                        
                        # Add some specific data if available
                        percentage_match = re.search(r'(\d+\.\d+)%', combined_results)
                        if percentage_match:
                            output += f"Notable movement of {percentage_match.group(1)}%.\n\n"
                        
                        # Add macroeconomic impact section
                        output += "**MACRO IMPACT & KEY DEVELOPMENTS**:\n\n"
                        
                        # Look for economic terms and their impacts
                        econ_terms = ["inflation", "interest rate", "fed", "central bank", "gdp", "employment", "unemployment", 
                                     "consumer", "manufacturing", "policy", "recession"]
                        
                        econ_impacts = []
                        for term in econ_terms:
                            # Find sentences containing the economic term
                            pattern = r'[^.!?]*\b' + term + r'\b[^.!?]*[.!?]'
                            matches = re.findall(pattern, combined_results, re.IGNORECASE)
                            if matches:
                                # Take the first match
                                econ_impacts.append(f"• {matches[0].strip()}")
                        
                        if econ_impacts:
                            output += "\n".join(econ_impacts[:3]) + "\n\n"  # Limit to 3 impacts
                        else:
                            output += "• Economic data for this period is limited in the search results.\n\n"
                        
                        # Add analytical outlook section based on the data gathered
                        output += "**ANALYSIS & OUTLOOK**:\n\n"
                        
                        # Generate analytical insights based on the data
                        if "inflation" in combined_results.lower():
                            if "higher" in combined_results.lower() or "increased" in combined_results.lower() or "rising" in combined_results.lower():
                                output += "• Rising inflation suggests central banks may maintain restrictive monetary policy, potentially creating headwinds for growth stocks and benefiting value sectors.\n"
                            elif "lower" in combined_results.lower() or "decreased" in combined_results.lower() or "falling" in combined_results.lower():
                                output += "• Declining inflation could provide central banks with flexibility to consider rate cuts, potentially benefiting both equity and bond markets in the coming months.\n"
                        
                        if "interest rate" in combined_results.lower() or "fed" in combined_results.lower() or "central bank" in combined_results.lower():
                            if "cut" in combined_results.lower() or "lower" in combined_results.lower() or "dovish" in combined_results.lower():
                                output += "• The dovish central bank stance could support market valuations, particularly for growth stocks and rate-sensitive sectors like real estate and utilities.\n"
                            elif "hike" in combined_results.lower() or "raise" in combined_results.lower() or "hawkish" in combined_results.lower():
                                output += "• Continued hawkish monetary policy may create volatility in both equity and fixed income markets, favoring defensive sectors and quality companies with strong balance sheets.\n"
                        
                        if "growth" in combined_results.lower() or "gdp" in combined_results.lower():
                            if "strong" in combined_results.lower() or "robust" in combined_results.lower() or "better" in combined_results.lower():
                                output += "• Signs of economic resilience suggest corporate earnings may hold up better than expected, potentially supporting equity valuations despite other headwinds.\n"
                            elif "weak" in combined_results.lower() or "slowing" in combined_results.lower() or "recession" in combined_results.lower():
                                output += "• Economic slowdown signals warrant caution, as corporate earnings may face pressure in coming quarters, suggesting a defensive positioning may be prudent.\n"
                        
                        # Add a forward-looking conclusion
                        output += "\nBased on current market conditions and economic indicators, investors should monitor upcoming economic data releases closely, as they will likely drive market sentiment in the near term. The interplay between inflation, central bank policy, and growth will remain central to market performance.\n\n"
                        
                        output += "Would you like a more detailed analysis of any specific aspect of the market outlook?"
                    except Exception:
                        # If direct search fails, provide a generic response
                        output = f"I'd like to provide you with a market summary for {market.title()} with macroeconomic impacts, but I'm having trouble gathering the data. Could you please try again with a more specific request?"
                else:
                    # For other queries, provide a generic response
                    output = "I'm here to provide weekly market reports. Which markets would you like information on (e.g., US, European, Swedish)?"
            else:
                # For other errors, provide the error message
                output = f"I apologize, but I encountered an error while preparing your report: {str(e)}. Please try again with a simpler query."
        
        # Update context tracker based on the user input and agent response
        # Check for mentions of indices
        index_pattern = r'\b(equity\s+\w+|s&p\s+500|nasdaq|ftse\s+100)\b'
        index_matches = re.findall(index_pattern, (user_input + " " + output).lower())
        if index_matches:
            context_tracker["last_index_discussed"] = index_matches[-1].title()
        
        # Check for mentions of years
        year_pattern = r'\b(20\d{2})\b'
        year_matches = re.findall(year_pattern, user_input + " " + output)
        if year_matches:
            context_tracker["last_year_discussed"] = year_matches[-1]
        
        # Add the agent's response to the history
        memory.chat_memory.add_ai_message(output)
        
        return output
    
    except Exception as e:
        # Catch any unexpected errors in the process_user_input function itself
        error_message = f"I apologize, but I encountered an unexpected error: {str(e)}. Please try again with a different query."
        memory.chat_memory.add_ai_message(error_message)
        return error_message

def chat():
    """Main chat loop function."""
    print("\nHi, I'm your market analyst from Stashly and I'm here to help you with a weekly market report. Which markets and focus areas are you interested in? Type 'quit' to exit.\n")
    
    while True:
        # Get user input
        user_input = input("\nUser: ")
        
        # Check for quit command
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nThanks for using Stashly's weekly market report service. Have a great day!")
            break
        
        try:
            # Process user input through the agent with history
            result = process_user_input(user_input)
            print("\nAI:", result)
        except Exception as e:
            print(f"\nAI: I'm sorry, I encountered an error while preparing your weekly market report: {str(e)}. Could you please try again with a more specific request?")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    chat()
