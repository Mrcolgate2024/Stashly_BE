from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime, timedelta
import re
import wikipedia
import traceback
import json

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

from agents import Agent, Runner, WebSearchTool

# --------------------
# Environment & Config
# --------------------

DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

# --------------------
# Memory & LLM
# --------------------

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit=10000)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6,
    request_timeout=120,
    max_retries=5
)

# --------------------
# Graph State (Optional if standalone)
# --------------------

class GraphState(TypedDict):
    input: str
    thread_id: str
    messages: List[dict]
    portfolio_output: Optional[str]
    market_report_output: Optional[str]
    conversational_output: Optional[str]
    websearch_output: Optional[str]
    stock_output: Optional[str]
    chat_output: Optional[str]
    fund_output: Optional[str]
    market_charts: Optional[List[dict]]
    user_name: Optional[str]
    last_ran_agent: Optional[str]


# --------------------
# Web + Wikipedia Tools
# --------------------

async def search_web(query: str) -> str:
    try:
        today = datetime.now()
        one_week_ago = today - timedelta(days=7)
        date_range = f"{one_week_ago.strftime('%b %d')}-{today.strftime('%b %d, %Y')}"

        if not any(term in query.lower() for term in ["last week", "past week", "recent"]):
            if not re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', query) and not re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', query.lower()):
                search_query = f"{query} from {date_range}"
            else:
                search_query = query
        else:
            search_query = query

        search_agent = Agent(
            name="websearch-agent",
            tools=[WebSearchTool()],
            instructions=f"Search for knowledge about: {search_query}.",
            model="gpt-4o-mini"
        )

        result = Runner.run_streamed(search_agent, search_query)
        full_response = []
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                full_response.append(event.data.delta)

        return "".join(full_response).strip() or f"No relevant results found for: {search_query}"
    except Exception as e:
        return f"Error performing web search: {str(e)}"


def search_wikipedia(query: str) -> str:
    try:
        results = wikipedia.search(query)
        if not results:
            return "No Wikipedia page found for this query."
        page = wikipedia.page(results[0])
        return f"Title: {page.title}\n\nContent: {page.content[:1000]}..."
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


# --------------------
# LangChain Tool Wrappers
# --------------------

@tool
async def search_web_tool(input: str) -> str:
    """Searches the web for recent information using the web search tool."""
    return await search_web(input)

@tool
def search_wikipedia_tool(input: str) -> str:
    """Searches Wikipedia and returns the first 1000 characters of the top article."""
    return search_wikipedia(input)



# --------------------
# Agent Setup
# --------------------

prompt = PromptTemplate.from_template("""You are a helpful research assistant who answers user questions using the tools below:

{tool_names}

{tools}

You must always think step-by-step and use this format:

Thought: Explain your reasoning.
Action: tool_name
Action Input: the input to that tool

When you have the final answer, write:
Final Answer: your complete response to the user.

Here are a few examples:

Example 1:
Thought: The user is asking about retirement planning. I should search for general guidelines while including appropriate disclaimers.
Action: search_web_tool
Action Input: expert guidelines retirement savings calculation methodology

Example 2:
Thought: I need a reliable summary, so I’ll check Wikipedia.
Action: search_wikipedia_tool
Action Input: history of the Roman Empire

Example 3:
Thought: The user is asking about a current event, so web search is appropriate.
Action: search_web_tool
Action Input: earthquake in Japan March 2025

Example 4:
Thought: For a general explanation, Wikipedia is likely sufficient.
Action: search_wikipedia_tool
Action Input: what is quantum computing

---

Previous conversation:
{chat_history}

User question: {input}

{agent_scratchpad}
""")

tools: List = [search_web_tool, search_wikipedia_tool]

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,  # ✅ Add this line
    max_iterations=10,
    max_execution_time=180
)


# --------------------
# Entrypoint for LangGraph
# --------------------

async def run_websearch_agent(state: dict) -> dict:
    last_message = state["messages"][-1]["content"]

    try:
        # Step 1: Use agent to try answering
        response = await agent_executor.ainvoke({"input": last_message})
        output = response.get("output", "")

        # Step 2: Manual fallback if unsure
        fallback_needed = any(phrase in output.lower() for phrase in [
            "i'm not sure", "i don't know", "unable to find", "no relevant results"
        ]) or not output.strip()

        if fallback_needed:
            wiki_result = search_wikipedia(last_message)
            if "No Wikipedia page found" not in wiki_result:
                output = wiki_result
            else:
                output = await search_web(last_message)

    except Exception as e:
        output = f"An error occurred while trying to answer the question: {str(e)}"
        fallback_needed = True

    return {
        **state,
        "websearch_output": output,
        "used_fallback": fallback_needed,
        "last_ran_agent": "websearch",
        "messages": state["messages"] + [{"role": "assistant", "content": output}]
    }



# --------------------
# Optional: Local test entrypoint
# --------------------

async def process_user_input(message: str, thread_id: str = "default") -> Dict[str, Any]:
    try:
        state: GraphState = {
            "input": message,
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": message}],
            "websearch_output": None,
            "used_fallback": None
        }

        result = await run_websearch_agent(state)
        print("WebSearch Agent Output:", json.dumps(result, indent=2))

        return {
            "response": result.get("websearch_output"),
            "thread_id": thread_id
        }

    except Exception as e:
        print("Error:", traceback.format_exc())
        return {
            "response": f"Error: {str(e)}",
            "thread_id": thread_id
        }


# --------------------
# Exported symbols
# --------------------

__all__ = ['run_websearch_agent', 'agent_executor', 'memory']
