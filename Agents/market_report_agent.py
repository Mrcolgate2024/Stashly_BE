# from portfolio_agent import PortfolioAgent  # was from react_agent import ReactAgent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from agents import Agent, Runner, WebSearchTool
from openai.types.responses import ResponseTextDeltaEvent
from datetime import datetime, timedelta
from typing import TypedDict, Optional, List
import wikipedia
import re
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import altair as alt

class GraphState(TypedDict):
    input: str
    thread_id: str
    messages: List[dict]
    portfolio_output: Optional[str]
    market_report_output: Optional[str]
    conversational_output: Optional[str]
    user_name: Optional[str]
    market_charts: Optional[List[dict]]

DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

# Initialize memory
memory = ConversationBufferMemory(return_messages=True, max_token_limit=10000)

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6,
    request_timeout=120,
    max_retries=5
)

@tool
def search_wikipedia(query: str):
    """Retrieve docs from wikipedia."""
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return "No Wikipedia page found for this query."

        page = wikipedia.page(search_results[0])
        return f"Title: {page.title}\n\nContent: {page.content[:1000]}..."
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

@tool
def generate_chart(data: dict, chart_type: str = "line", title: str = "Chart") -> dict:
    """Generate a Vega-Lite chart spec from data."""
    try:
        df = pd.DataFrame(data)

        if chart_type == "line":
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X(df.columns[0], title=df.columns[0]),
                y=alt.Y(df.columns[1], title=df.columns[1])
            ).properties(title=title)
        elif chart_type == "bar":
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(df.columns[0], title=df.columns[0]),
                y=alt.Y(df.columns[1], title=df.columns[1])
            ).properties(title=title)
        else:
            return {"error": "Unsupported chart type"}

        return chart.to_dict()
    except Exception as e:
        return {"error": str(e)}

@tool
async def search_web(query: str):
    """Retrieve docs from web search."""
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
            instructions=f"""Search for market and financial information about: {search_query}
                           Focus on recent data and include specific numbers and trends.
                           Include sources in the response.""",
            model="gpt-4o-mini"
        )

        result = Runner.run_streamed(search_agent, search_query)

        full_response = []
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response.append(event.data.delta)
                print(event.data.delta, end="", flush=True)

        return "".join(full_response).strip() or f"No relevant results found for: {search_query}"
    except Exception as e:
        return f"Error performing web search: {str(e)}"

prompt_template = PromptTemplate.from_template("""
You are a specialized market analyst from Stashly who creates weekly market reports.

{chat_history}

You have access to the following tools:
{tools}

Tool names: {tool_names}

##CURRENT DATE:
Today's date is {current_date}. All reports cover the past week ({date_range}).

##INSTRUCTIONS (IMPORTANT):
You must follow this exact reasoning format using **ReAct** steps. Do NOT skip steps or go directly to the Final Answer.
You MUST use at least one tool before concluding. Never jump directly to the Final Answer.

### REQUIRED FORMAT:
Question: {input}
Thought: [explain what to do or what to search for]
Action: [tool name, exactly as listed above]
Action Input: [specific search query]
Observation: [summarize result from tool call]
... (repeat Thought / Action / Observation if needed)
Thought: I now know the final answer
Final Answer: [write full market report with sections below]

##REPORT STRUCTURE:
**MARKET SUMMARY ({date_range})**

**MACRO IMPACT & KEY DEVELOPMENTS**

**ANALYSIS & OUTLOOK**
- Be concise and insightful
- Go beyond just summarizing
- Include sources and conclusions

Begin!

Question: {input}
{agent_scratchpad}
""")

tools = [search_web, search_wikipedia, generate_chart]
marketagent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=marketagent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=12,
    max_execution_time=180,
    output_parser="structured",
    agent_kwargs={
        "input_variables": ["input", "chat_history", "agent_scratchpad", "current_date", "date_range"]
    }
)

def get_chat_history() -> str:
    return memory.load_memory_variables({})["history"]

def get_date_range():
    today = datetime.now()
    one_week_ago = today - timedelta(days=7)
    return today.strftime("%B %d, %Y"), one_week_ago.strftime("%B %d, %Y"), f"{one_week_ago.strftime('%b %d')}-{today.strftime('%b %d, %Y')}"

async def process_user_input(user_input: str, thread_id: str = "default") -> str:
    try:
        if user_input.lower().strip() == "initial_greeting":
            response = "Hi, I'm your market analyst from Stashly. Which markets would you like in your weekly report?"
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response)
            return response

        memory.chat_memory.add_user_message(user_input)
        chat_history = get_chat_history()
        today_str, _, date_range = get_date_range()

        if user_input.lower() in ["market report", "weekly report"]:
            output = "Sure! Which country or index should I focus on?"
        else:
            modified_input = f"{user_input} market trends and economic data for the past week ({date_range})"
            result = await agent_executor.ainvoke({
                "input": modified_input,
                "chat_history": chat_history,
                "current_date": today_str,
                "date_range": date_range,
                "thread_id": thread_id
            })
            output = result.get("output", "")

        memory.chat_memory.add_ai_message(output)
        return output
    except Exception as e:
        return f"Error: {str(e)}"

async def run_market_report_agent(state: GraphState) -> GraphState:
    print("Market agent processing:", state["input"])
    try:
        memory.chat_memory.add_user_message(state["input"])
        chat_history = memory.load_memory_variables({})["history"]
        today_str, _, date_range = get_date_range()
        modified_input = f"{state['input']} market trends and economic data for the past week ({date_range})"
        result = await agent_executor.ainvoke({
            "input": modified_input,
            "chat_history": chat_history,
            "current_date": today_str,
            "date_range": date_range,
        })
        output = result.get("output", "No output was returned.")
        memory.chat_memory.add_ai_message(output)
        state["market_report_output"] = output
        return state
    except Exception as e:
        state["market_report_output"] = f"Error from market agent: {str(e)}"
        return state