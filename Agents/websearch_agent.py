from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from typing import List
from datetime import datetime, timedelta
import re
import wikipedia
from agents import Agent, Runner, WebSearchTool


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
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
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

from langchain.tools import tool

@tool
async def search_web_tool(query: str) -> str:
    return await search_web(query)

@tool
def search_wikipedia_tool(query: str) -> str:
    return search_wikipedia(query)


# --------------------
# Agent Setup
# --------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = PromptTemplate.from_template("""You are a helpful research assistant who answers user questions using the tools below:
{tools}

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
    verbose=True
)


# --------------------
# Entrypoint for LangGraph
# --------------------

async def run_websearch_agent(state) -> dict:
    last_message = state["messages"][-1]["content"]
    response = await agent_executor.ainvoke({"input": last_message})
    output = response.get("output", "")

    return {
        **state,
        "websearch_output": output,
        "messages": state["messages"] + [{"role": "assistant", "content": output}]
    }
