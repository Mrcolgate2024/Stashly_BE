from typing import Annotated
import operator
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI
from agents.research_agent.analysts import Analyst
from config.settings import Settings

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

class InterviewState(TypedDict):
    analyst: Analyst
    messages: list
    max_num_turns: int
    context: Annotated[list, operator.add]
    interview: str
    sections: list

def generate_question(state: InterviewState):
    analyst = state["analyst"]
    system_prompt = f"""
You are an AI analyst interviewing an expert on a topic.
Your role is: {analyst.role}
Stay in character. Start with an intro and ask a smart question.
Keep digging deeper until satisfied.
End with "Thank you so much for your help!"
"""
    question = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    return {"messages": [question]}

def search_context(state: InterviewState, settings: Settings):
    query = state["messages"][-1].content
    
    # Initialize OpenAI client for web search
    client = OpenAI(api_key=settings.openai_api_key)
    
    # Create the chat completion with web search
    completion = client.chat.completions.create(
        model=settings.web_search_model,
        web_search_options=settings.web_search_options,
        messages=[{
            "role": "user",
            "content": query
        }]
    )
    
    # Extract the response and citations
    openai_response = completion.choices[0].message.content
    citations = completion.choices[0].message.annotations if hasattr(completion.choices[0].message, 'annotations') else []
    
    # Get additional context from Tavily and Wikipedia
    tavily = TavilySearchResults(max_results=2)
    wiki_docs = WikipediaLoader(query=query, load_max_docs=1).load()
    tav_docs = tavily.invoke(query)

    # Combine all sources
    all_docs = ""
    for doc in tav_docs:
        all_docs += doc["content"] + "\n"
    for doc in wiki_docs:
        all_docs += doc.page_content + "\n"
    all_docs += f"\nOpenAI Web Search:\n{openai_response}"
    
    # Add citations if available
    if citations:
        all_docs += "\n\nSources:"
        for citation in citations:
            if citation["type"] == "url_citation":
                url_citation = citation["url_citation"]
                all_docs += f"\n- {url_citation['title']}: {url_citation['url']}"
    
    return {"context": [all_docs]}

def generate_answer(state: InterviewState):
    analyst = state["analyst"]
    context = "\n\n".join(state["context"])
    prompt = f"""
You are an expert answering this AI analyst:

{analyst.persona}

Using this context only:
{context}

Answer the last question clearly. Cite sources if applicable.
"""
    answer = llm.invoke([
        SystemMessage(content=prompt),
        *state["messages"]
    ])
    answer.name = "expert"
    return {"messages": [answer]}

def save_interview(state: InterviewState):
    full = get_buffer_string(state["messages"])
    return {"interview": full}

def write_section(state: InterviewState):
    analyst = state["analyst"]
    context = "\n\n".join(state["context"])
    prompt = f"""
You are a technical writer. Create a well-structured markdown section from this interview.

Focus area:
{analyst.description}

Content:
{context}
"""
    section = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Write the section.")
    ])
    return {"sections": [section.content]}

def route_messages(state: InterviewState):
    expert_turns = [m for m in state["messages"] if isinstance(m, AIMessage) and getattr(m, "name", "") == "expert"]
    if len(expert_turns) >= state["max_num_turns"]:
        return "save_interview"
    if "thank you" in state["messages"][-1].content.lower():
        return "save_interview"
    return "generate_question"
