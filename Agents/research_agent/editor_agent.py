from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

def write_report(sections: list, topic: str) -> str:
    content = "\n\n".join(sections)
    instructions = f"""
You are a technical writer creating a report on this overall topic: {topic}

You have received memos from analysts. Your task is to:
- Summarize key points across all memos
- Format it as a cohesive markdown report
- Start with "## Insights"
- Preserve citations (e.g. [1], [2]) from memos
- End with "## Sources" and a combined list
"""
    response = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=content)
    ])
    return response.content

def write_intro_or_conclusion(sections: list, topic: str, which: str = "introduction") -> str:
    content = "\n\n".join(sections)
    instructions = f"""
You are finishing a research report on: {topic}
Write a crisp {which} in markdown. Target 100 words. Do not include preamble.

Use:
# {topic}
## Introduction  ← or ## Conclusion
"""
    response = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=content)
    ])
    return response.content

def finalize_report(intro: str, summary: str, conclusion: str, sources: str = "") -> str:
    report = f"{intro}\n\n---\n\n{summary}\n\n---\n\n{conclusion}"
    if sources:
        report += f"\n\n## Sources\n{sources.strip()}"
    return report
