import re
from langchain_core.messages import AIMessage, HumanMessage
from agents.research_agent.analysts import create_analysts
from agents.research_agent.interview import (
    generate_question,
    search_context,
    generate_answer,
    route_messages,
    save_interview,
    write_section
)
from agents.research_agent.report_writer import (
    write_intro_or_conclusion,
    write_report,
    finalize_report
)
from config.settings import Settings

def extract_topic_and_count(message: str) -> tuple[str, int]:
    """
    Extract the topic and number of analysts from the message.
    Returns (topic, count) or (None, None) if not found.
    """
    # Try to find the number of analysts
    count_match = re.search(r'(\d+)\s*analysts?', message.lower())
    count = int(count_match.group(1)) if count_match else None
    
    # Try to find the topic
    topic_match = re.search(r'research\s+(.+?)(?:\s+with\s+\d+\s+analysts?)?$', message.lower())
    topic = topic_match.group(1).strip() if topic_match else None
    
    return topic, count

def generate_research_report(topic: str, num_analysts: int, context: str) -> str:
    """
    Generate a research report based on the topic, number of analysts, and context.
    """
    report = f"Research Report: {topic}\n\n"
    report += f"Analysis conducted by {num_analysts} analysts\n\n"
    report += "Key Findings:\n"
    report += context
    
    return report

def run_research_agent(state, settings: Settings) -> str:
    """
    Run the research agent to analyze a topic with multiple analysts.
    """
    try:
        # Get the last user message
        messages = state.get("messages", [])
        if not messages:
            return "I need a topic to research. Please provide a topic and the number of analysts you'd like to analyze it."
            
        last_message = messages[-1].content
        
        # Extract topic and number of analysts from the message
        topic, num_analysts = extract_topic_and_count(last_message)
        
        if not topic or not num_analysts:
            return "I need both a topic and the number of analysts. Please provide them in the format: 'Research [topic] with [number] analysts'"
        
        # Get context for the research
        context = search_context(topic, settings)
        
        # Generate research report
        report = generate_research_report(topic, num_analysts, context)
        
        return report
        
    except Exception as e:
        return f"Error in research agent: {str(e)}"
