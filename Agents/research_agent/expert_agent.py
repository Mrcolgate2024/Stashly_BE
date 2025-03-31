from dataclasses import dataclass
from typing import List

@dataclass
class Analyst:
    name: str
    role: str
    persona: str
    description: str

def create_analysts(state: dict) -> dict:
    """
    Create a list of analysts with different roles and perspectives.
    """
    topic = state["topic"]
    max_analysts = state.get("max_analysts", 3)
    
    analysts = [
        Analyst(
            name="Market Analyst",
            role="You are a market analyst specializing in market trends and analysis.",
            persona="You are an experienced market analyst with deep knowledge of financial markets and trends.",
            description="Market trends and analysis"
        ),
        Analyst(
            name="Technical Analyst",
            role="You are a technical analyst focusing on technical indicators and patterns.",
            persona="You are a technical analysis expert who understands market patterns and indicators.",
            description="Technical analysis and patterns"
        ),
        Analyst(
            name="Fundamental Analyst",
            role="You are a fundamental analyst examining underlying factors and fundamentals.",
            persona="You are a fundamental analyst who looks at core business and economic factors.",
            description="Fundamental analysis and underlying factors"
        )
    ]
    
    # Limit the number of analysts based on max_analysts
    analysts = analysts[:max_analysts]
    
    return {"analysts": analysts} 