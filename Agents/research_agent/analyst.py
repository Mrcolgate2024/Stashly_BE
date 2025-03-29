from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst's focus and concerns.")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="List of generated analysts.")

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]

def create_analysts(state: GenerateAnalystsState) -> GenerateAnalystsState:
    structured_llm = llm.with_structured_output(Perspectives)

    system_message = f"""
You are tasked with creating a set of AI analyst personas. Follow these instructions:

1. Review the research topic: {state['topic']}
2. Review any editorial feedback: {state['human_analyst_feedback']}
3. Identify {state['max_analysts']} distinct angles or subtopics.
4. Assign one analyst per subtopic.
"""

    result = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Generate the set of analysts.")
    ])

    return {
        "topic": state["topic"],
        "max_analysts": state["max_analysts"],
        "human_analyst_feedback": state["human_analyst_feedback"],
        "analysts": result.analysts
    }
