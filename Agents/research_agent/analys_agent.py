from langchain.agents import AgentExecutor, create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.tools import tool

@tool
def break_down_topic(topic: str) -> str:
    """
    Break down a research topic into 3-5 clear subtopics or questions.
    Input: Full research topic or question.
    Output: A list of subtopics as plain text.
    """
    return f"""Thought: I should break down the topic.
Action: break_down_topic
Action Input: {topic}
Observation: [1. Background, 2. Key challenges, 3. Innovations, 4. Market implications, 5. Future outlook]
Thought: I now know the final answer.
Final Answer: 1. Background\n2. Key challenges\n3. Innovations\n4. Market implications\n5. Future outlook
"""

# Prompt
analyst_prompt_template = """
You are a research analyst AI who breaks a research question into 3–5 relevant subtopics.
Use the format below:

Question: {input}
{agent_scratchpad}
"""

# Set up LLM
llm = ChatOpenAI(temperature=0.3)

# Create prompt
prompt = PromptTemplate.from_template(analyst_prompt_template)

# Set up memory (if needed)
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# Create agent
analyst_agent = create_react_agent(llm=llm, tools=[break_down_topic], prompt=prompt)

# Create executor
analyst_executor = AgentExecutor.from_agent_and_tools(
    agent=analyst_agent,
    tools=[break_down_topic],
    verbose=True,
    return_intermediate_steps=True,
    max_iterations=3,
)

async def run_analyst_agent(input_text: str) -> str:
    result = await analyst_executor.ainvoke({"input": input_text})
    return result.get("output")