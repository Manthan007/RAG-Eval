import RAG1.tools as t1
from .prompts import system_agent_prompt
from langchain_groq import ChatGroq
from .tools import collecting_info
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from .objects import Answer

# Import these to fix the parsing architecture
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=6)

# 1. REPLACE with_structured_output with a Pydantic parser chain
parser = PydanticOutputParser(pydantic_object=Answer)
format_instructions = parser.get_format_instructions()

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a parser. Parse the provided information exactly into JSON based on the format instructions.\n\n{format_instructions}"),
    ("human", "Based on this answer: {agent_text} and retrieved context: {retrieved_context}, populate the schema.")
])

# Build an extraction chain that parses text natively into your Answer object
structured_chain = extraction_prompt | llm | parser

agent = create_agent(
    model=llm,
    tools=[collecting_info],
    system_prompt=system_agent_prompt
)

def ask(question) -> dict:
    """ Answer the user's question about the PDF"""
    t1.last_retrieved_context = []

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )
    
    agent_text = result["messages"][-1].content

    # 2. INVOKE your newly structured chain with the format instructions
    structured_response = structured_chain.invoke({
        "agent_text": agent_text,
        "retrieved_context": t1.last_retrieved_context,
        "format_instructions": format_instructions
    })

    return {
        "answer": structured_response.answer,
        "context": structured_response.context
    }

print(ask("what is the diet of Penguins"))
