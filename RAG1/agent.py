import tools as t1
from prompts import system_agent_prompt
from langchain_groq import ChatGroq
from tools import collecting_info
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from objects import Answer

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

structured_llm = llm.with_structured_output(Answer)

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

    structured_response = structured_llm.invoke(
        f"Based on this answer: {agent_text} and retrieved context: {t1.last_retrieved_context}, populate the schema."
    )

    return {
        "answer": structured_response.answer,
        "context": structured_response.context
    }

print(ask("what is the diet of Penguins"))
