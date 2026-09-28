from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from tools import save_tool

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)


agent = create_agent(
    model=llm,
    tools=[save_tool],
    system_prompt=(
        "You are a research assistant. "
        "Answer the user's question clearly."
    ),
    response_format=ResearchResponse,
)


query = input("What can I help you research? ")

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }
)

print(response["structured_response"])