import uuid

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from tools import search_tool, wiki_tool, save_tool

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Change this to the Ollama model you installed
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)


tools = [
    search_tool,
    wiki_tool,
    save_tool
]


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a research assistant that helps generate research papers. "
        "Research the user's query using the available tools when necessary. "
        "Provide a clear topic, detailed summary, sources used, "
        "and list the tools you used."
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

print("\n--- Research Result ---")

result = response["structured_response"]

print(f"\nTopic: {result.topic}")
print(f"\nSummary:\n{result.summary}")
print(f"\nSources:")
for source in result.sources:
    print(f"- {source}")

print(f"\nTools Used:")
for tool in result.tools_used:
    print(f"- {tool}")