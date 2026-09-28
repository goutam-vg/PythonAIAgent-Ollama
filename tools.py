from langchain_community.tools import (
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime


# -------------------------
# Save Tool
# -------------------------

def save_to_txt(
    data: str,
    filename: str = "research_output.txt"
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    formatted_text = (
        "--- Research Output ---\n"
        f"Timestamp: {timestamp}\n\n"
        f"{data}\n\n"
    )

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves research data to a text file."
)


# -------------------------
# DuckDuckGo Search Tool
# -------------------------

search = DuckDuckGoSearchRun()

def web_search(query: str) -> str:
    try:
        return search.run(query)
    except Exception as error:
        return f"Web search unavailable: {error}"


search_tool = Tool(
    name="web_search",
    func=web_search,
    description=(
        "Search the web using DuckDuckGo. "
        "Use this when current or online information is required."
    )
)


# -------------------------
# Wikipedia Tool
# -------------------------

api_wrapper = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000
)

wiki_search = WikipediaQueryRun(
    api_wrapper=api_wrapper
)


def wikipedia_lookup(query: str) -> str:
    try:
        return wiki_search.run(query)
    except Exception as error:
        return f"Wikipedia unavailable: {error}"


wiki_tool = Tool(
    name="wikipedia_lookup",
    func=wikipedia_lookup,
    description="Look up background information on Wikipedia."
)