from dotenv import load_dotenv
from google import genai
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from tavily import TavilyClient


load_dotenv()
tavily = TavilyClient()



llm = init_chat_model(
    "gemini-3.8-flash",
    model_provider="google_genai"
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    industry: str
    candidate_companies: list[dict]

   
def find_companies(state: State):
    industry = state["messages"][-1].content

    # 1. Gemini decides what searches would help us find ML/DL companies
    query_prompt = f"""
    Generate 6 diverse web search queries for discovering real companies
    in the {industry} industry whose products materially use machine
    learning or deep learning.

    Cover different ML/DL applications relevant to this industry.

    Return ONLY the search queries, one query per line.
    Do not number them.
    """

    response = llm.invoke(query_prompt)

    query_text = response.content[0]["text"]

    queries = [
        query.strip()
        for query in query_text.splitlines()
        if query.strip()
    ]

    # 2. Tavily performs the actual web searches
    search_results = []

    for query in queries:
        results = tavily.search(
            query=query,
            max_results=5
        )

        search_results.extend(results["results"])

    # 3. Turn Tavily results into evidence Gemini can read
    evidence = ""

    for result in search_results:
        evidence += f"""
TITLE: {result.get("title")}
URL: {result.get("url")}
CONTENT: {result.get("content")}
"""

    # 4. Gemini decides which companies actually qualify
    analysis_prompt = f"""
    You are researching the {industry} industry.

    Below are real web search results.

    Identify companies whose products materially depend on
    machine learning or deep learning.

    Do not include a company merely because a webpage mentions AI.

    For each qualifying company return:

    - Company name
    - What the company does
    - How ML/DL is used
    - Source URL

    Remove duplicate companies.

    SEARCH EVIDENCE:

    {evidence}
    """

    company_response = llm.invoke(analysis_prompt)

    print(company_response.content[0]["text"])

def collect_evidence(state: State):
   pass

def research_agent(state: State):
   pass

def validation_agent(state: State):
   pass

def build_report(state: State):
   pass



graph_builder = StateGraph(State)

graph_builder.add_node("find_companies", find_companies)
graph_builder.add_edge(START, "find_companies")
graph = graph_builder.compile()

user_input = input("Enter an industry: ")

graph.invoke({
    "messages": [user_input],
    "industry": ""
})


