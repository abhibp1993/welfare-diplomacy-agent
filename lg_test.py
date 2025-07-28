from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from sympy.logic.boolalg import Boolean
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


openai_base_url = "http://localhost:12434/engines/llama.cpp/v1/"
openai_api_key = "docker"
model = ChatOpenAI(
    base_url = openai_base_url,
    api_key = openai_api_key,
    model = "ai/llama3.2",  # e.g., "llama-2", "opt-1.3b", "mistral"
)

class AgentState(TypedDict):
    graph_state: str
    is_truth: str




graph = StateGraph(AgentState)
graph.add_node("sentence_generator", sentence_generator)
graph.add_node("lie_detector", lie_detector)

graph.add_edge(START, "sentence_generator")
graph.add_edge("sentence_generator", "lie_detector")
graph.add_edge("lie_detector", END)

runnable = graph.compile()

initial_state = {
    "graph_state": "",
    "is_truth": ""
}

final_state = runnable.invoke(initial_state)
print (final_state)
