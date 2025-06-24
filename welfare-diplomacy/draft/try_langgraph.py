from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# llm = init_chat_model("")
# Create OpenAI-compatible model
openai_base_url = "http://localhost:12434/engines/llama.cpp/v1/"
openai_api_key = "docker"
model = ChatOpenAI(
    base_url=openai_base_url,
    api_key=openai_api_key,
    model="ai/llama3.2",  # e.g., "llama-2", "opt-1.3b", "mistral"
)


def chatbot(state: State):
    return {"messages": [model.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "Tell me a joke."}]})
print(type(result))
print(result)
