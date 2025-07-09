from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from data_types import AgentResponse


openai_base_url = "http://localhost:12434/engines/llama.cpp/v1/"
openai_api_key = "docker"
model = ChatOpenAI(
    base_url = openai_base_url,
    api_key = openai_api_key,
    model = "ai/llama3.2",  # e.g., "llama-2", "opt-1.3b", "mistral"
)
model_with_tools = model.bind_tools([AgentResponse])

