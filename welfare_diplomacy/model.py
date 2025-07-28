from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from data_types import AgentResponse
from pydantic import BaseModel

class ModelMessage(TypedDict):
    editing_context: str
    revise_message: str
    message: str
    messages: dict

class AgentMessage(BaseModel):
    response: dict


builder = StateGraph(ModelMessage)

checkpointer = MemorySaver()
openai_base_url = "http://localhost:12434/engines/llama.cpp/v1/"
openai_api_key = "docker"
model = ChatOpenAI(
    base_url = openai_base_url,
    api_key = openai_api_key,
    model = "ai/llama3.2",  # e.g., "llama-2", "opt-1.3b", "mistral"
)

def model_router(state: ModelMessage):
    if state.get("editing_context"):
        return "agent_revise"
    else:
        return "agent_response"



def agent_response(state: ModelMessage):
    while True:
        try:
            result = model.with_structured_output(AgentResponse).invoke(state["message"])
            return {"messages": result.messages}
        except Exception as e:
            print("Error in message generation", e)


def agent_revise(state: ModelMessage):
    while True:
        try:
            result = model.with_structured_output(AgentMessage).invoke(state["editing_context"])
            return {"messages": result.messages}
        except Exception as e:
            print ("Error in revise_messages", e)
            print ("raw output:")
            print(model.invoke(state["messages"]))

builder.add_node("agent_response", agent_response)
builder.add_node("agent_revise", agent_revise)

builder.add_conditional_edges(START, model_router)
builder.add_edge("agent_response", END)
builder.add_edge("agent_revise", END)

runnable2 = builder.compile(checkpointer = checkpointer)

