from pydantic import BaseModel, Field
from typing import List, Dict

class AgentResponse(BaseModel):
    reasoning: str = Field(description="Private reasoning for this turn")
    orders: List[str] = Field(description="Orders to execute")
    messages: Dict[str, str] = Field(description="Messages to send to other powers")



from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Bind the model to the schema using structured output
llm_with_schema = llm.with_structured_output(AgentResponse)

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# Define a node that runs the LLM with schema
def generate_agent_response(state):
    return {"response": llm_with_schema.invoke(state["prompt"])}

# Build the graph
graph_builder = StateGraph()
graph_builder.add_node("generate_response", RunnableLambda(generate_agent_response))
graph_builder.set_entry_point("generate_response")
graph_builder.set_finish_point("generate_response", END)

graph = graph_builder.compile()