from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict, Optional, Annotated
import operator
from lie_detection import lie_detection, MessageState
from data_types import AgentParams, AgentResponse
from prompts import get_system_prompt
from prompts import get_user_prompt
from pydantic import BaseModel
from model import model, runnable2

import time

config = {
    "configurable": {
        "thread_id": "1"
    }
}

class AgentState(TypedDict):
    messages: dict


def lie_detector_map_reduce(params: AgentParams):
    sys_prompt = get_system_prompt(params)
    user_prompt = get_user_prompt(params)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    def generate_messages(state: AgentState):
        while True:
            try:
                response = runnable2.invoke({"message": messages}, config = config)
                list_of_mappings = [{k: v} for k, v in response["messages"].items()]
                return {"messages": list_of_mappings}
            except Exception as e:
                print("Error in generate_message:", e)
                time.sleep(2)

    def run_lie_detection_subgraph(state: MessageState):
        lie_detection(params, state["message"])
        return {"message": state["message"]}

    def map_messages(state: AgentState):
        return [Send("run_lie_detection_subgraph", {"message": m}) for m in state["messages"]]

    def aggregate_results(state: AgentState):
        print (messages)
        return {"messages": state["messages"]}

    builder = StateGraph(AgentState)
    builder.add_node("generate_messages", generate_messages)
    builder.add_node("run_lie_detection_subgraph", run_lie_detection_subgraph)
    builder.add_node("aggregate_results", aggregate_results)


    builder.add_edge(START, "generate_messages")
    builder.add_conditional_edges("generate_messages", map_messages, ["run_lie_detection_subgraph"])
    builder.add_edge("run_lie_detection_subgraph", "aggregate_results")
    builder.add_edge("aggregate_results", END)

    initial_state = {
        "messages": []
    }
    graph = builder.compile()
    final_state = graph.invoke(initial_state)
    return final_state
