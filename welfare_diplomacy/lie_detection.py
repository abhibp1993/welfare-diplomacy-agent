from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, Optional
from pydantic import BaseModel
from data_types import AgentParams, AgentResponse
from lie_detection_prompts import (
    get_is_lie_candidate_prompt,
    get_extract_promises_prompt,
    get_lie_evaluator_prompt,
)
from model import model, runnable2, ModelMessage
import time

# ===== Setup =====
class Promise(BaseModel):
    sender: str
    receiver: str
    unit1: Optional[str] = None
    unit2: Optional[str] = None
    act1: Optional[str] = None
    act2: Optional[str] = None

class MessageState(TypedDict):
    message: str
    is_lie_candidate: Optional[bool]
    promises: dict
    is_lie: Optional[bool]
    editing_context: Optional[str]

class BooleanAnswer_LieCandidate(BaseModel):
    status: bool

class BooleanAnswer_LieEvaluator(BaseModel):
    status: bool
    context: Optional[str]


config = {
    "configurable": {
        "thread_id": "1"
    }
}

openai_base_url = "http://localhost:12434/engines/llama.cpp/v1/"
openai_api_key = "docker"

memory = InMemorySaver()

# ===== Lie Detection Subgraph =====
def lie_detection(params: AgentParams, message):
    graph = StateGraph(MessageState)

    def revise_messages(state: MessageState):
        user_prompt = (
            state["editing_context"]
        )
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        while True:
            try:
                response = runnable2.invoke({"editing_context": messages, "is_revise": True}, config = config)
                print("revise_message success")
                # Extract the updated message (assuming same format as before)
                return {
                    "message": response,
                    "editing_context": None  # clear context so it can exit loop
                }
            except Exception as e:
                print("Error in revise_message:", e)
                time.sleep(2)

    def is_lie_candidate(state: MessageState):
        sys_prompt = get_is_lie_candidate_prompt()
        user_prompt = str(state["message"])
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        while True:
            try:
                response = model.with_structured_output(BooleanAnswer_LieCandidate).invoke(messages)
                print("is_lie_candidate node success")
                return {"is_lie_candidate": response.status}

            except Exception as e:
                print("Error in is_lie_candidate:", e)
                time.sleep(2)

    def extract_promises(state: MessageState):
        sys_prompt = get_extract_promises_prompt()
        user_prompt = str(state["message"])
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        while True:
            try:
                response = model.with_structured_output(Promise).invoke(messages)
                print("extract_promises success")
                return {"promises": response}

            except Exception as e:
                print("Error in extract_promises:", e)
                time.sleep(2)
        if promises == {}:
            return {"is_lie_candidate": False}


    def lie_evaluator(state: MessageState):
        sys_prompt = get_lie_evaluator_prompt()
        user_prompt = str(state["promises"])
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        while True:
            try:
                response = model.with_structured_output(BooleanAnswer_LieEvaluator).invoke(messages)
                print("lie_evaluator success")
                return {
                    "is_lie": response.status,
                    "editing_context": response.context
                }
            except Exception as e:
                print("Error in lie_evaluator:", e)
                time.sleep(2)

    def candidate_router(state: MessageState):
        return "extract_promises" if state["is_lie_candidate"] is True else END

    def promise_router(state: MessageState):
        if not state.get("promises"):  # handles both {} and None
            return END  # or another terminal node
        return "lie_evaluator"

    def revision_router(state: MessageState):
        return "revise_messages" if state.get("editing_context") else END

    graph.add_node("is_lie_candidate", is_lie_candidate)
    graph.add_node("extract_promises", extract_promises)
    graph.add_node("lie_evaluator", lie_evaluator)
    graph.add_node("revise_messages", revise_messages)


    graph.add_edge(START, "is_lie_candidate")
    graph.add_conditional_edges("is_lie_candidate", candidate_router)
    graph.add_conditional_edges("extract_promises", promise_router)
    graph.add_conditional_edges("lie_evaluator", revision_router)
    graph.add_edge("revise_messages", "is_lie_candidate")
    graph.add_edge("lie_evaluator", END)

    runnable = graph.compile()
    initial_state = {
        "message": message,
        "is_lie_candidate": None,
        "promises": {},
        "is_lie": None,
        "editing_context": None
    }
    final_state = runnable.invoke(initial_state)
    return final_state




