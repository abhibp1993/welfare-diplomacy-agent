from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import random
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor


openai_base_url = "http://localhost:12434/engines/llama.cpp/v1/"
openai_api_key = "docker"
# model = ChatOpenAI(
#     base_url=openai_base_url,
#     api_key=openai_api_key,
#     model="ai/llama3.2"
# )
class AgentState(TypedDict):
    agent_response: str
    is_truth: str | None

# def get_prompts(params: AgentParams):
#     sys_prompt = prompts.get_system_prompt(params)
#     user_prompt = prompts.get_user_prompt(params)
#     return sys_prompt, user_prompt

msg_generator = create_react_agent(
    model = ChatOpenAI(
        base_url = openai_base_url,
        api_key = openai_api_key,
        model = "ai/llama3.2",
    ),
    tools = [],
    name = "msg_generator",
)

lie_detector = create_react_agent(
    model = ChatOpenAI(
        base_url = openai_base_url,
        api_key = openai_api_key,
        model = "ai/llama3.2",
    ),
    tools = [],
    name = "lie_detector",
)

supervisor_agent = create_react_agent(
    model = ChatOpenAI(
        base_url = openai_base_url,
        api_key = openai_api_key,
        model = "ai/llama3.2",
    ),
    tools=[],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor",
)

supervisor = create_supervisor(
    model = ChatOpenAI(
        base_url = openai_base_url,
        api_key = openai_api_key,
        model = "ai/llama3.2",
    ),
    agents=[msg_generator, lie_detector],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()


# Define the multi-agent supervisor graph
graph = (
    StateGraph(AgentState)
    # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
    .add_node(supervisor_agent, destinations=("msg_generator", "lie_detector", END))
    .add_node(msg_generator)
    .add_node(lie_detector)
    .add_edge(START, "supervisor")
    # always return back to the supervisor
    .add_edge("msg_generator", "supervisor")
    .add_edge("lie_detector", "supervisor")
    .compile()
)
