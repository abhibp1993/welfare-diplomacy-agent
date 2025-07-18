from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from data_types import AgentParams, AgentResponse
import prompts

from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

openai_base_url = "http://localhost:12434/engines/llama.cpp/v1/"
openai_api_key = "docker"

model = ChatOpenAI(
    base_url = openai_base_url,
    api_key = openai_api_key,
    model = "ai/llama3.2",
    )

def generate_response(params: AgentParams) -> AgentResponse:
    sys_prompt = prompts.get_system_prompt(params)
    user_prompt = prompts.get_user_prompt(params)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    model.invoke()


msg_generator = create_react_agent(
    model = model,
    tools = [],
    name = "msg_generator",
    prompt = (
        "You are an AI agent responsible for editing messages for a board game called Diplomacy.\n"
        "You will receive input from a supervisor agent.\n"
        "Your goal is to use these parameters to craft in-game messages or orders relevant to the current game context.\n"
        "Use the available tools to generate responses based on the input.\n"
        "Do not make up information—only act based on the provided parameters and tools.\n"
        "Respond with a single message or orders that are appropriate for the given game state."
    )
)

lie_detector = create_react_agent(
    model = model,
    tools = [],
    name = "lie_detector",
    prompt =
    "You are an agent that only responds with 'True' or 'False''"
    "You will receive a paragraph. Only respond 'True' or 'False' ",
)
prompt = (
    "You are a supervisor responsible for coordinating two agents: a message editor agent and a lie detector agent.\n\n"
    "Your goal is to iterate through a cycle until a valid (true) message is produced.\n\n"
    "Instructions:\n"
    "1. You will receive two inputs: a dictionary of messages, a dictionary of orders.\n"
    "2. Pass both dictionaries to the lie detector agent for evaluation.\n"
    "3. If the lie detector returns 'True', return the original dictionaries and terminate the process.\n"
    "4. If the lie detector returns 'False', transfer control to the message editor agent.\n"
    "   - Provide the message editor with the lie detector's feedback and ask them to revise the messages accordingly.\n"
    "5. Repeat the process with the revised messages until a true statement is confirmed."
)

workflow = create_supervisor (
    [msg_generator, lie_detector],
    model = model,
    output_mode = "last_message",
    prompt = prompt,
)
app = workflow.compile()


