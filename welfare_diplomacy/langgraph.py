from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph
from langgraph.graph import END
from typing import Annotated
from langchain_openai import ChatOpenAI

class DiplomacyAgent:
    pass


    model = "ai/llama3.2:1B-Q8_0"
    # System message
    sys_msg = SystemMessage(content="You are an expert AI playing the game Diplomacy. You use the tools given to take the most effective negotiating style")

    # Define the assistant node
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Define agent 1: Individualistic Agent
    def individualistic_agent(state: MessagesState):

    # Define agent 2: Cooperative Agent
    def cooperative_agent(state: MessagesState):

    # Define agent 3: Competitive Agent
    def competitive_agent(state: MessagesState):

    # Define agent 4: Altruistic Agent
    def altruistic_agent(state: MessagesState):

    # create a tool calling a LLM node (supervisor) and a tool executing node
    supervisor = create_supervisor (
        agents=[individualistic_agent, cooperative_agent, competitive_agent, altruistic_agent],
        model = init_chat_model("gpt-4o-mini") #temperature = 0.7
        # prompt = (...),
        add_handoff_back_messages = True,
        output_mode = "full_history",
    )

