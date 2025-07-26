# import json
# import random
# from typing import Dict, List, Optional, Literal
#
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, START
# from pydantic import BaseModel, Field
#
# import diplomacy
# from welfare_diplomacy.agents.base_agent import DiplomacyAgent
#
# Powers = Literal["FRA", "ITA", "RUS", "ENG", "GER", "AUS", "TUR"]
#
#
# class NegotiationMessage(BaseModel):
#     messages_to_send: Dict[Powers, str] = Field(default_factory=dict)
#
#
# class AgentState(BaseModel):
#     current_power: str
#     phase: str
#     received_messages: Dict[str, List[str]] = Field(default_factory=dict)
#     messages_to_send: Dict[Powers, str] = Field(default_factory=dict)
#     # game_state_summary: Optional[str] = None
#
#
# class WDAgent(DiplomacyAgent):
#
#     def __init__(self, game: diplomacy.Game, pow_name: str, **params):
#         super().__init__(game, pow_name, **params)
#
#         # Initialize LLM model with parameters
#         self._base_url = params["model_provider_url"]
#         self._api_key = params["api_key"]
#         self._model_name = params["model"]
#
#         if self._api_key == "docker":
#             self._model_name = "ai/" + params["model_name"]
#
#         self.model = ChatOpenAI(
#             base_url=self._base_url,
#             api_key=self._api_key,
#             model=self._model_name
#         )
#         self.model_message = self.model.with_structured_output(NegotiationMessage)
#
#         # Initialize generate-messages agent
#         graph = StateGraph(state_schema=AgentState)
#         graph.add_node("chatbot", self._node_chatbot)
#         graph.add_edge(START, "chatbot")
#         self.generate_messages_agent = graph.compile()
#
#     def generate_messages(self):
#         state = AgentState(
#             current_power=self.pow_name,
#             phase=self.game.get_current_phase(),
#             received_messages={
#                 "FRA": ["Let's work together against AUS."],
#                 "GER": ["Can I trust FRA?"],
#                 "AUS": ["Peace in the south?"]
#             },
#             # game_state_summary="FRA is posturing as cooperative. GER is cautious. AUS is hedging.",
#         )
#         msg = self.generate_messages_agent.invoke(state)
#         print(msg)
#
#         trial = {
#             "ENGLAND": "Hi!",
#             "FRANCE": "Hi!",
#             "TURKEY": "Hi!",
#             "GERMANY": "Hi!",
#             "RUSSIA": "Hi!",
#             "AUSTRIA": "Hi!",
#             "ITALY": "Hi!",
#         }
#         del trial[self.pow_name]
#         return trial
#
#     def generate_orders(self):
#         # Get all locations where this power can issue orders
#         orderable_locations = self.game.get_orderable_locations(self.pow_name)
#         orders = []
#
#         for location in orderable_locations:
#             # Get all possible orders for the current location
#             possible_orders = self.game.get_all_possible_orders()
#             if possible_orders[location]:
#                 # Randomly select one valid order
#                 orders.append(random.choice(possible_orders[location]))
#
#         return orders
#
#     def _node_chatbot(self, state: AgentState):
#         """
#         Node function to generate messages using the LLM model.
#         """
#         system_prompt = f"""
#         You are a skilled agent playing the board game Diplomacy.
#         You control the power of {self.pow_name}.
#         Your goal is to maximize your advantage through strategic negotiations with other powers.
#         During this negotiation phase ({self.game.get_current_phase()}), review incoming messages from other players
#         and decide what messages to send in return.
#
#         Keep your tone persuasive, strategic, and aligned with your power’s interests.
#         Do not reveal your full intentions. Use subtlety, alliances, and deception as appropriate to advance your position.
#
#         Output messages to send as reply, each specifying the recipient (power) and the message.
#         """
#
#         user_prompt = json.dumps(state.dict())
#
#         return self.model_message.invoke([
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=user_prompt)
#         ])