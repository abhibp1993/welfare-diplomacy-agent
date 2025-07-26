# # from data_types import AgentResponse, AgentParams, prompts
# # from diplomacy import Game, wandb
# # from typing import Dict
#
#
# # class DiplomacyAgent:
# #     pass
#
# # class WDAgent(DiplomacyAgent):
# #     def __init__(self, agent_model: "gpt-4o-mini", temperature: float = 1.0, top_p: object = 0.9,
# #                 max_completion_errors: object = 30, ) -> None:
# #         self.agent_model = agent_model
# #         self.temperature = temperature
# #         self.top_p = top_p
# #         self.max_completion_errors = max_completion_errors
#
#
# #     def generate_response(self, params: AgentParams) -> Dict[str, AgentResponse]:
# #         system_prompt = prompts.get_system_prompt(params)
# #         user_prompt = prompts.get_user_prompt(params)
# #         response = None
#
#
# import random
#
# import diplomacy
#
# from typing import Dict, List, Optional, Literal
# from typing import TypedDict, Annotated
# from langchain_core.messages import HumanMessage, AIMessage
# from langgraph_draft.graph import StateGraph, START, END
#
# Powers = Literal["FRA", "ITA", "RUS", "ENG", "GER", "AUS", "TUR"]
#
# def get_class(class_name: str):
#     if class_name.lower() == "WDAgent".lower():
#         return WDAgent
#     raise ValueError(f"Unknown class name: {class_name}")
#
#
# class DiplomacyAgent:
#     def __init__(self, game: diplomacy.Game, pow_name: str, **params):
#         assert pow_name in game.powers.keys(), \
#             f"Power {pow_name} not in game powers: {game.powers.keys()}. Names are case sensitive."
#         self.game = game
#         self.pow_name = pow_name
#
#         # Phase tracking
#         self._curr_phase = None
#         self._is_phase_ongoing = False
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.pow_name})"
#
#     def start_phase(self):
#         self._curr_phase = self.game.get_current_phase()
#         self._is_phase_ongoing = True
#
#     def end_phase(self):
#         self._is_phase_ongoing = False
#
#     def generate_messages(self):
#         raise NotImplementedError("generate_messages method must be implemented in subclasses")
#
#     def generate_orders(self):
#         raise NotImplementedError("generate_orders method must be implemented in subclasses")
#
#
# class WDAgent(DiplomacyAgent):
#     def __init__(self):
#         # initialize your Langgraph StateGraph
#         workflow = StateGraph()
#         self._graph = None  # langgraph course, single agent
#         super().__init__(game=None, pow_name=None)  # game and pow_name will
#
#     def generate_messages(self):
#     # Fetch relevant information for prompt (welfare-diplomacy specific)
#     # Generate user and system prompts (welfare-diplomacy specific)
#     # invoke graph to generate messages (langgraph specific)
#     # Return messages in the required format (use pydantic for structured output)
#
#     def generate_orders(self):
#     # Fetch relevant information for prompt
#     # Generate user and system prompts
#     # invoke graph to generate messages
#     # Return messages in the required format (use pydantic for structured output)
#
#
#
#
#
#
#
