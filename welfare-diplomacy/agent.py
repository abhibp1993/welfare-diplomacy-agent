from data_types import AgentResponse, AgentParams, prompts
from sample_agent import
from diplomacy import Game, wandb
from typing import Any, Dict

class DiplomacyAgent:
    pass

class WDAgent(DiplomacyAgent):
    def __init__(self, agent_model: "gpt-4o-mini", temperature: float = 0.20, top_p: object = 0.9,
                 max_completion_errors: object = 30, ) -> None:
        self.agent_model = agent_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_errors = max_completion_errors


    def generate_response(self, params: AgentParams) -> AgentResponse:
        system_prompt = prompts.get_system_prompt(params)
        user_prompt = prompts.get_user_prompt(params)
        response = None








