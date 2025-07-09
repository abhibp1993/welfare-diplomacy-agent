from diplomacy import Game
from data_types import MessageSummaryHistory, AgentParams
import prompts
from abc import ABC, abstractmethod
from diplomacy import Power
from try_langgraph import model_with_tools
import json


class DiplomacyAgent:
    pass

class WdAgent:

    def __init__(self, power_name, agent_model, temperature, top_p, max_completion_errors):
        self.power_name = power_name
        self.agent_model = agent_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_errors = max_completion_errors
        self.response = None

    def generate_response(self, params: AgentParams) -> dict:
        sys_prompt = prompts.get_system_prompt(params)
        user_prompt = prompts.get_user_prompt(params)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        result = model_with_tools.invoke({"messages": messages})
        print (result)
        return result

    def generate_messages(self, params: AgentParams):
        if self.response:
            return {params.power.name: self.response.get("messages", {})}
        else:
            return {}


    def generate_orders(self, params: AgentParams):
        if self.response:
            return self.response.get("orders", [])
        else:
            return []

    def generate_reasoning(self, params: AgentParams):
        if self.response:
            return {params.power.name: self.response.get("reasoning", {})}

agent_class_map = {
    "WdAgent": WdAgent
}

