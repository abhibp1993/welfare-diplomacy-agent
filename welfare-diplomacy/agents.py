from data_types import AgentParams, AgentResponse
import prompts
from try_langgraph import model_with_structure
import time
from welfare_diplomacy_baselines.baselines import no_press_policies
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
        while True:
            try:
                self.response = model_with_structure.invoke(messages)
                print ("success")
                break
            except Exception as e:
                time.sleep(1)
        return self.response


    def generate_messages(self, params: AgentParams):
        if self.response:
            return {params.power.name: self.response.messages}
        else:
            return {}


    def generate_orders(self, params: AgentParams):
        if self.response:
            return self.response.orders
        else:
            return []

    def generate_reasoning(self, params: AgentParams):
        if self.response:
            return self.response.reasoning
        return None

class RLAgent:

    def __init__(self, power_name, agent_model, temperature, top_p, max_completion_errors):
        self.power_name = power_name
        self.agent_model = agent_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_errors = max_completion_errors
        self.response = None
        self.rl_policy = no_press_policies.get_network_policy_instance()


    def generate_orders(self):



agent_class_map = {
    "WdAgent": WdAgent,
    "RLAgent": RLAgent
}

