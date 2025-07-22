import random
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from pydantic import BaseModel, Field

import diplomacy


def get_class(class_name: str):
    if class_name.lower() == "WDAgent".lower():
        return WDAgent
    raise ValueError(f"Unknown class name: {class_name}")


class DiplomacyAgent:
    def __init__(self, game: diplomacy.Game, pow_name: str, **params):
        assert pow_name in game.powers.keys(), \
            f"Power {pow_name} not in game powers: {game.powers.keys()}. Names are case sensitive."
        self.game = game
        self.pow_name = pow_name

        # Phase tracking
        self._curr_phase = None
        self._is_phase_ongoing = False

    def __repr__(self):
        return f"{self.__class__.__name__}({self.pow_name})"

    def start_phase(self):
        self._curr_phase = self.game.get_current_phase()
        self._is_phase_ongoing = True

    def end_phase(self):
        self._is_phase_ongoing = False

    def generate_messages(self):
        raise NotImplementedError("generate_messages method must be implemented in subclasses")

    def generate_orders(self):
        raise NotImplementedError("generate_orders method must be implemented in subclasses")


class WDAgent(DiplomacyAgent):
    class NegotiationMessage(BaseModel):
        to_power: str
        message: str

    class AgentState(BaseModel):
        current_power: str
        phase: str
        received_messages: Dict[str, List[str]] = Field(default_factory=dict)
        messages_to_send: List[NegotiationMessage] = Field(default_factory=list)
        game_state_summary: Optional[str] = None

    def __init__(self, game: diplomacy.Game, pow_name: str, **params):
        super().__init__(game, pow_name, **params)

        # Initialize LLM model with parameters
        self._base_url = params["model_provider_url"]
        self._api_key = params["api_key"]
        self._model_name = params["model"]

        if self._api_key == "docker":
            self._model_name = "ai/" + params["model_name"]

        self.model = ChatOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model_name
        )

        # Initialize generate-messages agent
        graph = StateGraph(state_schema=WDAgent.State)
        graph.add_node("chatbot", self._node_chatbot)
        graph.add_edge(START, "chatbot")
        self.generate_messages_agent = graph.compile()

    def generate_messages(self):
        msg = self.generate_messages_agent.invoke({"a": "hello"})
        print(msg)
        trial = {
            "ENGLAND": "Hi!",
            "FRANCE": "Hi!",
            "TURKEY": "Hi!",
            "GERMANY": "Hi!",
            "RUSSIA": "Hi!",
            "AUSTRIA": "Hi!",
            "ITALY": "Hi!",
        }
        del trial[self.pow_name]
        return trial

    def generate_orders(self):
        # Get all locations where this power can issue orders
        orderable_locations = self.game.get_orderable_locations(self.pow_name)
        orders = []

        for location in orderable_locations:
            # Get all possible orders for the current location
            possible_orders = self.game.get_all_possible_orders()
            if possible_orders[location]:
                # Randomly select one valid order
                orders.append(random.choice(possible_orders[location]))

        return orders

    def _node_chatbot(self, state: State):
        """
        Node function to generate messages using the LLM model.
        """
        return {"a": "goodbye"}


class HardcodedAgent(DiplomacyAgent):
    def generate_messages(self):
        trial = {
            "ENGLAND": "Hi!",
            "FRANCE": "Hi!",
            "TURKEY": "Hi!",
            "GERMANY": "Hi!",
            "RUSSIA": "Hi!",
            "AUSTRIA": "Hi!",
            "ITALY": "Hi!",
        }
        del trial[self.pow_name]
        return trial

    def generate_orders(self):
        # Get all locations where this power can issue orders
        orderable_locations = self.game.get_orderable_locations(self.pow_name)
        orders = []

        for location in orderable_locations:
            # Get all possible orders for the current location
            possible_orders = self.game.get_all_possible_orders()
            if possible_orders[location]:
                # Randomly select one valid order
                orders.append(random.choice(possible_orders[location]))

        return orders
