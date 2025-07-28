from welfare_diplomacy.agents.base_agent import DiplomacyAgent
from welfare_diplomacy.agents.wd_agent import WDAgent
from welfare_diplomacy.agents.personality_negotiators import PersonalityAgent, DynamicPersonalityAgent


def get_class(class_name: str):
    if class_name.lower() == "WDAgent".lower():
        return WDAgent
    elif class_name.lower() == "PersonalityAgent".lower():
        return PersonalityAgent
    elif class_name.lower() == "DynamicPersonalityAgent".lower():
        return DynamicPersonalityAgent

    raise ValueError(f"Unknown class name: {class_name}")
