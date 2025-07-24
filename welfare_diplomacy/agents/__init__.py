from welfare_diplomacy.agents.base_agent import DiplomacyAgent
from welfare_diplomacy.agents.wd_agent import WDAgent


def get_class(class_name: str):
    if class_name.lower() == "WDAgent".lower():
        return WDAgent
    raise ValueError(f"Unknown class name: {class_name}")
