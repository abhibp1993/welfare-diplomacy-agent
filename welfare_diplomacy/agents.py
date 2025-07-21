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


class WDAgent(DiplomacyAgent):
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