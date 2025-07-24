"""
Language model scaffolding to play Diplomacy.


"""

import argparse
import pprint
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
# import constants
# import utils
import wandb
<<<<<<< HEAD:welfare-diplomacy/simulate.py
=======
import yaml
from loguru import logger
from rich.progress import Progress
from tqdm import tqdm

import welfare_diplomacy.agents as agents
>>>>>>> origin/main:welfare_diplomacy/simulate.py
# from agents import Agent, AgentCompletionError, model_name_to_agent
# from data_types import (
#     AgentResponse,
#     AgentParams,
#     MessageSummaryHistory,
#     PromptAblation,
# )
from diplomacy import Game, Message


# logger.configure(handlers=[{"sink": RichHandler()}])


# from diplomacy.utils.export import to_saved_game_format
# from experiments import api_pricing
# from message_summarizers import (
#     MessageSummarizer,
#     model_name_to_message_summarizer,
# )


def main():
    # Load configuration
    # game_config: dict = parse_args()
    with open('run_configs/config0.yml', 'r') as file:
        game_config = yaml.safe_load(file)
    logger.info(f"Loaded game configuration: \n{pprint.pformat(game_config)}")

    # Initialize W&B
    wandb.init(
        entity=game_config["wandb"]["entity"],
        project=game_config["wandb"]["project"],
        dir=game_config["wandb"].get("wandb_dir", None),
        name=game_config["wandb"]["run_name"],
        save_code=game_config["wandb"]["save_code"],
        config=game_config,
        mode="disabled" if game_config["wandb"]["disable"] else "online",
        settings=wandb.Settings(code_dir="."),
    )
    assert wandb.run is not None

    # Initialize data logging
    data_dir = init_data_log_directory(
        run_name=wandb.run.name,
        prefix=Path(game_config["logging"]["output_folder"]).absolute(),
    )
    data = dict()
    data["responses"] = pd.DataFrame(
        columns=["phase", "time_sent", "sender", "recipient", "message", "message_round"])
    wandb_responses = wandb.Table(dataframe=data["responses"], log_mode="MUTABLE")
    logger.debug(f"Initialized data logging directory: {data_dir}")

    # Initialize game
    game: Game = initialize_game(game_config["game"]["map_name"], game_config["game"]["max_message_rounds"])
    logger.success(f"Initialized diplomacy game: {game}")

    # Initialize players
    players: Dict[str, agents.DiplomacyAgent] = initialize_players(game, game_config)
    logger.success(f"Players initialized: \n{pprint.pformat(players)}")

    # Run main loop
    with Progress() as progress:
        # Setup progress bar for tracking phases
        max_years = game_config["game"]["max_years"]
        progress_phases = progress.add_task("[red]🔄️ Phases...", total=max_years * 3)

        # Main loop for the game
        while not game.is_game_done:
            current_phase = game.get_current_phase()
            logger.info(f"🕰️  Beginning phase {current_phase}")

            # Start new phase
            for power_name, agent in players.items():
                agent.start_phase()

            # Run negotiation phase
            try:
                run_negotiation_phase(
                    game=game,
                    players=players,
                    game_config=game_config,
                    progress=progress,
                    log_dict=data,
                    wandb_log_messages=wandb_responses,
                )
            except Exception as e:
                logger.error(f"💥 Error during negotiation phase: \n{e}\n\n{traceback.format_exc()}")
                # TODO: Log the error (local and wandb), terminate game
                break

            # Run decision phase
            try:
                run_movement_phase(game, players)
            except Exception as e:
                logger.error(f"💥 Error during negotiation phase: \n{e}\n\n{traceback.format_exc()}")
                # TODO: Log the error (local and wandb), terminate game
                break

            # End phase
            for power_name, agent in players.items():
                agent.end_phase()

            # # Generate messages to be sent by each player in this phase.
            # #    Each player participates in message round.
            # #    There are two conditions under which player is not allowed to exchange messages:
            # #    1. Game is in "R" phase (retreats), or
            # #    2. There's an external sanction imposed on that player.
            # #    In first case, Agent class will return empty messages.
            # #    In second case, the main loop will skip calling agent's generate_message() function.
            # num_message_rounds = (
            #     1
            #     if game.phase_type == "R" or game.no_press
            #     else game_config["game"]["max_message_rounds"]
            # )
            # progress_message_rounds = progress.add_task(
            #     description="[blue]🙊 Messages",
            #     total=num_message_rounds * 7
            # )
            # for message_round in range(1, num_message_rounds + 1):
            #     # For each player, decide messages to send.
            #     for power_name, agent in players.items():
            #         # Step the player with entire history (i.e., game instance) to generate messages and orders
            #         messages: dict = agent.generate_messages(game)
            #
            #         # Execute send_message in game
            #         for recipient, message in messages.items():
            #             msg = Message(
            #                 sender=power_name,
            #                 recipient=recipient,
            #                 message=message,
            #                 phase=game.get_current_phase(),
            #             )
            #             game.add_message(msg)
            #
            #         #  Update W&B player-level logs (compute metrics)
            #         if not game_config["wandb"]["disable"]:
            #             update_wandb_player_logs(game, power_name, agent, messages)
            #
            #         # Update internal logs
            #         data = update_internal_player_logs(data, game, power_name, agent, messages)
            #
            #         # Update progress bar
            #         progress.update(progress_message_rounds, advance=1)
            #
            # # Update progress bar
            # progress.remove_task(progress_message_rounds)

            # Step game with game.process()
            game.process()
            if int(game.phase.split()[1]) - 1900 > game_config["game"]["max_years"]:
                game.finish()

            # Update W&B game-level logs
            if not game_config["wandb"]["disable"]:
                update_wandb_game_logs(game, players)

            # Update internal logs
            # data = update_internal_game_logs(data, game, players)

            # Update progress bar
            progress.update(progress_phases, advance=1)


def run_negotiation_phase(game, players, game_config, progress, log_dict, wandb_log_messages):
    def update_message_logs(msg_round):
        log_dict["responses"].loc[len(log_dict["responses"])] = [
            msg.phase,
            msg.time_sent,
            msg.sender,
            msg.recipient,
            msg.message,
            msg_round,
        ]
        wandb_log_messages.add_data(
            str(msg.phase),
            str(msg.time_sent),
            str(msg.sender),
            str(msg.recipient),
            str(msg.message),
            str(msg_round),
        )

    num_message_rounds = game_config["game"]["max_message_rounds"]
    progress_message_rounds = progress.add_task(
        description="[blue]🙊 Messages",
        total=num_message_rounds * 7
    )
    try:
        for message_round in range(1, num_message_rounds + 1):
            # For each player, decide messages to send.
            for power_name, agent in players.items():
                # Step the player with entire history (i.e., game instance) to generate messages and orders
                messages: dict = agent.generate_messages()

                # Execute send_message in game
                for recipient, message in messages.items():
                    msg = Message(
                        sender=power_name,
                        recipient=recipient,
                        message=message,
                        phase=game.get_current_phase(),
                    )
                    game.add_message(msg)

                    # Log message
                    update_message_logs(msg_round=message_round)

                # Update progress bar
                progress.update(progress_message_rounds, advance=1)
    except Exception as e:
        logger.exception(f"💥 Error during negotiation phase: \n{e}\n\n{traceback.format_exc()}")
    finally:
        if not game_config["wandb"]["disable"]:
            wandb.log({"responses": wandb_log_messages})
        # Update progress bar
        progress.remove_task(progress_message_rounds)


def run_movement_phase(game, players):
    orders = dict()
    for power_name, agent in players.items():
        # Step the player with entire history (i.e., game instance) to generate orders
        try:
            orders[power_name] = agent.generate_orders()
        except Exception as e:
            logger.error(f"💥 Error during movement phase for {power_name}: {e}")
            raise e

    for power_name, order in orders.items():
        game.set_orders(power_name, order)


def update_wandb_player_logs(game, power_name, agent, messages):
    pass


def update_internal_player_logs(data, game, power_name, agent, messages):
    pass


def update_wandb_game_logs(game, players):
    pass


def update_internal_game_logs(data, game, players):
    pass


def init_data_log_directory(run_name: str, prefix: Path = Path() / "out", overwrite: bool = False):
    """
    :return: (pathlib.Path) Path to the directory where data will be logged.
    """
    # Format the directory name
    current_time = datetime.now()
    dir_name = f"{current_time.strftime('%Y_%m_%d')}_{current_time.strftime('%H_%M')}_{run_name}"

    # Determine the full path
    if prefix:
        full_path = prefix / dir_name
    else:
        full_path = Path(dir_name)

    # Check if the directory exists
    if full_path.exists():
        if not overwrite:
            raise FileExistsError(f"Directory '{full_path}' already exists and overwrite is set to False.")
        else:
            # Delete the existing directory
            full_path.rmdir()

    # Create the new directory
    full_path.mkdir(parents=True, exist_ok=True)

    return Path()


def initialize_game(map_name: str, max_message_rounds: int) -> Game:
    """
    :param game_config:
    :return:
    """
    game: Game = Game(map_name=map_name)
    if max_message_rounds <= 0:
        game.add_rule("NO_PRESS")
    else:
        game.remove_rule("NO_PRESS")
    return game


def initialize_players(game, game_config):
    """
    Initialize agents for each power in the game based on the game configuration.

    :param game: (Game) Diplomacy game instance.
    :param game_config: (dict) Game configuration dictionary containing player configurations.
    :return: Dict[str, Agent] Dictionary mapping power names to initialized agent instances.
    """
    assert game_config["players"].keys() == game.powers.keys(), \
        f"Game config has mismatched powers: {game_config['players'].keys()=} & {game.powers.keys()=}"

    power_name_to_agent = dict()
    for name, params in game_config["players"].items():
        agent_cls_name = params["agent_class"]
        agent_cls = agents.get_class(agent_cls_name)
        power_name_to_agent[name] = agent_cls(game=game, pow_name=name, **params["agent_params"])

    return power_name_to_agent


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulate a game of Diplomacy with the given parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level", dest="log_level", default="INFO", help="🪵 Logging level."
    )
    parser.add_argument(
        "--map_name",
        dest="map_name",
        default="standard_welfare",
        help="🗺️ Map name which switches between rulesets.",
    )
    parser.add_argument(
        "--run_name",
        dest="run_name",
        default=None,
        help="🏗️ Weights & Biases run name.",
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        default="games",
        help="📁Folder to save the game to.",
    )
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        help="💾Save the game to disk (uses W&B run ID & name).",
    )
    parser.add_argument("--seed", dest="seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--entity",
        dest="entity",
        default=None,
        help="👤Weights & Biases entity name (defaults to your username). Note you can also use the WANDB_ENTITY env var.",
    )
    parser.add_argument(
        "--project",
        dest="project",
        default="temporary",
        help="🏗️ Weights & Biases project name.",
    )
    parser.add_argument(
        "--disable_wandb",
        dest="disable_wandb",
        action="store_true",
        help="🚫Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--max_years",
        dest="max_years",
        type=int,
        default=10,
        help="🗓️ Ends the game after this many years (~3x as many turns).",
    )
    parser.add_argument(
        "--early_stop_max_years",
        dest="early_stop_max_years",
        type=int,
        default=0,
        help="⏱️ Early stop while telling the models the game lasts --max_years long. No effect if 0.",
    )
    parser.add_argument(
        "--max_message_rounds",
        dest="max_message_rounds",
        type=int,
        default=3,
        help="📨Max rounds of messaging per turn. 0 is no-press/gunboat diplomacy.",
    )
    parser.add_argument(
        "--agent_model",
        dest="agent_model",
        default="gpt-4o-mini",
        help="🤖Model name to use for the agent. Can be an OpenAI Chat model, 'random', or 'manual' (see --manual_orders_path).",
    )
    parser.add_argument(
        "--manual_orders_path",
        dest="manual_orders_path",
        type=str,
        help="📝YAML file path to manually enter orders for all powers (see ./manual_orders).",
    )
    parser.add_argument(
        "--summarizer_model",
        dest="summarizer_model",
        default="gpt-4o-mini",
        help="✍️ Model name to use for the message summarizer. Can be an OpenAI Chat model or 'passthrough'.",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=1.0,
        help="🌡️ Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        dest="top_p",
        type=float,
        default=0.9,
        help="⚛️ Top-p for nucleus sampling.",
    )
    parser.add_argument(
        "--max_completion_errors",
        dest="max_completion_errors",
        type=int,
        default=30,
        help="🚫Max number of completion errors before killing the run.",
    )
    # parser.add_argument(
    #     "--prompt_ablations",
    #     type=str,
    #     default="",
    #     help=f"🧪Ablations to apply to the agent prompts. Separate multiple ablations by commas. All available values are {', '.join([elem.name.lower() for elem in PromptAblation])}",
    # )
    parser.add_argument(
        "--exploiter_prompt",
        dest="exploiter_prompt",
        type=str,
        default="",
        help="🤫If specified along with --exploiter_powers, adds this into the system prompt of each exploiter power. Useful for asymmetrically conditioning the agents, e.g. for exploitability experiments. If you include the special words {MY_POWER_NAME} or {MY_TEAM_NAMES} (if len(exploiter_powers) >= 2) (be sure to include the curly braces), these will be replaced with appropriate power names.",
    )
    parser.add_argument(
        "--exploiter_powers",
        dest="exploiter_powers",
        type=str,
        default="",
        help="😈Comma-separated list of case-insensitive power names for a exploiter. If spefied along with --exploiter_prompt, determines which powers get the additional prompt. Useful for asymmetrically conditioning the agents, e.g. for exploitability experiments.",
    )
    parser.add_argument(
        "--exploiter_model",
        dest="exploiter_model",
        type=str,
        default="gpt-4o-mini",
        help="🦾 Separate model name (see --agent_model) to use for the exploiter (see --exploiter_prompt) if desired. If omitted, uses the --agent_model.",
    )
    parser.add_argument(
        "--local_llm_path",
        dest="local_llm_path",
        type=str,
        default=None,
        help="📁Path to a local LLM model to use instead of downloading from HuggingFace.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="auto",
        help="📱Device to use for the model. Can be 'cpu', 'cuda', or 'auto'.",
    )
    parser.add_argument(
        "--quantization",
        dest="quantization",
        type=int,
        default=None,
        help="📉Quantization level to use for the model. If None, no quantization is used. If 8, uses 8-bit quantization. If 4, uses 4-bit quantization.",
    )
    parser.add_argument(
        "--fourbit_compute_dtype",
        dest="fourbit_compute_dtype",
        type=int,
        default=32,
        help="📉Compute dtype to use for 4-bit quantization. If 32, uses 32-bit compute dtype. If 16, uses 16-bit compute dtype.",
    )
    parser.add_argument(
        "--disable_completion_preface",
        dest="disable_completion_preface",
        action="store_true",
        help="⏭️ Don't use the completion preface (which helps agents comply with the json format).",
    )
    )
    parser.add_argument(
        "--no_press_powers",
        dest="no_press_powers",
        type=str,
        default="",
        help="🤐Comma-separated list of case-insensitive power names to run standard no-press policy.",
    )
    parser.add_argument(
        "--no_press_policy",
        dest="no_press_policy",
        type=int,
        default=0,
        help="🤐Policy to use for no-press powers. Provide an integer to select a policy from no_press_policies.policy_map.",
    )
    parser.add_argument(
        "--super_exploiter_powers",
        dest="super_exploiter_powers",
        type=str,
        default="",
        help="🤐Comma-separated list of case-insensitive powers to use hybrid LM + RL exploiter policy.",
    )
    parser.add_argument(
        "--unit_threshold",
        dest="unit_threshold",
        type=int,
        default=10,
        help="🤐Number of enemy units on the board below which a super exploiter switches from the LLMAgent policy to the RL policy.",
    )
    parser.add_argument(
        "--center_threshold",
        dest="center_threshold",
        type=int,
        default=10,
        help="🤐Number of centers a super exploiter acquires before it switches back to the LLMAgent policy.",
    )

    args = parser.parse_args()
    # if args.save is False:
    #     if "y" in input("Do you want to save the game? (yes/no)").lower():
    #         args.save = True
    return vars(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Manually write it with the logger so it doesn't get hidden in wandb logs
        tqdm.write("\n\n\n")  # Add some spacing
        exception_trace = "".join(
            traceback.TracebackException.from_exception(exc).format()
        )
        # utils.log_error(
        #     logger,
        #     f"💀 FATAL EXCEPTION: {exception_trace}",
        # )
        # wandb.log(
        #     {
        #         "fatal_exception_trace": wandb.Table(
        #             columns=["trace"], data=[[exception_trace]]
        #         )
        #     }
        # )
        tqdm.write("\n\n\n")  # Add some spacing
        raise exc
