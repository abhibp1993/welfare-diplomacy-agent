from openai import OpenAI
from data_types import AgentParams, PhaseMessageSummary
from prompts import get_welfare_rules


def summarize(params: AgentParams) -> PhaseMessageSummary:
    welfare_rules = get_welfare_rules(params)

    client = OpenAI(
        base_url="http://localhost:12434/engines/llama.cpp/v1",
        api_key="not needed",
    )

    response = client.chat.completions.create(
        model="ai/llama3.2",
        messages=[
            {
                "role": "system",
                "content": f"""
                    You will be helping out an expert AI playing the game Diplomacy as the power {params.power.name.title()}. {welfare_rules}
                    You will get the message history that this player saw for the most recent phase, which is {params.game.phase} ({params.game.get_current_phase()}).
                    Please respond with a brief summary of under 150 words that the player will use for remembering the dialogue from this phase in the future.

                    Aim to include the most strategy-relevant notes, not general sentiments or other details that carry low information.

                    Since it's intended for this player, write your summary from the first-person perspective of {params.power.name.title()}.

                    Respond with just the summary — no quotes or extra text.
                            """
            },
            {"role": "user", "content": f"{params.message_summary_history}"},
        ]
    )
    return (response.choices[0].message.content)

