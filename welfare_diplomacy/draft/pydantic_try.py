from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    model_name="ai/mistral",
    provider=OpenAIProvider(
        # base_url="http://model-runner.docker.internal/engines/v1",
        base_url="http://localhost:12434/engines/v1",
        api_key="docker",
    ),
)

agent = Agent(
    model=model,
    instructions=["Reply in one sentence."],
)

response = agent.run_sync("Tell me a joke.")
print(response.output)

response = agent.run_sync("Explain?", message_history=response.new_messages())
print(response.output)
