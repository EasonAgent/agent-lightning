import logging
from typing import Any, cast, Dict

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings

from agentlightning import (
    LLM,
    Trainer,
    LitAgent,
    Rollout,
    NamedResources,
    configure_logger,
)

configure_logger(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LitBaseAgent(LitAgent[Any]):
    def __init__(self, trained_agents: str | None = None) -> None:
        super().__init__(trained_agents=trained_agents)

    async def rollout_async(self, task: Dict[str, Any], resources: NamedResources, rollout: Rollout) -> float | None:
        rollout_id = rollout.rollout_id
        logger.debug(f"Starting rollout {rollout_id} for task: {task}")

        llm: LLM = cast(LLM, resources.get("main_llm"))
        agent = Agent(
            model=LitellmModel(model="hosted_vllm/" + llm.model, base_url=llm.endpoint),
            model_settings=ModelSettings(
                max_tokens=4096,
                temperature=0.7,
            ),
            name="Assistant",
            instructions="You are a helpful assistant.",
        )
        result = await Runner.run(agent, task["question"])
        logger.debug(f"Completed rollout {rollout_id} with result: {result.final_output}")
        # TODO: calc reward
        return 0
