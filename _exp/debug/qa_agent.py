import logging
from typing import Any, cast, Dict

from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings

from utu.config import ConfigLoader
from utu.agents import SimpleAgent, get_agent


from agentlightning import (
    LLM,
    Trainer,
    LitAgent,
    Rollout,
    NamedResources,
    configure_logger,
)

from utils import evaluate

configure_logger(level=logging.INFO)
logger = logging.getLogger(__name__)


class LitBaseAgent(LitAgent[Any]):
    def __init__(self, trained_agents: str | None = None) -> None:
        super().__init__(trained_agents=trained_agents)

    async def rollout_async(self, task: Dict[str, Any], resources: NamedResources, rollout: Rollout) -> float | None:
        """ task (data) structure: {question, data_source, golden_answers} """
        rollout_id = rollout.rollout_id
        print(f"Starting rollout {rollout_id} for task: {task}")

        llm: LLM = cast(LLM, resources.get("main_llm"))

        config = ConfigLoader.load_agent_config("simple/base")
        # overwrite
        config.model.model_provider.model = llm.model
        config.model.model_provider.base_url = llm.endpoint
        config.model.model_provider.api_key = llm.api_key or "xxx"
        config.max_turns = llm.sampling_parameters.get("max_turns", 5)
        config.model.model_settings.temperature = llm.sampling_parameters.get("temperature", 1.0)
        config.model.model_settings.top_p = llm.sampling_parameters.get("top_p", 0.6)
        config.model.termination_max_tokens = 32000
        agent = get_agent(config=config)

        result = await agent.run(task["question"])
        print(f"Completed rollout {rollout_id} with result: {result.final_output}")
        
        # calc reward
        reward = await evaluate(
            question=task["question"],
            gt=task["golden_answers"],
            pred=result.final_output,
        )
        return reward
