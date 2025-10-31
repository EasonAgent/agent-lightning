from typing import Any, Dict, Optional

import agentlightning as agl

# setting 1: base
# from base_data import TRAIN_DATA, VAL_DATA, RL_TRAINING_CONFIG
# # from base_openai_agent import LitBaseAgent
# from base_utu_agent import LitBaseAgent

# setting 2: qa by @yulei
from qa_agent import LitBaseAgent
from qa_data import TRAIN_DATA, VAL_DATA, RL_TRAINING_CONFIG


def main(active_agent: Optional[str] = None) -> None:
    """Train the SQL agent with the given configuration."""

    config = RL_TRAINING_CONFIG
    agent = LitBaseAgent()
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=2, algorithm=algorithm, adapter={"agent_match": active_agent})
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore

    trainer.fit(agent, train_dataset=TRAIN_DATA, val_dataset=VAL_DATA)  # type: ignore

if __name__ == "__main__":
    main()
