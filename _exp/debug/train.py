from typing import Any, Dict, Optional

import agentlightning as agl

from data import TRAIN_DATA
from base_agent import LitBaseAgent


RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/train_spider.parquet",
        "val_files": "data/test_dev_500.parquet",
        "train_batch_size": 32,
        "max_prompt_length": 4096,
        "max_response_length": 2048,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.8,
        },
        "actor": {
            "ppo_mini_batch_size": 32,
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "spider",
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 2,
    },
}



def main(active_agent: Optional[str] = None) -> None:
    """Train the SQL agent with the given configuration."""

    config = RL_TRAINING_CONFIG
    agent = LitBaseAgent()
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=2, algorithm=algorithm, adapter={"agent_match": active_agent})
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore

    train_data = TRAIN_DATA
    val_data = TRAIN_DATA
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore

if __name__ == "__main__":
    main()