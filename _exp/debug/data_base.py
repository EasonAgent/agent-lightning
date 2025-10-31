from typing import Any, Dict

from utu.utils import LLMOutputParser
from utu.config import ConfigLoader
from utu.agents import LLMAgent

# -----------------------------------------------------------------------------------------------------------
TRAIN_DATA = VAL_DATA = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
    },
    {
        "question": "Who wrote 'To Kill a Mockingbird'?",
        "answer": "Harper Lee",
    }
] * 20

# -----------------------------------------------------------------------------------------------------------
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
        "n_gpus_per_node": 2,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "debug",  # set exp name
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 2,
    },
}


# -----------------------------------------------------------------------------------------------------------
LLM = LLMAgent(ConfigLoader.load_model_config("base"))
TEMPLATE = """Please grade the following answer based on the question.

question: {question}
ground truth answer: {gt}
student's answer: {pred}

Please give a score from 0 to 1, where 1 means the model answer is completely correct, and 0 means it is completely incorrect.
Only output the score as a float number, e.g. 0.75, 1.0, 0.0
"""

async def evaluate(question: str, gt: str, pred: str) -> float:
    prompt = TEMPLATE.format(question=question, gt=gt, pred=pred)
    result = await LLM.run(prompt)
    try:
        score = LLMOutputParser.extract_float_number(result.final_output.strip()) or 0.0
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.0

if __name__ == '__main__':
    import asyncio

    async def test():
        res = await evaluate(
            question="What is the capital of France?",
            gt="Paris",
            pred="The capital of France is Paris.",
        )
        print("Evaluation score:", res)
    asyncio.run(test())
