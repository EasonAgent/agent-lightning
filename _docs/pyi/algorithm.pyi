from typing import TYPE_CHECKING, Any, Optional, Awaitable, Union

from .verl import run_ppo
from .store import LightningStore

def VERL(*args: Any, **kwargs: Any) -> VERL:
    return VERL(*args, **kwargs)


# ===================================================================================================================
# agentlightning/algorithm/base.py
class Algorithm:
    """Algorithm is the strategy, or tuner to train the agent."""
    def run(
        self,
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
    ) -> Union[None, Awaitable[None]]:
        """Subclasses should implement this method to implement the algorithm.

        Args:
            train_dataset: The dataset to train on. Not all algorithms require a training dataset.
            val_dataset: The dataset to validate on. Not all algorithms require a validation dataset.

        Returns:
            Algorithm should refrain from returning anything. It should just run the algorithm.
        """

    def get_store(self) -> LightningStore:
        """
        Retrieve the store for this algorithm to communicate with the runners.
        """
        return self._store
    def get_llm_proxy(self) -> Optional[LLMProxy]:
        """
        Retrieve the configured LLM proxy instance, if one has been set.
        """
        return self._llm_proxy_ref()
    def get_adapter(self) -> TraceAdapter[Any]:
        """
        Retrieve the adapter for this algorithm to communicate with the runners.
        """
        return self._adapter_ref()

# ===================================================================================================================
# agentlightning/algorithm/verl/interface.py
from hydra import compose, initialize
from omegaconf import OmegaConf

class VERL(Algorithm):
    """VERL-powered algorithm that delegates training to the VERL PPO runner.

    !!! warning
        Advanced customisation currently requires copying the VERL source and
        modifying it directly. Native hooks for overriding training behaviour
        will land in a future release.

    Args:
        config: Dictionary mirroring the overrides passed to the VERL CLI. The
            overrides are merged with VERL's packaged defaults via Hydra before
            launching training.

    Examples:
        ```python
        from agentlightning.algorithm.verl import VERL

        algorithm = VERL(
            config={
                "algorithm": {
                    "adv_estimator": "grpo",
                    "use_kl_in_reward": False,
                },
                "data": {
                    "train_batch_size": 32,
                    "max_prompt_length": 4096,
                    "max_response_length": 2048,
                },
                "actor_rollout_ref": {
                    "rollout": {
                        "tensor_model_parallel_size": 1,
                        "n": 4,
                        "log_prob_micro_batch_size_per_gpu": 4,
                        "multi_turn": {"format": "hermes"},
                        "name": "vllm",
                        "gpu_memory_utilization": 0.6,
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
                        "path": "Qwen/Qwen2.5-1.5B-Instruct",
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
                    "experiment_name": "calc_x",
                    "nnodes": 1,
                    "save_freq": 64,
                    "test_freq": 32,
                    "total_epochs": 2,
                },
            }
        )
        trainer.fit(algorithm, train_dataset=my_train_dataset)
        ```
    """
    def __init__(self, config: dict[str, Any]):
        # Compose the base config exactly like your decorator:
        with initialize(version_base=None, config_path="pkg://agentlightning/verl"):
            base_cfg = compose(config_name="config")
        # Merge your dict overrides
        override_conf = OmegaConf.create(config)
        self.config = OmegaConf.merge(base_cfg, override_conf)

    def run(
        self,
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
    ) -> None:
        """Launch the VERL PPO entrypoint with the configured runtime context.

        Args:
            train_dataset: Optional dataset forwarded to VERL for training.
            val_dataset: Optional dataset forwarded to VERL for evaluation.

        Raises:
            ValueError: If required dependencies such as the store, LLM proxy, or
                adapter have been garbage-collected when using the V1 execution
                mode.
        """
        try:
            store = self.get_store()
            llm_proxy = self.get_llm_proxy()
            adapter = self.get_adapter()
            run_ppo(
                self.config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                store=store,
                llm_proxy=llm_proxy,
                adapter=adapter,
            )
