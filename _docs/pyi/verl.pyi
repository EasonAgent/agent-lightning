# ===================================================================================================================
# agentlightning/verl/entrypoint.py
@hydra.main(config_path="pkg://agentlightning/verl", config_name="config", version_base=None)
def main(config):
    run_ppo(config, train_dataset=None, val_dataset=None, store=None, llm_proxy=None, adapter=None)

def run_ppo(
    config: Any,
    train_dataset: Dataset[Any] | None,
    val_dataset: Dataset[Any] | None,
    store: LightningStore | None,
    llm_proxy: LLMProxy | None,
    adapter: TraceAdapter[Any] | None,
) -> None: