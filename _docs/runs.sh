# -----------------------------------------------------------------------------
# install in tione
git clone https://github.com/EasonAgent/agent-lightning
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# /root/.local/bin/uv

uv venv
source .venv/bin/activate

/root/.local/bin/uv sync --extra verl \
    --group torch-gpu-stable

# or, mamually install 
/root/.local/bin/uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu122
/root/.local/bin/uv pip install flash-attn --no-build-isolation
/root/.local/bin/uv pip install vllm==0.10.2
/root/.local/bin/uv pip install verl==0.5.0

# install in macos (for debug)
uv venv
source .venv/bin/activate

uv pip install uvicorn fastapi opentelemetry-sdk flask agentops setproctitle openai-agents
uv pip install hydra-core  # https://hydra.cc/docs/intro/

# restart ray
bash scripts/restart_ray.sh
# env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0

# -----------------------------------------------------------------------------
# spider | sql
# data
cd examples/spider
pip install gdown  # included in the 'experiment' optional dependency
gdown --fuzzy https://drive.google.com/file/d/1oi9J1jZP9TyM35L85CL3qeGWl2jqlnL6/view
unzip -q spider-data.zip -d data
rm spider-data.zip

# deps
uv pip install pandas termcolor langchain langchain_community langgraph sqlparse

# train
python train_sql_agent.py fast
python train_sql_agent.py qwen   # Default Qwen-2.5-Coder-1.5B run
python train_sql_agent.py llama  # LLaMA-3.2-1B with llama3_json tool parser | need HF_TOKEN

# examples/spider/train_sql_agent.py | config same as:
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-1.5B-Instruct \
    trainer.n_gpus_per_node=1 \
    trainer.val_before_train=True \
    trainer.test_freq=32 \
    trainer.save_freq=64 \
    trainer.total_epochs=2


# -----------------------------------------------------------------------------
# youtu-agent
git clone https://github.com/Tencent/Youtu-agent
uv pip install -e youtu-agent
# ...env