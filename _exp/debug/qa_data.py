import pandas as pd
from copy import deepcopy
from typing import Dict, Any

# -----------------------------------------------------------------------------------------------------------
train_asearcher_base="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-train-data/base/ASearcher-Base-35k_train.parquet"
train_asearcher_lrm="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-train-data/lrm/ASearcher-LRM-35k_train.parquet"

## single hop qa
##  Natural Questions [15], TriviaQA [12] and PopQA
test_nq="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/NQ_rand1000_test.parquet"
test_triviaqa="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/TriviaQA_rand1000_test.parquet"
test_popqa="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/PopQA_rand1000_test.parquet"

## multihop qa
# HotpotQA [44], 2WikiMultiHopQA [10], MuSiQue [36], and Bamboogle
test_hotpotqa="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/HotpotQA_rand1000_test.parquet"
test_2wikimultihopqa="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/2WikiMultihopQA_rand1000_test.parquet"
test_musique="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/Musique_rand1000_test.parquet"
test_bamboogle="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/Bamboogle_test.parquet"

## deep search
test_frames="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/frames_test.parquet"
test_gaia="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/GAIA_test.parquet"
test_xbench="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/xbench-deepsearch_test.parquet"

## single hop qa quick100
test_nq_rand100="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/NQ_rand1000_rand100_test.parquet"
test_triviaqa_rand100="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/TriviaQA_rand1000_rand100_test.parquet"
test_popqa_rand100="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/PopQA_rand1000_rand100_test.parquet"

## multihop qa quick100
# HotpotQA [44], 2WikiMultiHopQA [10], MuSiQue [36], and Bamboogle
test_hotpotqa_rand100="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/HotpotQA_rand1000_rand100_test.parquet"
test_2wikimultihopqa_rand100="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/2WikiMultihopQA_rand1000_rand100_test.parquet"
test_musique_rand100="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/Musique_rand1000_rand100_test.parquet"
test_bamboogle_rand100="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/Bamboogle_rand100_test.parquet"

## deep search quick100
test_frames_rand100="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/frames_rand100_test.parquet"
test_gaia_rand100="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/GAIA_rand100_test.parquet"
test_xbench_rand100="/cfs_turbo/yuleiqin/Research/agent-lightning/datasets/asearcher_data/ASearcher-test-data/xbench-deepsearch_rand100_test.parquet"


TRAIN_FILES=[train_asearcher_base]
TEST_FILES=[test_nq, test_triviaqa, test_popqa, test_hotpotqa, test_2wikimultihopqa, test_musique, test_bamboogle]
TEST_FILES_FAST=[test_nq_rand100, test_triviaqa_rand100, test_popqa_rand100, test_hotpotqa_rand100, test_2wikimultihopqa_rand100, test_musique_rand100, test_bamboogle_rand100]

TRAIN_DATA, VAL_DATA = [], []
for train_file in TRAIN_FILES:
    TRAIN_DATA += pd.read_parquet(train_file).to_dict(orient="records")
for val_file in TEST_FILES_FAST:
    VAL_DATA += pd.read_parquet(val_file).to_dict(orient="records")


# -----------------------------------------------------------------------------------------------------------
def config_train_qwen_7b() -> dict[str, Any]:
    """A configuration for training with Qwen-2.5."""
    from base_data import RL_TRAINING_CONFIG
    
    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["model"]["path"] = "/cfs_turbo/yuleiqin/models/Qwen2.5-7B-Instruct_Qwen"
    config["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = 2

    config["trainer"]["experiment_name"] = "debug/youtu-agent-7b"
    return config

RL_TRAINING_CONFIG = config_train_qwen_7b()