"""
Example Python configuration file for DAB Evaluation SDK
"""

import os

# LLM Configuration
llm_config = {
    "model": os.environ.get("LLM_MODEL", "doubao-seed-1-6-251015"),
    "base_url": os.environ.get("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
    "api_key": os.environ.get("ARK_API_KEY", ""),
    "temperature": 0.3,
    "max_tokens": 2000
}

# Agent Configuration
agent_config = {
    "url": os.environ.get("AGENT_URL", "http://localhost:8002"),
    "capabilities": ["web_retrieval", "web_onchain_retrieval", "onchain_retrieval"],
    "timeout": 30,
    "close_endpoint": os.environ.get("AGENT_CLOSE_ENDPOINT", "http://localhost:8002/close")
}

# Dataset Configuration
dataset_config = {
    "type": "csv",
    "path": "data/benchmark.csv",
    "abbr": "benchmark",
    "reader_cfg": {
        "input_columns": ["question"],
        "output_column": "answer"
    }
}

# Evaluator Configuration
evaluator_config = {
    "type": "hybrid",
    "rule_based_weight": 0.3,
    "llm_based_weight": 0.7,
    "llm_evaluation_threshold": 0.5,
    "use_llm_evaluation": True
}

# Runner Configuration
runner_config = {
    "type": "local",
    "max_workers": 4,
    "max_workers_per_gpu": 1,
    "timeout": 300
}

# Complete Configuration
config = {
    "llm_config": llm_config,
    "agent_config": agent_config,
    "dataset_config": dataset_config,
    "evaluator_config": evaluator_config,
    "runner_config": runner_config,
    "work_dir": "output",
    "max_tasks": None  # Set to a number to limit tasks, None for all
}

