# DAB Evaluation SDK


**DAB (Decentralized Agent Benchmark)** is a comprehensive evaluation framework designed specifically for **Web3 agents**. As the decentralized ecosystem continues to evolve, autonomous agents are playing an increasingly critical role in interacting with blockchain networks, DeFi protocols, NFT marketplaces, and other Web3 infrastructure.

DAB provides standardized benchmarks and evaluation tools to assess the capabilities of Web3 agents across key scenarios:

- **Web Retrieval**: Agents that fetch and process information from traditional web sources to answer questions about Web3 events, protocols, and market data
- **On-chain Retrieval**: Agents that query blockchain data directly, extracting transaction histories, smart contract states, token balances, and on-chain analytics
- **Web + On-chain Retrieval**: Hybrid agents that combine web and blockchain data sources to provide comprehensive answers requiring both off-chain context and on-chain verification

These evaluation scenarios reflect real-world use cases where Web3 agents must navigate complex information landscapes, verify data across multiple sources, and provide accurate, actionable insights for users interacting with decentralized systems.

## Overview

DAB Evaluation SDK is a lightweight Python framework for evaluating Web3 agents, designed to assess agent capabilities and performance across various task categories with intelligent evaluation strategies.

The SDK provides a flexible, configuration-driven approach to agent evaluation. Whether you're testing a single agent on a specific question or running comprehensive benchmarks across entire datasets, the SDK adapts to your needs with automatic method selection, multi-dimensional scoring, and detailed analytics.

The framework intelligently combines rule-based precision with LLM-based understanding, ensuring fair and accurate assessments even when reference answers aren't available. Built with modularity in mind, it separates evaluation logic, task execution, and result summarization for maximum flexibility and extensibility.

## Key Capabilities

**Smart Evaluation Selection** - The SDK automatically chooses the most appropriate evaluation method based on your task category. Web retrieval tasks benefit from precise rule-based matching, while complex reasoning tasks leverage LLM understanding. For tasks requiring both precision and comprehension, a hybrid approach seamlessly combines both methods.

**Multi-Dimensional Assessment** - Beyond simple correctness scores, the framework evaluates responses across multiple dimensions including factual accuracy, technical expertise, completeness, and relevance. This gives you a comprehensive view of agent performance, not just a binary pass/fail.

**Adaptive Scoring** - When reference answers are available, the system performs detailed format normalization, key fact extraction, and semantic matching. When they're not, it falls back to intelligent semantic and relevance scoring, ensuring concise but accurate responses aren't unfairly penalized.

**Comprehensive Analytics** - Results are automatically aggregated with category-based statistics, success rate analysis, and score distributions. Export to JSON or CSV formats for further analysis or reporting.

**Configuration-Driven** - Define your evaluation setup through clean configuration files (JSON or Python), making it easy to reproduce experiments, share evaluation setups, and manage different testing scenarios.

## Quick Start

### Basic Usage

The SDK uses a configuration-driven approach. Start by creating a configuration:

```python
import asyncio
import os
from dab_eval import (
    DABEvaluator,
    EvaluationConfig,
    LLMConfig,
    AgentConfig,
    DatasetConfig,
    AgentMetadata,
    TaskCategory,
    load_config
)

async def main():
    # Option 1: Load from config file
    config = load_config("configs/example_config.json")
    
    # Option 2: Create programmatically
    config = EvaluationConfig(
        llm_config=LLMConfig(
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            temperature=0.3,
            max_tokens=2000
        ),
        agent_config=AgentConfig(
            url="http://localhost:8002",
            capabilities=[TaskCategory.WEB_RETRIEVAL, TaskCategory.ONCHAIN_RETRIEVAL],
            timeout=30
        ),
        dataset_config=DatasetConfig(),
        work_dir="output"
    )
    
    # Create evaluator
    evaluator = DABEvaluator(config)
    
    # Define your agent
    agent = AgentMetadata(
        url="http://localhost:8002",
        capabilities=[TaskCategory.WEB_RETRIEVAL],
        timeout=30,
        close_endpoint="http://localhost:8002/close"
    )
    
    # Evaluate a single question
    result = await evaluator.evaluate_agent(
        question="What is the date (UTC) when the US SEC approved Bitcoin spot ETF?",
        agent_metadata=agent,
        category=TaskCategory.WEB_RETRIEVAL,
        expected_answer="2024/1/10"
    )
    
    print(f"Score: {result.evaluation_score:.2f}")
    print(f"Response: {result.agent_response}")
    print(f"Reasoning: {result.evaluation_reasoning}")

asyncio.run(main())
```

### Dataset Evaluation

Evaluate your agent against a complete benchmark dataset:

```python
async def evaluate_dataset():
    config = load_config("configs/example_config.json")
    evaluator = DABEvaluator(config)
    
    agent = AgentMetadata(
        url="http://localhost:8002",
        capabilities=[TaskCategory.WEB_RETRIEVAL, TaskCategory.ONCHAIN_RETRIEVAL],
        timeout=30
    )
    
    # Evaluate against benchmark
    results = await evaluator.evaluate_agent_with_dataset(
        agent_metadata=agent,
        dataset_path="data/benchmark.csv",
        max_tasks=10  # Optional: limit number of tasks
    )
    
    # Export comprehensive results
    summary = evaluator.export_results("json")
    
    print(f"Total Tasks: {summary['overall']['total_tasks']}")
    print(f"Success Rate: {summary['overall']['success_rate']:.2%}")
    print(f"Average Score: {summary['overall']['average_score']:.3f}")
    
    # Results are automatically saved to output/ directory
    # - evaluation_results.json: Detailed results
    # - evaluation_summary.json: Aggregated statistics

asyncio.run(evaluate_dataset())
```

## Configuration

The SDK uses a **separated configuration architecture** that distinguishes between business and infrastructure configurations:

- **Business Config**: LLM, Agent, Dataset, Evaluator settings
- **Infrastructure Config**: Runner, Storage, and system-level settings

This separation allows you to reuse infrastructure configurations across different evaluation scenarios.

### Configuration Structure

The SDK supports both JSON and Python configuration files. Here's a complete example:

**configs/example_config.json:**
```json
{
  "llm_config": {
    "model": "gpt-4",
    "base_url": "https://api.openai.com/v1",
    "api_key": "${OPENAI_API_KEY}",
    "temperature": 0.3,
    "max_tokens": 2000
  },
  "agent_config": {
    "url": "http://localhost:8002",
    "capabilities": ["web_retrieval", "web_onchain_retrieval", "onchain_retrieval"],
    "timeout": 30,
    "close_endpoint": "http://localhost:8002/close"
  },
  "dataset_config": {
    "type": "csv",
    "path": "data/benchmark.csv",
    "abbr": "benchmark"
  },
  "evaluator_config": {
    "type": "hybrid",
    "rule_based_weight": 0.3,
    "llm_based_weight": 0.7,
    "llm_evaluation_threshold": 0.5
  },
  "runner_config": {
    "type": "local",
    "max_workers": 4,
    "timeout": 300
  },
  "work_dir": "output"
}
```

See `configs/example_config.py` for a Python-based configuration example.

## Evaluation Methods

The SDK automatically selects evaluation methods based on task categories, but you can also specify them explicitly:

### Rule-Based Evaluation
Best for tasks requiring precise matching, such as:
- Factual Q&A with specific answers
- Event timeline extraction
- Technical parameter queries

### LLM-Based Evaluation
Ideal for tasks requiring understanding and reasoning:
- Abstractive summarization
- Complex technical explanations
- Disambiguation tasks

### Hybrid Evaluation
Combines both approaches for tasks needing both precision and comprehension:
- Cross-source verification
- Multi-step reasoning
- Tasks with both factual and explanatory components

## Results and Analytics

The SDK provides comprehensive result analysis:

**Overall Statistics:**
- Total tasks, success/failure rates
- Average scores and confidence levels
- Processing time metrics

**Category Breakdown:**
- Performance by task category
- Success rates per category
- Average scores per category

**Score Distribution:**
- Excellent (≥0.8)
- Good (0.6-0.8)
- Fair (0.4-0.6)
- Poor (<0.4)

Results are exported in both JSON (detailed) and CSV (tabular) formats for easy analysis and reporting.

## Architecture

The SDK follows a modular architecture inspired by industry best practices:

- **EvaluationEngine**: Core evaluation logic and method selection
- **Runner**: Task execution and scheduling (supports local and distributed execution)
- **Summarizer**: Result aggregation and statistical analysis
- **Config System**: Centralized configuration management

This separation of concerns makes the framework easy to extend, test, and customize for your specific needs.

## Examples

Check out the `examples/` directory for complete working examples:

- `basic_usage.py` - Simple single-question evaluation
- `batch_evaluation.py` - Batch processing with custom logic
- `config_based_evaluation.py` - Using configuration files
- `enhanced_batch_evaluation.py` - Advanced batch evaluation with statistics
- `evaluate_accuracy.py` - Evaluation system accuracy analysis

## API Reference

### DABEvaluator

Main evaluation class.

```python
evaluator = DABEvaluator(config: EvaluationConfig)
```

**Methods:**

- `evaluate_agent(question, agent_metadata, category, evaluation_method, expected_answer, context)` - Evaluate a single agent response
- `evaluate_agent_with_dataset(agent_metadata, dataset_path, max_tasks)` - Evaluate against a dataset
- `export_results(format)` - Export results (format: "json" or "csv")

### Configuration Classes

- `EvaluationConfig` - Complete evaluation configuration
- `LLMConfig` - LLM model configuration
- `AgentConfig` - Agent endpoint configuration
- `DatasetConfig` - Dataset loading configuration
- `EvaluatorConfig` - Evaluation method configuration
- `RunnerConfig` - Task execution configuration

### Enums

- `TaskCategory`: `WEB_RETRIEVAL`, `WEB_ONCHAIN_RETRIEVAL`, `ONCHAIN_RETRIEVAL`
- `EvaluationMethod`: `RULE_BASED`, `LLM_BASED`, `HYBRID`
- `EvaluationStatus`: `PENDING`, `IN_PROGRESS`, `COMPLETED`, `FAILED`

## Requirements

- Python 3.8+
- httpx
- openai (or compatible API client)
- sentence-transformers
- scikit-learn
- numpy

## Evaluation Accuracy

The SDK includes tools for analyzing the accuracy of the evaluation system itself. Use `EvaluationAccuracyAnalyzer` to:

- Measure consistency across multiple evaluation runs
- Detect biases (length, category, score clustering)
- Analyze confidence-accuracy correlation
- Identify false positives and false negatives

See `examples/evaluate_accuracy.py` for usage examples and `EVALUATION_ACCURACY.md` for detailed documentation.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
