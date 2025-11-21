# DAB Evaluation SDK - Web3 Agent Evaluation SDK

DAB Evaluation SDK is a Python SDK focused on Web3 Agent evaluation, providing simple APIs to evaluate Agent capabilities and performance.

## Features

- **Intelligent Evaluation Strategy**: Automatically select evaluation methods based on task category and type
- **Multi-dimensional Evaluation**: Fact accuracy, technical expertise, detail level, source credibility
- **Hybrid Evaluation**: Combine rule-based and LLM-based evaluation advantages with adaptive weighting
- **Enhanced Scoring Engine**: Normalizes formats, checks key facts, and provides confidence-aware scores for robust rule evaluation
- **Comprehensive Statistics**: Category-based performance analysis with detailed success rates and score distribution
- **Enum-based API**: Use enums instead of strings for better type safety
- **Configurable LLM**: Support custom LLM configurations
- **Automatic Export**: Results automatically saved to specified output directory

## Quick Start

### Install Dependencies

```bash
pip install httpx openai
```

### Configure LLM Access

The SDK does not ship with any credentials or LLM defaults. You must provide one of the following when constructing `DABEvaluator`:
- Pass an OpenAI-compatible client instance via `llm_config["client"]`.
- Or provide `llm_config["api_key"]`, `llm_config["model"]`, and optionally `llm_config["base_url"]`; the SDK will build the client for you.
- As a fallback, set the `ARK_API_KEY` environment variable together with `llm_config["model"]` and, if needed, `llm_config["base_url"]` before running the SDK.

To keep evaluations reproducible and fair, the LLM evaluator now accepts a few reliability-focused options:

| Key | Default | Description |
| --- | --- | --- |
| `num_samples` | `1` | Number of independent LLM judgments to gather before aggregation. Useful for reducing variance. |
| `max_retries` | `2` | How many times to re-ask the model when it fails to return valid JSON. |
| `require_valid_json` | `True` | When `True`, invalid responses are dropped and re-tried rather than silently scored. |
| `retry_invalid_json` | `True` | Disable if you want a single pass even when the output is malformed. |

Each sample stores a raw response plus schema flags, and the final result exposes trimmed-mean aggregate scores in `details["samples"]` and `details["dimension_breakdown"]`.

### Basic Usage

```python
import asyncio
from dab_eval import DABEvaluator, AgentMetadata, TaskCategory

async def main():
    # LLM configuration
    llm_config = {
        "model": "doubao-seed-1-6",
        "temperature": 0.3,
        "max_tokens": 2000,
        "api_key": os.environ["ARK_API_KEY"],
        "base_url": ""
    }
    
    # Create SDK instance with LLM config and output path
    evaluator = DABEvaluator(llm_config, "output")  # Uses enhanced hybrid evaluation by default
    
    # Define Agent
    agent = AgentMetadata(
        url="http://localhost:8002",
        capabilities=[TaskCategory.ONCHAIN_RETRIEVAL],
        timeout=30
    )
    
    # Evaluate Agent using enums
    result = await evaluator.evaluate_agent(
        question="What is the date (UTC) when the US SEC approved Bitcoin spot ETF?",
        agent_metadata=agent,
        category=TaskCategory.WEB_RETRIEVAL
    )
    
    print(f"Evaluation Score: {result.evaluation_score}")
    print(f"Agent Response: {result.agent_response}")

# Dataset-based evaluation
async def dataset_evaluation():
    """Evaluate Agent using dataset"""
    # LLM configuration
    llm_config = {
        "model": "doubao-seed-1-6-251015",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key": os.environ.get("ARK_API_KEY", "your_api_key_here"),
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    # Create SDK instance
    evaluator = DABEvaluator(llm_config, "output")
    
    # Define Agent
    agent = AgentMetadata(
        url="http://localhost:8002",
        capabilities=[TaskCategory.WEB_RETRIEVAL, TaskCategory.ONCHAIN_RETRIEVAL],
        timeout=30
    )
    
    # Evaluate using dataset
    results = await evaluator.evaluate_agent_with_dataset(
        agent_metadata=agent,
        dataset_path="data/benchmark.csv",
        max_tasks=5  # Limit to first 5 tasks
    )
    
    # Export results
    export_data = evaluator.export_results("json")
    print(f"Total Score: {export_data.get('total_score', 0):.2f}")
    print(f"Successful Tasks: {export_data.get('successful_tasks', 0)}")

asyncio.run(main())

### Using the Enhanced Hybrid Evaluator

The default `DABEvaluator` now builds on the enhanced hybrid evaluator within `dab_eval.evaluation`. The hybrid workflow:
- Normalises dates / numbers / addresses before matching expected answers.
- Multiplies rule-based scores by calibrated confidence and blends them with LLM scores only when the LLM output is well-formed.
- Drops LLM scores that fall below the configured `llm_evaluation_threshold`.
- Surfaces multi-dimensional metrics (`accuracy`, `completeness`, `professionalism`, `usefulness`) by merging rule and LLM evidence so you can audit how a score was produced.

If you need to tweak settings while using `DABEvaluator`, access the underlying evaluator via `evaluator.evaluators["hybrid"]` and adjust its attributes (for example, `llm_evaluation_threshold` or `rule_based_weight`).

To customise the evaluation pipeline, you can instantiate the hybrid evaluator directly:

```python
from dab_eval.evaluation import HybridEvaluator

hybrid = HybridEvaluator({
    "use_llm_evaluation": True,
    "llm_evaluation_threshold": 0.6,
    "rule_based_weight": 0.4,
    "llm_based_weight": 0.6,
    "use_enhanced_scoring": True,
    "llm_config": llm_config,
})
result = await hybrid.evaluate(
    question="When did the SEC approve the bitcoin spot ETF?",
    agent_response="The SEC approved it on January 10, 2024.",
    expected_answer="2024-01-10",
    context={"category": "web_retrieval"},
)
print(result["score"], result["reasoning"])
```

You can also reuse the enhanced scoring utilities independently:

```python
from dab_eval.evaluation import EnhancedScoringSystem, ScoringMethod

scoring = EnhancedScoringSystem()
result = scoring.score_answer(
    expected="Approval date: 2024-01-10",
    agent="The SEC approved it on January 10, 2024.",
    method=ScoringMethod.FORMAT_STRICT,
)
print(result.score, result.confidence, result.reasoning)
```
```

## API Documentation

### DABEvaluator Class

Main SDK class providing Agent evaluation functionality.

#### Constructor

```python
DABEvaluator(llm_config: Dict[str, Any], output_path: str = "output")
```

- `llm_config`: LLM configuration dictionary
- `output_path`: Output directory for results

#### Methods

##### `evaluate_agent(question, agent_metadata, task_type, context, category, evaluation_method, expected_answer)`

Evaluate single Agent.

**Parameters:**
- `question` (str): Evaluation question
- `agent_metadata` (AgentMetadata): Agent metadata
- `task_type` (TaskType): Task type enum
- `context` (dict): Context information
- `category` (TaskCategory): Task category enum
- `evaluation_method` (EvaluationMethod, optional): Evaluation method enum (auto-selected if not provided)
- `expected_answer` (str, optional): Expected answer

**Returns:**
- `EvaluationResult`: Evaluation result

##### `get_task_status(task_id)`

Get task status.

**Parameters:**
- `task_id` (str): Task ID

**Returns:**
- `EvaluationTask`: Task status

##### `list_tasks()`

List all tasks.

**Returns:**
- `List[EvaluationTask]`: All tasks

##### `export_results(format)`

Export evaluation results to output_path.

**Parameters:**
- `format` (str): Export format ("json" or "csv")

**Returns:**
- `Union[str, Dict[str, Any]]`: Exported results

#### Enums

##### TaskCategory

- `WEB_RETRIEVAL`: Web retrieval tasks
- `WEB_ONCHAIN_RETRIEVAL`: Web + On-chain retrieval tasks
- `ONCHAIN_RETRIEVAL`: On-chain retrieval tasks

##### TaskType

- `SINGLE_DOC_FACT_QA`: Single document fact Q&A
- `EVENT_TIMELINE`: Event timeline extraction
- `DISAMBIGUATION`: Disambiguation tasks
- `ABSTRACTIVE_WITH_EVIDENCE`: Abstractive tasks with evidence
- `CLAIM_VERIFICATION`: Claim verification
- `RECONCILIATION`: Reconciliation tasks
- `CROSS_SOURCE_EXPLANATION`: Cross-source explanation
- `PARAMETER_PERMISSION_CHANGE`: Parameter/permission change detection
- `TECHNICAL_KNOWLEDGE`: Technical knowledge tasks
- `TOP_K_AGGREGATION`: Top-K aggregation tasks

##### EvaluationMethod

- `RULE_BASED`: Rule-based evaluation
- `LLM_BASED`: LLM-based evaluation
- `HYBRID`: Hybrid evaluation (rule-based + LLM-based)

### AgentMetadata Class

Agent metadata for evaluation.

**Parameters:**
- `url` (str): Agent API URL
- `capabilities` (List[str]): Agent capabilities
- `timeout` (int): Request timeout (default: 30)
- `close_endpoint` (str, optional): Agent close endpoint
- `api_key` (str, optional): API key

### EvaluationResult Class

Evaluation result data.

**Attributes:**
- `task_id` (str): Task ID
- `question` (str): Evaluation question
- `agent_response` (str): Agent response
- `evaluation_score` (float): Evaluation score (0.0-1.0)
- `evaluation_reasoning` (str): Evaluation reasoning
- `confidence` (float): Agent confidence
- `processing_time` (float): Processing time
- `tools_used` (List[str]): Tools used
- `metadata` (dict): Additional metadata
- `status` (EvaluationStatus): Task status
- `error` (str, optional): Error message

## Intelligent Evaluation Strategy

The SDK automatically selects the most appropriate evaluation method based on task category and type:

### Web Retrieval Tasks
- **Single Document Fact Q&A**: Rule-based evaluation
- **Event Timeline**: Rule-based evaluation
- **Abstractive with Evidence**: LLM-based evaluation
- **Disambiguation**: LLM-based evaluation

### Web + On-chain Retrieval Tasks
- **Claim Verification**: Hybrid evaluation
- **Reconciliation**: Hybrid evaluation
- **Cross-source Explanation**: LLM-based evaluation

### On-chain Retrieval Tasks
- **Event Timeline**: Rule-based evaluation
- **Parameter/Permission Change**: Rule-based evaluation
- **Technical Knowledge**: LLM-based evaluation

## Examples

### Basic Evaluation

```python
from dab_eval import DABEvaluator, AgentMetadata, TaskCategory

    # LLM configuration
    llm_config = {
        "model": "doubao-seed-1-6-251015",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key": os.environ.get("ARK_API_KEY", "your_api_key_here"),
        "temperature": 0.3,
        "max_tokens": 2000
    }

# Create evaluator
evaluator = DABEvaluator(llm_config, "output")

# Define agent
agent = AgentMetadata(
    url="http://localhost:8002",
    capabilities=["web3_analysis"],
    timeout=30
)

# Evaluate agent
result = await evaluator.evaluate_agent(
    question="What is the topic0 of ERC-20 Transfer event?",
    agent_metadata=agent,
    task_type=TaskType.TECHNICAL_KNOWLEDGE,
    category=TaskCategory.ONCHAIN_RETRIEVAL
)
```

### Batch Evaluation

```python
import pandas as pd
from dab_eval import DABEvaluator, AgentMetadata, TaskCategory

# Load benchmark data
df = pd.read_csv("data/benchmark.csv")

# Create evaluator
evaluator = DABEvaluator(llm_config, "output")

# Define agent
agent = AgentMetadata(
    url="http://localhost:8002",
    capabilities=["web3_analysis"],
    timeout=30
)

# Batch evaluation
results = []
for _, row in df.iterrows():
    result = await evaluator.evaluate_agent(
        question=row["question"],
        agent_metadata=agent,
        task_type=TaskType(row["task_type"]),
        category=TaskCategory(row["category"]),
        expected_answer=row["answer"]
    )
    results.append(result)

# Export results
json_results = evaluator.export_results("json")
csv_results = evaluator.export_results("csv")
```

## Output Files

The SDK automatically generates the following files in the specified output directory:

- `evaluation_results.json`: Detailed evaluation results in JSON format
- `evaluation_results.csv`: Evaluation results in CSV format

## Requirements

- Python 3.8+
- httpx
- openai

## License

MIT License
