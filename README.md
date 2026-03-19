# DAB Evaluation

**DAB (Decentralized Agent Benchmark)** is a **production-grade Agent Evaluation Pipeline** designed specifically for **Web3 agents**. It goes beyond simple Q&A scoring — it's a full evaluation infrastructure that continuously controls agent quality, cost, and risk across iterative development cycles.

## What This Is (And What It Isn't)

| Old thinking | Production thinking |
|---|---|
| "Test good or not" | "Stabilise quality, cost, and risk during continuous iteration" |
| Score a single answer | Evaluate trajectory + outcome across N trials |
| Run once manually | Regression detection + CI/CD gate |
| Black-box agent call | Full trace: thought → tool call → observation → state |

---

## Architecture

```
┌──────────────────────────────────────────┐
│           Task Dataset Hub               │
│  Gold (human) / Synthetic / Production   │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│          Scenario Generator              │
│  tool_failure / missing_info /           │
│  conflicting / price_change / redirect   │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│        Agent Runner  (Multi-Trial)       │
│  Trajectory · Outcome · TrackedMetrics   │
└────────┬──────────────┬──────────────────┘
         │              │
┌────────▼────┐  ┌──────▼──────────────────┐
│  4 Graders  │  │     Metrics System       │
│ deterministic│  │ quality · efficiency     │
│ llm_rubric  │  │ cost · stability ·       │
│ state_check │  │ robustness               │
│ tool_calls  │  └──────┬──────────────────┘
└────────┬────┘         │
         └──────┬───────┘
                │
┌───────────────▼──────────────────────────┐
│              Analyzer                    │
│  Failure Clustering · Regression Diff    │
│  Root Cause Attribution                  │
└───────────────┬──────────────────────────┘
                │
┌───────────────▼──────────────────────────┐
│            CI / CD Gate                  │
│  success_rate ≥ 92% · cost Δ ≤ 5%       │
│  no new failure modes  →  Pass / Fail    │
└──────────────────────────────────────────┘
```

---

## Modules

### Task Dataset Hub

Three-tier task library that prevents evaluation distortion:

| Tier | Source | Purpose |
|---|---|---|
| **Gold** | Human-annotated | Regression / quality gate |
| **Synthetic** | LLM-generated variants | Coverage expansion |
| **Production** | Live traffic replay | Real-world fidelity |

```python
from dab_eval import TaskHub, TaskTier

hub = TaskHub(seed=42)
hub.load_gold("data/benchmark.csv")          # 100+ Web3 benchmark tasks
hub.load_dmind("data/dmind/dmind_objective.csv")  # 3000+ MC questions

# 50% gold / 30% synthetic / 20% production, with filters
batch = hub.sample(100, category_filter="onchain_retrieval")

# Stream by tier
for task in hub.iter_tier(TaskTier.GOLD):
    ...
```

### Scenario Generator

Tests not just *capability* but *resilience under chaos*:

```python
from dab_eval import ScenarioGenerator, Perturbation

gen = ScenarioGenerator(seed=42)

# Single scenario
scenario = gen.generate(task, [
    Perturbation.TOOL_FAILURE,   # API timeout / HTTP error / rate limit
    Perturbation.MISSING_INFO,   # Remove a context key
    Perturbation.CONFLICTING,    # Inject a contradictory hint
    Perturbation.PRICE_CHANGE,   # Data changes mid-run
    Perturbation.USER_REDIRECT,  # User changes their question
])

# All combinations up to 2 perturbations
scenarios = gen.generate_all_combinations(task, max_combinations=10)
```

### Agent Runner + Multi-Trial

Every task runs N times to handle non-determinism:

```python
from dab_eval import MultiTrialRunner

# run_fn: async (task, trial_id) -> Trial
runner = MultiTrialRunner(run_fn, num_trials=3, variance_threshold=0.15)
result = await runner.run(task)

print(result.mean_score)       # 0.87
print(result.variance)         # 0.02
print(result.worst_case_score) # 0.83
print(result.pass_at_k)        # {1: 0.83, 2: 0.97, 3: 1.0}
print(result.is_unstable)      # False  (variance < 0.15)
```

### 4 Graders (run in parallel per trial)

| Grader | Type | Evaluates |
|---|---|---|
| `DeterministicTestsGrader` | Code-based | Outcome: exact match, regex, MC options, Web3 address format |
| `LLMRubricGrader` | Model-based | Trajectory + Outcome: rubric scoring |
| `StateCheckGrader` | Code-based | Outcome: source reliability, state consistency |
| `ToolCallsGrader` | Code-based | Trajectory: success rate, relevance, loop detection |

```python
from dab_eval import build_grader

grader = build_grader("tool_calls", config={"max_tool_calls": 10})
result = await grader.grade(outcome, trajectory, task)

print(result.score)      # 0.82
print(result.passed)     # True
print(result.assertions) # [{"name": "tool_success_rate", "passed": True, "value": 0.9}, ...]
```

### Tool Mock Registry

Deterministic, replayable tool execution — no real API calls needed:

```python
from dab_eval import MockToolRegistry, ToolFailureMode, register_default_tools

registry = MockToolRegistry(cache_dir="output/tool_cache")
register_default_tools(registry)  # web_search, blockchain_query, contract_read, ...

# Inject a failure for the next call
registry.inject_failure("blockchain_query", ToolFailureMode.TIMEOUT)

# All calls are cached by (tool_name, args_hash) → full replay capability
result = await registry.call("web_search", {"query": "bitcoin ETF approval"})
```

### Analyzer

```python
from dab_eval import FailureClassifier, RegressionDetector, RootCauseAnalyzer

# Classify why each trial failed
classifier = FailureClassifier()
failures = classifier.classify_batch(trials)
# → [ClassifiedFailure(category=FailureCategory.TOOL_ERROR, confidence=0.85), ...]

# Compare two runs (detect regressions)
detector = RegressionDetector(regression_threshold=0.05)
report = detector.compare(baseline_snapshots, current_snapshots)
print(report.summary_text())
# Score delta: -0.02
# Regressions: 2   task_042: 0.91 → 0.78 (-0.13)
# New failure modes: []

# Root cause percentage breakdown
analyzer = RootCauseAnalyzer()
rc = analyzer.analyse_from_trials(trials, total_trials=300)
print(rc.summary_text())
# tool_error       ████████████ 60%
# reasoning_error  █████        25%
# planning_error   ███          15%
```

### CI/CD Gate

```python
from dab_eval import CIGate, GateConfig

gate = CIGate(GateConfig(
    min_success_rate=0.92,       # ≥ 92% tasks must pass
    max_cost_increase_pct=5.0,   # cost must not grow > 5% vs baseline
    max_regressions=0,           # zero regressions allowed
    max_new_failure_modes=0,     # no new failure categories
))

result = gate.evaluate(current_results, baseline_results)

if not result.passed:
    print(result.report)  # full breakdown
    sys.exit(1)           # blocks deployment
```

**CLI** (for GitHub Actions / Jenkins):

```bash
python -m dab_eval.ci_gate \
  --current  output/results.json \
  --baseline output/baseline.json \
  --min-success-rate 0.92 \
  --max-cost-increase 5.0
# Exit 0 = PASS, 1 = FAIL
```

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Mock Pipeline (No Agent Required)

```bash
python examples/mock_accuracy_run.py
```

### Evaluate Your Agent (v1 API — backward compatible)

```python
import asyncio, os
from dab_eval import DABEvaluator, EvaluationConfig, LLMConfig, AgentMetadata, TaskCategory

async def main():
    config = EvaluationConfig(
        llm_config=LLMConfig(
            model="gpt-4",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    )
    evaluator = DABEvaluator(config)
    agent = AgentMetadata(url="http://localhost:8002",
                          capabilities=[TaskCategory.WEB_RETRIEVAL])

    result = await evaluator.evaluate_agent(
        question="What date did the SEC approve Bitcoin spot ETF?",
        agent_metadata=agent,
        category=TaskCategory.WEB_RETRIEVAL,
        expected_answer="2024/1/10",
    )
    print(f"Score: {result.evaluation_score:.2f}")

asyncio.run(main())
```

### Full Dataset Evaluation

```python
results = await evaluator.evaluate_agent_with_dataset(
    agent_metadata=agent,
    dataset_path="data/benchmark.csv",
)
summary = evaluator.export_results("json")
print(f"Success rate: {summary['overall']['success_rate']:.2%}")
```

---

## Dataset

### Benchmark (`data/benchmark.csv`)

100 Web3 tasks across three categories:

| Category | Description |
|---|---|
| `web_retrieval` | Fetch / reason over traditional web sources |
| `onchain_retrieval` | Query blockchain data directly |
| `web_onchain_retrieval` | Hybrid: web + chain verification |

CSV columns: `id`, `question`, `answer`, `category`, `task_type`, `evaluation_method`, `difficulty` (optional), `ground_truth_score` (optional).

### DMind (`data/dmind/`)

3 156 multiple-choice questions across 9 Web3 domains: Blockchain Fundamentals, DeFi, NFT, DAO, Security, Smart Contracts, Tokenomics, MEME, Infrastructure.

```python
hub.load_dmind("data/dmind/dmind_objective.csv")
# or by domain
hub.load_dmind("data/dmind/objective/DeFi.csv", domain="DeFi")
```

---

## Agent Response Schema (v2)

To unlock trajectory-based grading (Tool Grader, State Check), agents should return structured JSON:

```json
{
  "answer": "2024/1/10",
  "confidence": 0.92,
  "trajectory": {
    "steps": [
      {"step_id": 1, "type": "reasoning", "content": "...", "timestamp": 1704067200.0},
      {"step_id": 2, "type": "tool_call", "tool_name": "web_search",
       "tool_input": {"query": "SEC Bitcoin ETF approval date"},
       "tool_output": {"results": [...]},
       "success": true, "timestamp": 1704067201.5},
      {"step_id": 3, "type": "reasoning", "content": "Found the date...", "timestamp": 1704067203.0}
    ],
    "n_turns": 3,
    "total_duration": 3.0
  },
  "state": {
    "final_answer_source": "web_search",
    "sources_consulted": ["sec.gov", "reuters.com"]
  },
  "tools_used": ["web_search"],
  "token_usage": {"prompt": 150, "completion": 80, "total": 230}
}
```

Agents that return only `{"answer": "..."}` still work — the framework degrades gracefully to deterministic + LLM grading only.

---

## Module Reference

### v2 Data Structures

| Class | Description |
|---|---|
| `Step` | Single action within a trajectory |
| `Trajectory` | Full execution trace (steps, counters, loop detection) |
| `Outcome` | Final answer + state + sources |
| `TrackedMetrics` | n_turns, n_toolcalls, tokens, latency, cost |
| `GradeResult` | Score + assertions from one grader |
| `Trial` | One complete run: Trajectory + Outcome + Metrics + GradeResults |

### Graders

```python
from dab_eval import build_grader, GRADER_REGISTRY  # from dab_eval.graders

grader = build_grader("deterministic_tests", config={"regex_patterns": [r"\d{4}/\d{1,2}/\d{1,2}"]})
```

Available: `deterministic_tests`, `llm_rubric`, `state_check`, `tool_calls`.

### Task Dataset Hub

```python
TaskHub(seed=42)
  .load_gold(path)
  .load_dmind(path, domain="")
  .add_synthetic(tasks)
  .add_production_log(entries)
  .load_production_jsonl(path)
  .sample(n, gold_ratio=0.5, synthetic_ratio=0.3, production_ratio=0.2,
          category_filter=None, difficulty_filter=None, domain_filter=None)
  .iter_tier(TaskTier.GOLD)
  .stats()
```

### Scenario Generator

```python
ScenarioGenerator(seed=42, cache_dir="output/tool_cache")
  .generate(task, perturbations=[Perturbation.TOOL_FAILURE, ...])
  .generate_batch(tasks, perturbations_per_task)
  .generate_all_combinations(task, max_combinations=10)
```

### Tool Mock Registry

```python
MockToolRegistry(cache_dir="output/tool_cache", default_latency_ms=50)
  .register(name, fn)
  .inject_failure(tool_name, ToolFailureMode.TIMEOUT)
  .call(tool_name, args)
  .call_log   # List[MockToolResponse]
```

Failure modes: `NONE`, `TIMEOUT`, `WRONG_DATA`, `EMPTY_RESULT`, `HTTP_ERROR`, `RATE_LIMIT`.

### Analyzer

```python
FailureClassifier(tool_error_threshold=0.7, pass_threshold=0.6)
  .classify(trial) -> Optional[ClassifiedFailure]
  .classify_batch(trials) -> List[ClassifiedFailure]

RegressionDetector(regression_threshold=0.05)
  .compare(baseline, candidate) -> RegressionReport
  .snapshots_from_results(results) -> List[TaskSnapshot]
  .save_snapshot(snapshots, path)
  .load_snapshot(path) -> List[TaskSnapshot]

RootCauseAnalyzer()
  .analyse(failures, total_trials) -> RootCauseReport
  .analyse_from_trials(trials, total_trials) -> RootCauseReport
```

Failure categories: `tool_error`, `reasoning_error`, `planning_error`, `loop_detected`, `no_answer`, `unknown`.

### CI Gate

```python
GateConfig(
    min_success_rate=0.92,
    max_cost_increase_pct=5.0,
    max_regressions=0,
    max_new_failure_modes=0,
    variance_threshold=0.15,
    unstable_task_limit=None,
)

CIGate(config).evaluate(current_results, baseline_results) -> GateResult
# GateResult: passed, blocking_issues, warnings, metrics, regression_report, report
```

---

## Examples

| File | Description |
|---|---|
| `examples/basic_usage.py` | Single-question evaluation |
| `examples/batch_evaluation.py` | Batch processing |
| `examples/config_based_evaluation.py` | Config-file driven evaluation |
| `examples/enhanced_batch_evaluation.py` | Advanced batch with statistics |
| `examples/evaluate_accuracy.py` | Evaluation system accuracy analysis |
| `examples/mock_accuracy_run.py` | Full pipeline offline (no real agent) |
| `examples/prepare_dmind.py` | Convert DMind dataset to DAB format |

---

## Requirements

- Python 3.8+
- `httpx`, `openai`, `pydantic`, `fastapi`, `uvicorn`
- `sentence-transformers`, `scikit-learn`, `numpy`

---

## Backward Compatibility

v2 is **fully backward compatible**. All v1 APIs (`DABEvaluator`, `evaluate_agent()`, `EvaluationResult`, config classes) continue to work unchanged. The new pipeline modules are additive.

---

## License

MIT License
