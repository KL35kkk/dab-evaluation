"""
DAB Evaluation SDK - Web3 Agent Evaluation Framework

A comprehensive evaluation framework for Web3 agents with support for:
- Multi-dimensional evaluation (factual accuracy, completeness, precision, relevance, conciseness)
- Semantic similarity using sentence transformers
- Rule-based, LLM-based, and hybrid evaluation methods
- Support for Web Retrieval, Web+Onchain Retrieval, and Onchain Retrieval tasks
- Configurable LLM backends (OpenAI, Anthropic, Google, Custom)
- Export results in JSON, CSV, and other formats
"""

__version__ = "2.0.0"
__author__ = "DAB Team"
__email__ = "dab@example.com"

# Import enums and dataclasses
from .enums import TaskCategory, EvaluationMethod, EvaluationStatus
from .dataclasses import AgentMetadata, EvaluationTask, EvaluationResult

# Import main classes and functions
from .dab_eval import (
    DABEvaluator,
    evaluate_agent,
)

# Import configuration system
from .config import (
    EvaluationConfig,
    LLMConfig,
    AgentConfig,
    DatasetConfig,
    EvaluatorConfig,
    RunnerConfig,
    StorageConfig,
    BusinessConfig,
    InfrastructureConfig,
    load_config,
)

# Import evaluation modules
from .evaluation import (
    BaseEvaluator,
    LLMEvaluator,
    HybridEvaluator,
)

# Import runners and summarizers
from .runners import BaseRunner, LocalRunner, MultiTrialRunner, MultiTrialResult
from .summarizers import BaseSummarizer, DefaultSummarizer

# Import v2 data structures
from .trajectory import (
    Step,
    StepType,
    Trajectory,
    Outcome,
    TrackedMetrics,
    GradeResult,
    Trial,
)

# Import datasets
from .datasets import TaskHub, TaskTier, HubTask

# Import tool mocking
from .tools import MockToolRegistry, ToolFailureMode, register_default_tools

# Import scenario generator
from .scenario_generator import ScenarioGenerator, ScenarioTask, Perturbation

# Import CI gate
from .ci_gate import CIGate, GateConfig, GateResult

# Import analyzer
from .analyzer import (
    FailureCategory,
    FailureClassifier,
    ClassifiedFailure,
    RegressionDetector,
    RegressionReport,
    TaskSnapshot,
    RootCauseAnalyzer,
    RootCauseReport,
)

# Import graders
from .graders import (
    BaseGrader,
    DeterministicTestsGrader,
    LLMRubricGrader,
    StateCheckGrader,
    ToolCallsGrader,
    build_grader,
)

# Import storage and task management
from .storage import ResultStorage
from .task_manager import TaskManager

__all__ = [
    # Main classes
    "DABEvaluator",
    "AgentMetadata", 
    "EvaluationTask",
    "EvaluationResult",
    
    # Configuration
    "EvaluationConfig",
    "LLMConfig",
    "AgentConfig",
    "DatasetConfig",
    "EvaluatorConfig",
    "RunnerConfig",
    "StorageConfig",
    "BusinessConfig",
    "InfrastructureConfig",
    "load_config",
    
    # Storage and Task Management
    "ResultStorage",
    "TaskManager",
    
    # Enums
    "TaskCategory",
    "EvaluationMethod", 
    "EvaluationStatus",
    
    # Convenience functions (deprecated)
    "evaluate_agent",
    
    # Evaluation modules
    "BaseEvaluator",
    "LLMEvaluator", 
    "HybridEvaluator",
    
    # Runners
    "BaseRunner",
    "LocalRunner",
    "MultiTrialRunner",
    "MultiTrialResult",

    # v2 trajectory / trial data structures
    "Step",
    "StepType",
    "Trajectory",
    "Outcome",
    "TrackedMetrics",
    "GradeResult",
    "Trial",

    # Graders
    "BaseGrader",
    "DeterministicTestsGrader",
    "LLMRubricGrader",
    "StateCheckGrader",
    "ToolCallsGrader",
    "build_grader",

    # Datasets
    "TaskHub",
    "TaskTier",
    "HubTask",

    # Tool mocking
    "MockToolRegistry",
    "ToolFailureMode",
    "register_default_tools",

    # Scenario generator
    "ScenarioGenerator",
    "ScenarioTask",
    "Perturbation",

    # CI Gate
    "CIGate",
    "GateConfig",
    "GateResult",

    # Analyzer
    "FailureCategory",
    "FailureClassifier",
    "ClassifiedFailure",
    "RegressionDetector",
    "RegressionReport",
    "TaskSnapshot",
    "RootCauseAnalyzer",
    "RootCauseReport",
    
    # Summarizers
    "BaseSummarizer",
    "DefaultSummarizer",
    
    # Package info
    "__version__",
    "__author__",
    "__email__",
]

def main():
    """Main entry point for command line usage"""
    import asyncio
    from .examples.basic_usage import main as example_main
    asyncio.run(example_main())

if __name__ == "__main__":
    main()
