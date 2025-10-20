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

__version__ = "1.0.0"
__author__ = "DAB Team"
__email__ = "dab@example.com"

# Import main classes and functions
from .dab_eval import (
    DABEvaluator,
    AgentMetadata,
    EvaluationTask,
    EvaluationResult,
    TaskCategory,
    EvaluationMethod,
    EvaluationStatus,
    evaluate_agent,
)

# Import evaluation modules
from .evaluation import (
    BaseEvaluator,
    LLMEvaluator,
    HybridEvaluator,
)

__all__ = [
    # Main classes
    "DABEvaluator",
    "AgentMetadata", 
    "EvaluationTask",
    "EvaluationResult",
    
    # Enums
    "TaskCategory",
    "EvaluationMethod", 
    "EvaluationStatus",
    
    # Convenience functions
    "evaluate_agent",
    
    # Evaluation modules
    "BaseEvaluator",
    "LLMEvaluator", 
    "HybridEvaluator",
    
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
