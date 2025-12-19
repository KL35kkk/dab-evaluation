"""
Enums for DAB Evaluation SDK
"""

from enum import Enum


class TaskCategory(Enum):
    """Task category enumeration"""
    WEB_RETRIEVAL = "web_retrieval"
    WEB_ONCHAIN_RETRIEVAL = "web_onchain_retrieval"
    ONCHAIN_RETRIEVAL = "onchain_retrieval"


class EvaluationMethod(Enum):
    """Evaluation method enumeration"""
    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"
    HYBRID = "hybrid"
    CASCADE = "cascade"


class EvaluationStatus(Enum):
    """Evaluation status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
