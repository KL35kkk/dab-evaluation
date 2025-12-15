#!/usr/bin/env python3
"""
DAB Evaluation SDK - Web3 Agent Evaluation SDK
Refactored with modular architecture
"""

import asyncio
import csv
import time
import os
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Import evaluation capabilities
from .evaluation.base_evaluator import BaseEvaluator
from .evaluation.llm_evaluator import LLMEvaluator
from .evaluation.hybrid_evaluator import HybridEvaluator

# Import new modular components
from .config import (
    EvaluationConfig, LLMConfig, AgentConfig, DatasetConfig,
    EvaluatorConfig, RunnerConfig, TaskCategory, EvaluationMethod
)
from .evaluation_engine import EvaluationEngine
from .runners.local import LocalRunner
from .runners.agent_runner import AgentRunner
from .summarizers.default import DefaultSummarizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Evaluation status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMetadata:
    """Agent metadata"""
    url: str
    capabilities: List[TaskCategory]
    timeout: int = 30
    close_endpoint: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class EvaluationTask:
    """Evaluation task"""
    task_id: str
    question: str
    agent_metadata: AgentMetadata
    context: Dict[str, Any]
    category: TaskCategory = TaskCategory.WEB_RETRIEVAL
    evaluation_method: EvaluationMethod = EvaluationMethod.HYBRID
    expected_answer: Optional[str] = None
    status: EvaluationStatus = EvaluationStatus.PENDING
    agent_response: Optional[Dict[str, Any]] = None
    evaluation_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None


@dataclass
class EvaluationResult:
    """Evaluation result"""
    task_id: str
    question: str
    agent_response: str
    evaluation_score: float
    evaluation_reasoning: str
    confidence: float
    processing_time: float
    tools_used: List[str]
    metadata: Dict[str, Any]
    status: EvaluationStatus
    error: Optional[str] = None


class DABEvaluator:
    """DAB Evaluator with modular architecture.
    
    This version separates concerns:
    - EvaluationEngine: Handles evaluation logic
    - Runner: Handles task execution
    - Summarizer: Handles result aggregation and reporting
    """
    
    def __init__(self, 
                 config: EvaluationConfig):
        """
        Initialize DAB Evaluator.
        
        Args:
            config: EvaluationConfig instance
        """
        if config is None:
            raise ValueError("config must be provided")
        
        self.config = config
        self.output_path = config.work_dir
        
        # Initialize components
        self.evaluation_engine = EvaluationEngine(
            llm_config=config.llm_config.to_dict(),
            evaluator_config=config.evaluator_config.to_dict() if config.evaluator_config else None
        )
        
        runner_config = config.runner_config.to_dict() if config.runner_config else {}
        self.runner = LocalRunner(config=runner_config)
        self.agent_runner = AgentRunner(config={'timeout': 30})
        
        self.summarizer = DefaultSummarizer(config={})
        
        # Task storage
        self.results: List[Dict[str, Any]] = []
    
    async def evaluate_agent(self,
                            question: str,
                            agent_metadata: AgentMetadata,
                            context: Optional[Dict[str, Any]] = None,
                            category: TaskCategory = TaskCategory.WEB_RETRIEVAL,
                            evaluation_method: Optional[EvaluationMethod] = None,
                            expected_answer: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate single Agent.
        
        Args:
            question: Evaluation question
            agent_metadata: Agent metadata
            context: Additional context
            category: Task category
            evaluation_method: Evaluation method (auto-selected if None)
            expected_answer: Expected answer (optional)
            
        Returns:
            EvaluationResult instance
        """
        result_dict = await self.evaluation_engine.evaluate_task(
            question=question,
            agent_url=agent_metadata.url,
            agent_timeout=agent_metadata.timeout,
            expected_answer=expected_answer,
            category=category,
            evaluation_method=evaluation_method,
            context=context,
            close_endpoint=agent_metadata.close_endpoint
        )
        
        # Convert to EvaluationResult
        task_id = f"task_{int(time.time())}_{len(self.results)}"
        result = EvaluationResult(
            task_id=task_id,
            question=result_dict["question"],
            agent_response=result_dict["agent_response"],
            evaluation_score=result_dict["evaluation_score"],
            evaluation_reasoning=result_dict["evaluation_reasoning"],
            confidence=result_dict["confidence"],
            processing_time=result_dict["processing_time"],
            tools_used=result_dict.get("tools_used", []),
            metadata=result_dict.get("metadata", {}),
            status=EvaluationStatus(result_dict["status"]),
            error=result_dict.get("error")
        )
        
        # Store result
        self.results.append(result_dict)
        
        return result
    
    async def evaluate_agent_with_dataset(self,
                                         agent_metadata: AgentMetadata,
                                         dataset_path: str,
                                         max_tasks: Optional[int] = None) -> List[EvaluationResult]:
        """
        Evaluate Agent using dataset.
        
        Args:
            agent_metadata: Agent metadata
            dataset_path: Path to dataset CSV file
            max_tasks: Maximum number of tasks to evaluate (None = all)
            
        Returns:
            List of EvaluationResult instances
        """
        # Load dataset
        tasks = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    category_map = {
                        "web_retrieval": TaskCategory.WEB_RETRIEVAL,
                        "web_onchain_retrieval": TaskCategory.WEB_ONCHAIN_RETRIEVAL,
                        "onchain_retrieval": TaskCategory.ONCHAIN_RETRIEVAL
                    }
                    
                    method_map = {
                        "rule_based": EvaluationMethod.RULE_BASED,
                        "llm_based": EvaluationMethod.LLM_BASED,
                        "hybrid": EvaluationMethod.HYBRID
                    }
                    
                    task_data = {
                        "question": row["question"],
                        "expected_answer": row.get("answer", ""),
                        "category": category_map.get(row.get("category", "web_retrieval"), TaskCategory.WEB_RETRIEVAL),
                        "evaluation_method": method_map.get(row.get("evaluation_method", "hybrid"), EvaluationMethod.HYBRID),
                        "context": {"dataset_id": row.get("id", "")}
                    }
                    tasks.append(task_data)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        if max_tasks:
            tasks = tasks[:max_tasks]
        
        # Evaluate all tasks
        results = []
        for i, task_data in enumerate(tasks):
            try:
                logger.info(f"Evaluating task {i+1}/{len(tasks)}: {task_data['question'][:50]}...")
                
                result = await self.evaluate_agent(
                    question=task_data["question"],
                    agent_metadata=agent_metadata,
                    context=task_data["context"],
                    category=task_data["category"],
                    evaluation_method=task_data["evaluation_method"],
                    expected_answer=task_data["expected_answer"]
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate task {i+1}: {e}")
                # Create failed result
                failed_result = EvaluationResult(
                    task_id=f"failed_{i+1}",
                    question=task_data["question"],
                    agent_response="",
                    evaluation_score=0.0,
                    evaluation_reasoning="",
                    confidence=0.0,
                    processing_time=0.0,
                    tools_used=[],
                    metadata={},
                    status=EvaluationStatus.FAILED,
                    error=str(e)
                )
                results.append(failed_result)
                self.results.append({
                    "status": "failed",
                    "question": task_data["question"],
                    "error": str(e)
                })
        
        return results
    
    def export_results(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export evaluation results.
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Exported data (dict for JSON, string for CSV)
        """
        # Convert results to summary format
        summary = self.summarizer.summarize(self.results)
        
        if format == "json":
            # Save summary
            summary_path = self.summarizer.export(summary, self.output_path, "json")
            
            # Also save detailed results
            details_path = os.path.join(self.output_path, "evaluation_results.json")
            with open(details_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_score": summary["overall"]["average_score"],
                    "total_tasks": summary["overall"]["total_tasks"],
                    "successful_tasks": summary["overall"]["successful_tasks"],
                    "failed_tasks": summary["overall"]["failed_tasks"],
                    "results": self.results
                }, f, indent=2, ensure_ascii=False)
            
            return summary
        
        elif format == "csv":
            # Export summary as CSV
            summary_path = self.summarizer.export(summary, self.output_path, "csv")
            
            # Also export detailed results
            csv_path = os.path.join(self.output_path, "evaluation_results.csv")
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'task_id', 'question', 'category', 'evaluation_method',
                    'agent_response', 'evaluation_score', 'evaluation_reasoning',
                    'confidence', 'processing_time', 'tools_used', 'status', 'error'
                ])
                for result in self.results:
                    writer.writerow([
                        result.get('task_id', ''),
                        result.get('question', ''),
                        result.get('category', ''),
                        result.get('evaluation_method', ''),
                        result.get('agent_response', ''),
                        result.get('evaluation_score', 0.0),
                        result.get('evaluation_reasoning', ''),
                        result.get('confidence', 0.0),
                        result.get('processing_time', 0.0),
                        ','.join(result.get('tools_used', [])),
                        result.get('status', ''),
                        result.get('error', '')
                    ])
            
            return summary_path
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience function for backward compatibility (deprecated)
async def evaluate_agent(question: str, agent_url: str, llm_config: Dict[str, Any], 
                        output_path: str = "output", **kwargs) -> EvaluationResult:
    """
    Convenient Agent evaluation function (deprecated).
    
    This function is deprecated. Please use DABEvaluator with EvaluationConfig instead.
    """
    import warnings
    warnings.warn(
        "evaluate_agent() is deprecated. Use DABEvaluator with EvaluationConfig instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create minimal config
    llm_cfg = LLMConfig.from_dict(llm_config)
    agent_cfg = AgentConfig(
        url=agent_url,
        capabilities=kwargs.get("capabilities", [TaskCategory.WEB_RETRIEVAL]),
        timeout=kwargs.get("timeout", 30),
        close_endpoint=kwargs.get("close_endpoint"),
        api_key=kwargs.get("api_key")
    )
    dataset_cfg = DatasetConfig()
    config = EvaluationConfig(
        llm_config=llm_cfg,
        agent_config=agent_cfg,
        dataset_config=dataset_cfg,
        work_dir=output_path
    )
    
    evaluator = DABEvaluator(config=config)
    
    agent_metadata = AgentMetadata(
        url=agent_url,
        capabilities=kwargs.get("capabilities", [TaskCategory.WEB_RETRIEVAL]),
        timeout=kwargs.get("timeout", 30),
        close_endpoint=kwargs.get("close_endpoint"),
        api_key=kwargs.get("api_key")
    )
    
    return await evaluator.evaluate_agent(
        question=question,
        agent_metadata=agent_metadata,
        context=kwargs.get("context", {}),
        category=kwargs.get("category", TaskCategory.WEB_RETRIEVAL),
        evaluation_method=kwargs.get("evaluation_method"),
        expected_answer=kwargs.get("expected_answer")
    )
