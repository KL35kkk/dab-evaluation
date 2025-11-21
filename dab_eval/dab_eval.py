#!/usr/bin/env python3
"""
DAB Evaluation SDK - Web3 Agent Evaluation SDK
DAB Evaluation SDK - Focused on Agent evaluation and API calling capabilities
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import evaluation capabilities
from .evaluation.base_evaluator import BaseEvaluator
from .evaluation.llm_evaluator import LLMEvaluator
from .evaluation.hybrid_evaluator import HybridEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationStatus(Enum):
    """Evaluation status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class EvaluationMethod(Enum):
    """Evaluation method enumeration"""
    RULE_BASED = "rule_based"      # Rule-based evaluation
    LLM_BASED = "llm_based"        # LLM-based evaluation
    HYBRID = "hybrid"              # Hybrid evaluation

class TaskCategory(Enum):
    """Task category enumeration"""
    WEB_RETRIEVAL = "web_retrieval"           # Web retrieval
    WEB_ONCHAIN_RETRIEVAL = "web_onchain_retrieval"  # Web + On-chain retrieval
    ONCHAIN_RETRIEVAL = "onchain_retrieval"   # On-chain retrieval


@dataclass
class AgentMetadata:
    """Agent metadata"""
    url: str
    capabilities: List[TaskCategory]  # Use TaskCategory enum instead of strings
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
    """DAB Evaluator"""
    
    def __init__(self, llm_config: Dict[str, Any], output_path: str = "output"):
        self.tasks: Dict[str, EvaluationTask] = {}
        self.evaluators = {}
        self._task_counter = 0
        self.llm_config = llm_config
        self.output_path = output_path
        
        # Initialize evaluators
        self._init_evaluators()
    
    def _init_evaluators(self):
        """Initialize evaluators with LLM configuration"""
        try:
            # Validate LLM configuration
            required_keys = ["model", "base_url", "api_key"]
            missing_keys = [key for key in required_keys if key not in self.llm_config]
            if missing_keys:
                raise ValueError(f"Missing required LLM config keys: {missing_keys}")
            
            # Initialize LLM Evaluator with full configuration
            llm_evaluator_config = {
                "model": self.llm_config["model"],
                "base_url": self.llm_config["base_url"],
                "api_key": self.llm_config["api_key"],
                "temperature": self.llm_config.get("temperature", 0.3),
                "max_tokens": self.llm_config.get("max_tokens", 2000)
            }
            self.evaluators["llm"] = LLMEvaluator(llm_evaluator_config)
            
            # Initialize Hybrid Evaluator with full configuration
            hybrid_config = {
                "use_llm_evaluation": True,
                "llm_evaluation_threshold": 0.5,
                "rule_based_weight": 0.3,
                "llm_based_weight": 0.7,
                "llm_config": llm_evaluator_config
            }
            self.evaluators["hybrid"] = HybridEvaluator(hybrid_config)
            
            # Initialize semantic similarity model
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic similarity model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic model: {e}")
                self.semantic_model = None
            
            logger.info("Evaluators initialized successfully with LLM configuration")
            
        except Exception as e:
            logger.warning(f"Failed to initialize evaluators: {e}")
            logger.info("Using simplified evaluation logic")
    
    def _generate_task_id(self) -> str:
        """Generate task ID"""
        self._task_counter += 1
        return f"task_{self._task_counter}_{int(time.time())}"
    
    async def _call_agent_api(self, task: EvaluationTask) -> Dict[str, Any]:
        """Call Agent API"""
        try:
            async with httpx.AsyncClient(timeout=task.agent_metadata.timeout) as client:
                # Build request data
                request_data = {
                    "question": task.question,
                    "context": task.context
                }
                
                # Call Agent's /process endpoint
                response = await client.post(
                    f"{task.agent_metadata.url}/process",
                    json=request_data
                )
                
                if response.status_code != 200:
                    raise Exception(f"Agent API returned status {response.status_code}")
                
                result = response.json()
                return result
                
        except Exception as e:
            logger.error(f"Failed to call agent API: {e}")
            raise
    
    def _select_evaluation_method(self, category: TaskCategory) -> EvaluationMethod:
        """Intelligently select evaluation method based on task category"""
        
        # Web retrieval tasks: use rule-based evaluation for precise matching
        if category == TaskCategory.WEB_RETRIEVAL:
                return EvaluationMethod.RULE_BASED
        
        # Web + On-chain retrieval: use hybrid evaluation for cross-source verification
        elif category == TaskCategory.WEB_ONCHAIN_RETRIEVAL:
                return EvaluationMethod.HYBRID
        
        # On-chain retrieval: use LLM-based evaluation for technical analysis
        elif category == TaskCategory.ONCHAIN_RETRIEVAL:
                return EvaluationMethod.LLM_BASED
        
        # Default to hybrid evaluation
        return EvaluationMethod.HYBRID
    
    def _enhance_evaluation_with_expected_answer(self, evaluation_result: Dict[str, Any], 
                                               expected_answer: Optional[str], 
                                               agent_answer: str) -> Dict[str, Any]:
        """Enhance evaluation with detailed expected answer comparison"""
        if not expected_answer:
            return evaluation_result
        
        # Extract key information from expected answer
        expected_lower = expected_answer.lower().strip()
        agent_lower = agent_answer.lower().strip()
        
        # Calculate expected answer match score
        expected_score = 0.0
        expected_analysis = {}
        
        # 1. Exact match check
        if expected_lower in agent_lower:
            expected_score += 0.4
            expected_analysis["exact_match"] = True
        else:
            expected_analysis["exact_match"] = False
        
        # 2. Key information extraction and matching
        key_info_score = self._extract_and_match_key_info(expected_answer, agent_answer)
        expected_score += key_info_score * 0.3
        expected_analysis["key_info_score"] = key_info_score
        
        # 3. Factual accuracy check
        factual_score = self._check_factual_accuracy(expected_answer, agent_answer)
        expected_score += factual_score * 0.2
        expected_analysis["factual_score"] = factual_score
        
        # 4. Completeness check
        completeness_score = self._check_completeness(expected_answer, agent_answer)
        expected_score += completeness_score * 0.1
        expected_analysis["completeness_score"] = completeness_score
        
        # Adjust original score based on expected answer analysis
        original_score = evaluation_result.get("score", 0.0)
        expected_weight = 0.6  # Weight for expected answer analysis
        original_weight = 0.4  # Weight for original evaluation
        
        final_score = (original_score * original_weight + expected_score * expected_weight)
        final_score = min(1.0, max(0.0, final_score))
        
        # Enhanced reasoning
        original_reasoning = evaluation_result.get("reasoning", "")
        expected_reasoning = f"Expected answer analysis: {expected_score:.2f} (exact_match: {expected_analysis['exact_match']}, key_info: {expected_analysis['key_info_score']:.2f}, factual: {expected_analysis['factual_score']:.2f}, completeness: {expected_analysis['completeness_score']:.2f})"
        
        enhanced_reasoning = f"{original_reasoning}\n{expected_reasoning}"
        
        return {
            "score": final_score,
            "reasoning": enhanced_reasoning,
            "expected_answer_analysis": expected_analysis
        }
    
    def _extract_and_match_key_info(self, expected_answer: str, agent_answer: str) -> float:
        """Extract and match key information from expected answer"""
        score = 0.0
        
        # Extract dates, numbers, addresses, etc. from expected answer
        import re
        
        # Date patterns
        date_patterns = [
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD or YYYY-MM-DD
            r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',  # MM/DD/YYYY or MM-DD-YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',        # YYYY-MM-DD
        ]
        
        # Number patterns
        number_patterns = [
            r'\$[\d,]+\.?\d*',  # Dollar amounts
            r'\d+\.?\d*',       # General numbers
        ]
        
        # Address patterns
        address_patterns = [
            r'0x[a-fA-F0-9]{40}',  # Ethereum addresses
            r'0x[a-fA-F0-9]{64}',  # Transaction hashes
        ]
        
        all_patterns = date_patterns + number_patterns + address_patterns
        
        expected_matches = set()
        for pattern in all_patterns:
            matches = re.findall(pattern, expected_answer, re.IGNORECASE)
            expected_matches.update(matches)
        
        agent_matches = set()
        for pattern in all_patterns:
            matches = re.findall(pattern, agent_answer, re.IGNORECASE)
            agent_matches.update(matches)
        
        if expected_matches:
            match_ratio = len(expected_matches.intersection(agent_matches)) / len(expected_matches)
            score += match_ratio * 0.5
        
        # Check for key terms
        expected_terms = set(re.findall(r'\b\w+\b', expected_answer.lower()))
        agent_terms = set(re.findall(r'\b\w+\b', agent_answer.lower()))
        
        if expected_terms:
            term_match_ratio = len(expected_terms.intersection(agent_terms)) / len(expected_terms)
            score += term_match_ratio * 0.5
        
        return min(1.0, score)
    
    def _check_factual_accuracy(self, expected_answer: str, agent_answer: str) -> float:
        """Check factual accuracy against expected answer"""
        score = 0.0
        
        # Check for contradictory information
        expected_lower = expected_answer.lower()
        agent_lower = agent_answer.lower()
        
        # Look for negation patterns that might indicate contradiction
        negation_words = ['not', 'no', 'never', 'none', 'false', 'incorrect', 'wrong']
        
        expected_has_negation = any(word in expected_lower for word in negation_words)
        agent_has_negation = any(word in agent_lower for word in negation_words)
        
        if expected_has_negation == agent_has_negation:
            score += 0.3
        
        # Check for specific factual claims
        if 'true' in expected_lower and 'true' in agent_lower:
            score += 0.3
        elif 'false' in expected_lower and 'false' in agent_lower:
            score += 0.3
        
        # Check for numerical accuracy
        import re
        expected_numbers = re.findall(r'\d+\.?\d*', expected_answer)
        agent_numbers = re.findall(r'\d+\.?\d*', agent_answer)
        
        if expected_numbers and agent_numbers:
            # Check if any expected numbers appear in agent answer
            number_matches = sum(1 for num in expected_numbers if num in agent_answer)
            number_accuracy = number_matches / len(expected_numbers)
            score += number_accuracy * 0.4
        
        return min(1.0, score)
    
    def _check_completeness(self, expected_answer: str, agent_answer: str) -> float:
        """Check if agent answer covers all aspects of expected answer"""
        score = 0.0
        
        # Split expected answer into sentences/phrases
        expected_parts = [part.strip() for part in expected_answer.split('.') if part.strip()]
        agent_lower = agent_answer.lower()
        
        if expected_parts:
            covered_parts = 0
            for part in expected_parts:
                if part.lower() in agent_lower or any(word in agent_lower for word in part.split() if len(word) > 3):
                    covered_parts += 1
            
            coverage_ratio = covered_parts / len(expected_parts)
            score += coverage_ratio * 0.6
        
        # Check for key concepts coverage
        key_concepts = ['date', 'time', 'address', 'hash', 'number', 'amount', 'price', 'value']
        expected_concepts = [concept for concept in key_concepts if concept in expected_answer.lower()]
        agent_concepts = [concept for concept in key_concepts if concept in agent_answer.lower()]
        
        if expected_concepts:
            concept_coverage = len(set(expected_concepts).intersection(set(agent_concepts))) / len(expected_concepts)
            score += concept_coverage * 0.4
        
        return min(1.0, score)

    def _get_evaluator_for_task(self, task: EvaluationTask) -> Optional[BaseEvaluator]:
        """Select appropriate evaluator based on task type"""
        evaluation_method = task.evaluation_method
        
        if evaluation_method == EvaluationMethod.LLM_BASED and "llm" in self.evaluators:
            return self.evaluators["llm"]
        elif evaluation_method == EvaluationMethod.HYBRID and "hybrid" in self.evaluators:
            return self.evaluators["hybrid"]
        elif evaluation_method == EvaluationMethod.RULE_BASED:
            # Use simplified rule evaluation
            return None
        else:
            # Default to hybrid
            return self.evaluators.get("hybrid")
    
    async def _evaluate_response(self, task: EvaluationTask) -> Dict[str, Any]:
        """Evaluate Agent response"""
        try:
            agent_response = task.agent_response
            answer = agent_response.get("answer", "")
            confidence = agent_response.get("confidence", 0.0)
            
            # Try to use professional evaluator
            evaluator = self._get_evaluator_for_task(task)
            
            if evaluator:
                try:
                    # Use professional evaluator for evaluation
                    evaluation_result = await evaluator.evaluate(
                        question=task.question,
                        agent_response=answer,
                        expected_answer=task.expected_answer,
                        context=task.context
                    )
                    
                    # Enhance evaluation with detailed expected answer comparison
                    enhanced_result = self._enhance_evaluation_with_expected_answer(
                        evaluation_result, task.expected_answer, answer
                    )

                    evaluator_details = evaluation_result.get("details", {}) or {}
                    
                    return {
                        "evaluation_score": enhanced_result.get("score", evaluation_result.get("score", 0.0)),
                        "evaluation_reasoning": enhanced_result.get("reasoning", evaluation_result.get("reasoning", "")),
                        "confidence": confidence,
                        "details": {
                            "evaluation_method": task.evaluation_method,
                            "response_length": len(answer),
                            "confidence_score": confidence,
                            "processing_time": agent_response.get("processing_time", 0.0),
                            "tools_used": agent_response.get("tools_used", []),
                            "evaluator_details": evaluator_details,
                            "dimension_breakdown": evaluator_details.get("dimension_breakdown", {}),
                            "expected_answer_analysis": enhanced_result.get("expected_answer_analysis", {})
                        }
                    }
                    
                except Exception as e:
                    logger.warning(f"Professional evaluator failed, using fallback: {e}")
            
            # Use simplified evaluation logic as fallback
            return await self._simple_evaluate_response(task)
            
        except Exception as e:
            logger.error(f"Failed to evaluate response: {e}")
            raise
    
    async def _simple_evaluate_response(self, task: EvaluationTask) -> Dict[str, Any]:
        """Enhanced simplified evaluation logic with better discrimination"""
        agent_response = task.agent_response
        answer = agent_response.get("answer", "")
        confidence = agent_response.get("confidence", 0.0)
        
        # Use multi-dimensional evaluation if expected answer is available
        if task.expected_answer:
            # Multi-dimensional evaluation
            dimensions = self._multi_dimensional_evaluation(task.expected_answer, answer)
            
            # Calculate weighted score with stricter standards
            weights = {
                "factual_accuracy": 0.4,  # Most important
                "completeness": 0.25,
                "precision": 0.15,
                "relevance": 0.15,
                "conciseness": 0.05
            }
            
            # Calculate weighted score
            weighted_score = sum(dimensions[key] * weights[key] for key in weights)
            
            # Apply stricter base score (reduce from 0.4-0.5 to 0.2-0.3)
            base_score = weighted_score * 0.6  # Reduce base score significantly
            
            # Apply confidence adjustment (reduced impact)
            confidence_factor = 0.8 + (confidence * 0.4)  # Range: 0.8-1.2
            final_score = base_score * confidence_factor
            
            # Apply failure penalty if task failed
            if task.status == EvaluationStatus.FAILED:
                final_score *= 0.3  # Severe penalty for failed tasks
            
            reasoning = f"Multi-dimensional evaluation: factual_accuracy={dimensions['factual_accuracy']:.2f}, completeness={dimensions['completeness']:.2f}, precision={dimensions['precision']:.2f}, relevance={dimensions['relevance']:.2f}, conciseness={dimensions['conciseness']:.2f}; "
            reasoning += f"Base score: {base_score:.2f}, Confidence factor: {confidence_factor:.2f}; "
            
        else:
            # Fallback to basic evaluation with stricter standards
            score = 0.0
            reasoning = ""
            
            # Check for repetitive/meaningless content
            repetitiveness_penalty = _check_repetitiveness(answer)
            if repetitiveness_penalty > 0.5:
                score -= repetitiveness_penalty * 0.3
                reasoning += f"Repetitive content detected (penalty: {repetitiveness_penalty:.2f}); "
            
            # Check for meaningful content vs fluff
            meaningful_content_score = _check_meaningful_content(answer)
            score += meaningful_content_score * 0.3  # Reduced weight
            reasoning += f"Meaningful content: {meaningful_content_score:.2f}; "
            
            # Category-based evaluation with reduced weight
            category_score = _category_based_evaluation(task, answer)
            score += category_score * 0.2  # Reduced weight
            reasoning += f"Category-based evaluation: {category_score:.2f}; "
            
            # Confidence adjustment (reduced impact)
            if confidence > 0.8:
                score += 0.05  # Reduced bonus
                reasoning += "High confidence; "
            elif confidence < 0.3:
                score -= 0.05  # Reduced penalty
                reasoning += "Low confidence; "
            
            # Apply stricter base score
            final_score = score * 0.5  # Reduce base score significantly
            
            # Apply failure penalty if task failed
            if task.status == EvaluationStatus.FAILED:
                final_score *= 0.3  # Severe penalty for failed tasks
        
        # Ensure score is between 0 and 1
        final_score = min(1.0, max(0.0, final_score))
        
        return {
            "evaluation_score": final_score,
            "evaluation_reasoning": reasoning,
            "confidence": confidence,
            "details": {
                "evaluation_method": "simplified_enhanced_strict",
                "response_length": len(answer),
                "confidence_score": confidence,
                "processing_time": agent_response.get("processing_time", 0.0),
                "tools_used": agent_response.get("tools_used", []),
                "multi_dimensional_scores": dimensions if task.expected_answer else None
            }
        }
    
    async def evaluate_agent(
        self,
        question: str,
        agent_metadata: AgentMetadata,
        context: Optional[Dict[str, Any]] = None,
        category: TaskCategory = TaskCategory.WEB_RETRIEVAL,
        evaluation_method: Optional[EvaluationMethod] = None,
        expected_answer: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate single Agent"""
        
        # Select evaluation method if not provided
        if evaluation_method is None:
            evaluation_method = self._select_evaluation_method(category)
        
        # Create task
        task_id = self._generate_task_id()
        task = EvaluationTask(
            task_id=task_id,
            question=question,
            agent_metadata=agent_metadata,
            context=context or {},
            category=category,
            evaluation_method=evaluation_method,
            expected_answer=expected_answer,
            status=EvaluationStatus.IN_PROGRESS,
            created_at=time.time()
        )
        
        self.tasks[task_id] = task
        
        try:
            # Call Agent API
            logger.info(f"Calling agent API for task {task_id}")
            agent_response = await self._call_agent_api(task)
            task.agent_response = agent_response
            
            # Evaluate response
            logger.info(f"Evaluating response for task {task_id}")
            evaluation_result = await self._evaluate_response(task)
            task.evaluation_result = evaluation_result
            
            # Complete task
            task.status = EvaluationStatus.COMPLETED
            task.completed_at = time.time()
            
            # Close Agent (if close_endpoint is configured)
            if task.agent_metadata.close_endpoint:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        await client.post(
                            task.agent_metadata.close_endpoint,
                            json={"reason": "Task completed"}
                        )
                    logger.info(f"Agent closed for task {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to close agent: {e}")
            
            # Return result
            return EvaluationResult(
                task_id=task_id,
                question=question,
                agent_response=agent_response.get("answer", ""),
                evaluation_score=evaluation_result["evaluation_score"],
                evaluation_reasoning=evaluation_result["evaluation_reasoning"],
                confidence=evaluation_result["confidence"],
                processing_time=agent_response.get("processing_time", 0.0),
                tools_used=agent_response.get("tools_used", []),
                metadata=agent_response.get("metadata", {}),
                status=EvaluationStatus.COMPLETED
            )
            
        except Exception as e:
            # Handle error
            task.status = EvaluationStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            logger.error(f"Task {task_id} failed: {e}")
            
            return EvaluationResult(
                task_id=task_id,
                question=question,
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
    
    async def evaluate_multiple_agents(
        self,
        question: str,
        agents: List[AgentMetadata],
        task_type: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """Evaluate multiple Agents"""
        
        tasks = []
        for agent in agents:
            task = self.evaluate_agent(question, agent, task_type, context)
            tasks.append(task)
        
        # Execute all evaluations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exception results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(EvaluationResult(
                    task_id=f"failed_{i}",
                    question=question,
                    agent_response="",
                    evaluation_score=0.0,
                    evaluation_reasoning="",
                    confidence=0.0,
                    processing_time=0.0,
                    tools_used=[],
                    metadata={},
                    status=EvaluationStatus.FAILED,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def evaluate_agent_with_dataset(
        self,
        agent_metadata: AgentMetadata,
        dataset_path: str,
        max_tasks: Optional[int] = None
    ) -> List[EvaluationResult]:
        """Evaluate Agent using dataset"""
        import csv
        
        # Load dataset
        tasks = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Map category string to enum
                    category_map = {
                        "web_retrieval": TaskCategory.WEB_RETRIEVAL,
                        "web_onchain_retrieval": TaskCategory.WEB_ONCHAIN_RETRIEVAL,
                        "onchain_retrieval": TaskCategory.ONCHAIN_RETRIEVAL
                    }
                    
                    # Map evaluation method string to enum
                    method_map = {
                        "rule_based": EvaluationMethod.RULE_BASED,
                        "llm_based": EvaluationMethod.LLM_BASED,
                        "hybrid": EvaluationMethod.HYBRID
                    }
                    
                    task_data = {
                        "question": row["question"],
                        "expected_answer": row["answer"],
                        "category": category_map.get(row["category"], TaskCategory.WEB_RETRIEVAL),
                        "evaluation_method": method_map.get(row["evaluation_method"], EvaluationMethod.HYBRID),
                        "context": {"dataset_id": row["id"]}
                    }
                    tasks.append(task_data)
                    
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        # Limit tasks if specified
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
        
        return results
    
    def _calculate_total_score(self, completed_tasks: List[EvaluationTask]) -> float:
        """Calculate weighted total score for all completed tasks with failure penalties"""
        if not completed_tasks:
            return 0.0
        
        # Separate successful and failed tasks
        successful_tasks = [task for task in completed_tasks if task.status == EvaluationStatus.COMPLETED]
        failed_tasks = [task for task in completed_tasks if task.status == EvaluationStatus.FAILED]
        
        # Calculate successful tasks score
        successful_score = 0.0
        total_weight = 0.0
        
        for task in successful_tasks:
            if task.evaluation_result:
                # Get base score
                base_score = task.evaluation_result.get("evaluation_score", 0.0)
                
                # Calculate task weight based on category
                task_weight = self._calculate_task_weight(task)
                
                # Apply confidence adjustment (reduced impact)
                confidence = task.agent_response.get("confidence", 0.5) if task.agent_response else 0.5
                confidence_factor = 0.8 + (confidence * 0.4)  # Range: 0.8-1.2
                
                # Apply expected answer bonus if available
                expected_answer_bonus = 1.0
                if task.expected_answer:
                    expected_answer_bonus = 1.1  # Reduced bonus: 10% instead of 20%
                
                # Calculate weighted score
                weighted_score = base_score * task_weight * confidence_factor * expected_answer_bonus
                
                successful_score += weighted_score
                total_weight += task_weight
        
        # Calculate base score from successful tasks
        if total_weight == 0:
            base_score = 0.0
        else:
            base_score = successful_score / total_weight
        
        # Apply severe failure penalty
        failure_penalty = 0.0
        if failed_tasks:
            # Each failed task reduces score by 0.2
            failure_penalty = len(failed_tasks) * 0.2
            logger.warning(f"Failed tasks penalty: {failure_penalty:.2f} for {len(failed_tasks)} failed tasks")
        
        # Calculate final score with failure penalty
        final_score = base_score - failure_penalty
        
        # Ensure score is between 0 and 1
        final_score = min(1.0, max(0.0, final_score))
        
        logger.info(f"Total score calculation: successful_score={base_score:.3f}, failure_penalty={failure_penalty:.3f}, final_score={final_score:.3f}")
        
        return final_score
    
    def _calculate_task_weight(self, task: EvaluationTask) -> float:
        """Calculate weight for a task based on its characteristics"""
        base_weight = 1.0
        
        # Category-based weights
        category_weights = {
            TaskCategory.WEB_RETRIEVAL: 1.0,
            TaskCategory.WEB_ONCHAIN_RETRIEVAL: 1.3,  # More complex, higher weight
            TaskCategory.ONCHAIN_RETRIEVAL: 1.2       # Technical, higher weight
        }
        base_weight *= category_weights.get(task.category, 1.0)
        
        # Evaluation method weights
        method_weights = {
            EvaluationMethod.RULE_BASED: 0.9,    # Simpler evaluation
            EvaluationMethod.LLM_BASED: 1.1,     # More sophisticated
            EvaluationMethod.HYBRID: 1.2          # Most comprehensive
        }
        base_weight *= method_weights.get(task.evaluation_method, 1.0)
        
        # Expected answer bonus
        if task.expected_answer:
            base_weight *= 1.1  # 10% bonus for tasks with expected answers
        
        return base_weight
    
    def _semantic_similarity_evaluation(self, expected_answer: str, agent_answer: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        if not self.semantic_model or not expected_answer or not agent_answer:
            return 0.0
        
        try:
            # Get embeddings
            embeddings = self.semantic_model.encode([expected_answer, agent_answer])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def _multi_dimensional_evaluation(self, expected_answer: str, agent_answer: str) -> Dict[str, float]:
        """Multi-dimensional evaluation of agent response"""
        if not expected_answer or not agent_answer:
            return {
                "factual_accuracy": 0.0,
                "completeness": 0.0,
                "precision": 0.0,
                "relevance": 0.0,
                "conciseness": 0.0
            }
        
        # 1. Factual Accuracy
        factual_accuracy = self._evaluate_factual_accuracy(expected_answer, agent_answer)
        
        # 2. Completeness
        completeness = self._evaluate_completeness(expected_answer, agent_answer)
        
        # 3. Precision
        precision = self._evaluate_precision(expected_answer, agent_answer)
        
        # 4. Relevance
        relevance = self._evaluate_relevance(expected_answer, agent_answer)
        
        # 5. Conciseness
        conciseness = self._evaluate_conciseness(agent_answer)
        
        return {
            "factual_accuracy": factual_accuracy,
            "completeness": completeness,
            "precision": precision,
            "relevance": relevance,
            "conciseness": conciseness
        }
    
    def _evaluate_factual_accuracy(self, expected_answer: str, agent_answer: str) -> float:
        """Evaluate factual accuracy"""
        # Extract key facts from expected answer
        expected_facts = self._extract_facts(expected_answer)
        agent_facts = self._extract_facts(agent_answer)
        
        if not expected_facts:
            return 0.5  # Neutral if no facts to check
        
        # Check for contradictory information
        contradictions = self._check_contradictions(expected_facts, agent_facts)
        contradiction_penalty = contradictions * 0.3
        
        # Check for correct facts
        correct_facts = self._count_correct_facts(expected_facts, agent_facts)
        accuracy_score = correct_facts / len(expected_facts)
        
        final_score = max(0.0, accuracy_score - contradiction_penalty)
        return min(1.0, final_score)
    
    def _evaluate_completeness(self, expected_answer: str, agent_answer: str) -> float:
        """Evaluate completeness of response"""
        expected_lower = expected_answer.lower()
        agent_lower = agent_answer.lower()
        
        # Check if key information is covered
        key_terms = self._extract_key_terms(expected_answer)
        covered_terms = sum(1 for term in key_terms if term in agent_lower)
        
        if not key_terms:
            return 0.5
        
        completeness_score = covered_terms / len(key_terms)
        return min(1.0, completeness_score)
    
    def _evaluate_precision(self, expected_answer: str, agent_answer: str) -> float:
        """Evaluate precision - how much of the answer is relevant"""
        # Use semantic similarity as precision measure
        semantic_sim = self._semantic_similarity_evaluation(expected_answer, agent_answer)
        
        # Penalize for excessive length (verbose answers)
        length_penalty = 0.0
        if len(agent_answer) > len(expected_answer) * 3:
            length_penalty = 0.2
        
        precision_score = semantic_sim - length_penalty
        return max(0.0, min(1.0, precision_score))
    
    def _evaluate_relevance(self, expected_answer: str, agent_answer: str) -> float:
        """Evaluate relevance to the question"""
        # Check if answer addresses the same topic
        expected_topic = self._extract_topic(expected_answer)
        agent_topic = self._extract_topic(agent_answer)
        
        if expected_topic and agent_topic:
            topic_similarity = self._semantic_similarity_evaluation(expected_topic, agent_topic)
            return topic_similarity
        
        return 0.5  # Neutral if can't determine topic
    
    def _evaluate_conciseness(self, agent_answer: str) -> float:
        """Evaluate conciseness - penalize overly verbose answers"""
        if len(agent_answer) < 50:
            return 0.3  # Too short
        elif len(agent_answer) < 200:
            return 1.0  # Good length
        elif len(agent_answer) < 500:
            return 0.8  # Acceptable
        elif len(agent_answer) < 1000:
            return 0.6  # Too verbose
        else:
            return 0.3  # Very verbose
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual information from text"""
        import re
        
        facts = []
        
        # Extract dates
        dates = re.findall(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', text)
        facts.extend(dates)
        
        # Extract numbers/amounts
        amounts = re.findall(r'\$[\d,]+\.?\d*', text)
        facts.extend(amounts)
        
        # Extract addresses/hashes
        addresses = re.findall(r'0x[a-fA-F0-9]{40,64}', text)
        facts.extend(addresses)
        
        # Extract percentages
        percentages = re.findall(r'\d+\.?\d*%', text)
        facts.extend(percentages)
        
        return facts
    
    def _check_contradictions(self, expected_facts: List[str], agent_facts: List[str]) -> int:
        """Check for contradictions between expected and agent facts"""
        contradictions = 0
        
        for expected_fact in expected_facts:
            # Look for contradictory information
            if expected_fact.startswith('$'):
                # Check for different amounts
                agent_amounts = [fact for fact in agent_facts if fact.startswith('$')]
                if agent_amounts and expected_fact not in agent_amounts:
                    contradictions += 1
        
        return contradictions
    
    def _count_correct_facts(self, expected_facts: List[str], agent_facts: List[str]) -> int:
        """Count how many expected facts are present in agent facts"""
        correct_count = 0
        
        for expected_fact in expected_facts:
            if expected_fact in agent_facts:
                correct_count += 1
        
        return correct_count
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        import re
        
        # Extract important words (length > 3, not common words)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'a', 'an', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [word for word in words if len(word) > 3 and word not in common_words]
        
        return key_terms[:10]  # Limit to top 10 terms
    
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from text"""
        # Simple topic extraction - take first sentence or first 50 characters
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip()[:50]
        return text[:50]
    
    def get_task_status(self, task_id: str) -> Optional[EvaluationTask]:
        """Get task status"""
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[EvaluationTask]:
        """List all tasks"""
        return list(self.tasks.values())
    
    def export_results(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export evaluation results to specified output path"""
        import os
        
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        
        completed_tasks = [
            task for task in self.tasks.values()
            if task.status == EvaluationStatus.COMPLETED
        ]
        
        if format == "json":
            # Calculate total score with weighted evaluation
            total_score = self._calculate_total_score(completed_tasks)
            
            result_data = {
                "total_score": total_score,
                "total_tasks": len(completed_tasks),
                "successful_tasks": len(completed_tasks),
                "failed_tasks": len(self.tasks) - len(completed_tasks),
                "results": [
                    {
                        "task_id": task.task_id,
                        "question": task.question,
                        "category": task.category.value,
                        "evaluation_method": task.evaluation_method.value,
                        "agent_response": task.agent_response.get("answer", "") if task.agent_response else "",
                        "evaluation_score": task.evaluation_result.get("evaluation_score", 0.0) if task.evaluation_result else 0.0,
                        "evaluation_reasoning": task.evaluation_result.get("evaluation_reasoning", "") if task.evaluation_result else "",
                        "confidence": task.agent_response.get("confidence", 0.0) if task.agent_response else 0.0,
                        "processing_time": task.agent_response.get("processing_time", 0.0) if task.agent_response else 0.0,
                        "tools_used": task.agent_response.get("tools_used", []) if task.agent_response else [],
                        "status": task.status.value
                    }
                    for task in completed_tasks
                ]
            }
            
            # Save to file
            output_file = os.path.join(self.output_path, "evaluation_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            return result_data
        elif format == "csv":
            # Simplified CSV format
            csv_lines = ["task_id,question,category,evaluation_method,score,confidence,status"]
            for task in completed_tasks:
                score = task.evaluation_result.get("evaluation_score", 0.0) if task.evaluation_result else 0.0
                confidence = task.agent_response.get("confidence", 0.0) if task.agent_response else 0.0
                csv_lines.append(f"{task.task_id},{task.question[:50]},{task.category.value},{task.evaluation_method.value},{score},{confidence},{task.status.value}")
            
            csv_content = "\n".join(csv_lines)
            
            # Save to file
            output_file = os.path.join(self.output_path, "evaluation_results.csv")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            
            return csv_content
        else:
            raise ValueError(f"Unsupported format: {format}")

# Convenience functions
async def evaluate_agent(question: str, agent_url: str, llm_config: Dict[str, Any], output_path: str = "output", **kwargs) -> EvaluationResult:
    """Convenient Agent evaluation function"""
    # Create evaluator instance
    evaluator = DABEvaluator(llm_config, output_path)
    
    agent_metadata = AgentMetadata(
        url=agent_url,
        capabilities=kwargs.get("capabilities", ["general"]),
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

# Helper functions (moved outside class)
def _check_repetitiveness(answer: str) -> float:
    """Check for repetitive/meaningless content"""
    if len(answer) < 50:
        return 0.0
    
    # Split into sentences
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    if len(sentences) < 2:
        return 0.0
    
    # Check for repeated phrases
    repeated_phrases = 0
    for i, sentence in enumerate(sentences):
        for j, other_sentence in enumerate(sentences[i+1:], i+1):
            # Check for similar sentence structure
            words1 = set(sentence.lower().split())
            words2 = set(other_sentence.lower().split())
            if len(words1) > 3 and len(words2) > 3:
                similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                if similarity > 0.7:
                    repeated_phrases += 1
    
    # Check for repeated words
    words = answer.lower().split()
    word_counts = {}
    for word in words:
        if len(word) > 3:  # Only count meaningful words
            word_counts[word] = word_counts.get(word, 0) + 1
    
    repeated_words = sum(1 for count in word_counts.values() if count > 3)
    
    # Calculate repetitiveness score
    total_sentences = len(sentences)
    repetitiveness_score = (repeated_phrases + repeated_words) / max(total_sentences, 1)
    
    return min(1.0, repetitiveness_score)

def _check_meaningful_content(answer: str) -> float:
    """Check for meaningful content vs fluff"""
    if len(answer) < 20:
        return 0.0
    
    score = 0.0
    
    # Check for specific information
    specific_indicators = [
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # Dates
        r'\$[\d,]+\.?\d*',                # Money amounts
        r'0x[a-fA-F0-9]{40}',            # Addresses
        r'\d+\.?\d*%',                   # Percentages
        r'\d+\.?\d*',                    # Numbers
    ]
    
    import re
    specific_count = 0
    for pattern in specific_indicators:
        matches = re.findall(pattern, answer)
        specific_count += len(matches)
    
    if specific_count > 0:
        score += min(0.4, specific_count * 0.1)
    
    # Check for technical terms
    technical_terms = [
        'blockchain', 'ethereum', 'bitcoin', 'contract', 'transaction',
        'hash', 'address', 'token', 'defi', 'nft', 'smart contract',
        'protocol', 'consensus', 'mining', 'staking', 'liquidity'
    ]
    
    technical_count = sum(1 for term in technical_terms if term in answer.lower())
    if technical_count > 0:
        score += min(0.3, technical_count * 0.05)
    
    # Check for action words (indicating specific actions)
    action_words = [
        'analyze', 'calculate', 'determine', 'identify', 'verify',
        'confirm', 'check', 'validate', 'compute', 'derive'
    ]
    
    action_count = sum(1 for word in action_words if word in answer.lower())
    if action_count > 0:
        score += min(0.3, action_count * 0.1)
    
    return min(1.0, score)

def _enhanced_expected_answer_match(expected_answer: str, agent_answer: str) -> float:
    """Enhanced expected answer matching"""
    if not expected_answer:
        return 0.0
    
    score = 0.0
    
    # Exact match
    if expected_answer.lower().strip() in agent_answer.lower():
        score += 0.5
    
    # Key information extraction and matching
    key_info_score = _extract_and_match_key_info(expected_answer, agent_answer)
    score += key_info_score * 0.3
    
    # Factual accuracy
    factual_score = _check_factual_accuracy(expected_answer, agent_answer)
    score += factual_score * 0.2
    
    return min(1.0, score)

def _category_based_evaluation(task: EvaluationTask, answer: str) -> float:
    """Category-based evaluation when no expected answer - using quasi-exact match approach"""
    if not task.expected_answer:
        # If no expected answer, use basic content quality assessment
        return _assess_content_quality(answer)
    
    # Use quasi-exact match for rule-based evaluation
    return _quasi_exact_match(task.expected_answer, answer)

def _quasi_exact_match(expected_answer: str, agent_answer: str) -> float:
    """Quasi-exact match evaluation - more sophisticated than simple string matching"""
    if not expected_answer or not agent_answer:
        return 0.0
    
    expected_lower = expected_answer.lower().strip()
    agent_lower = agent_answer.lower().strip()
    
    # 1. Exact match (highest score)
    if expected_lower == agent_lower:
        return 1.0
    
    # 2. Substring match (high score)
    if expected_lower in agent_lower:
        return 0.9
    
    # 3. Key information extraction and matching
    key_info_score = _extract_and_match_key_info(expected_answer, agent_answer)
    if key_info_score > 0.8:
        return 0.8
    elif key_info_score > 0.6:
        return 0.7
    elif key_info_score > 0.4:
        return 0.5
    
    # 4. Semantic similarity (using word overlap)
    word_overlap_score = _calculate_word_overlap(expected_answer, agent_answer)
    if word_overlap_score > 0.7:
        return 0.6
    elif word_overlap_score > 0.5:
        return 0.4
    elif word_overlap_score > 0.3:
        return 0.2
    
    # 5. Very low similarity
    return 0.1

def _calculate_word_overlap(expected_answer: str, agent_answer: str) -> float:
    """Calculate word overlap similarity"""
    import re
    
    # Extract meaningful words (length > 2, not common words)
    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'a', 'an', 'this', 'that', 'these', 'those'}
    
    expected_words = set(re.findall(r'\b\w+\b', expected_answer.lower()))
    agent_words = set(re.findall(r'\b\w+\b', agent_answer.lower()))
    
    # Remove common words
    expected_words = {word for word in expected_words if len(word) > 2 and word not in common_words}
    agent_words = {word for word in agent_words if len(word) > 2 and word not in common_words}
    
    if not expected_words:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(expected_words.intersection(agent_words))
    union = len(expected_words.union(agent_words))
    
    return intersection / union if union > 0 else 0.0

def _assess_content_quality(answer: str) -> float:
    """Assess content quality when no expected answer is available"""
    if not answer or len(answer.strip()) < 10:
        return 0.0
    
    score = 0.0
    
    # Length assessment (not too short, not too verbose)
    length = len(answer.strip())
    if 50 <= length <= 500:
        score += 0.3
    elif 20 <= length < 50 or 500 < length <= 1000:
        score += 0.2
    elif length > 1000:
        score += 0.1  # Too verbose might indicate poor quality
    
    # Check for specific information indicators
    specific_indicators = [
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # Dates
        r'\$[\d,]+\.?\d*',                # Money amounts
        r'0x[a-fA-F0-9]{40,64}',         # Addresses/hashes
        r'\d+\.?\d*%',                   # Percentages
    ]
    
    import re
    specific_count = 0
    for pattern in specific_indicators:
        matches = re.findall(pattern, answer)
        specific_count += len(matches)
    
    if specific_count > 0:
        score += min(0.4, specific_count * 0.1)
    
    # Check for technical terms
    technical_terms = [
        'blockchain', 'ethereum', 'bitcoin', 'contract', 'transaction',
        'hash', 'address', 'token', 'defi', 'nft', 'smart contract',
        'protocol', 'consensus', 'mining', 'staking', 'liquidity'
    ]
    
    technical_count = sum(1 for term in technical_terms if term in answer.lower())
    if technical_count > 0:
        score += min(0.3, technical_count * 0.05)
    
    return min(1.0, score)

def _extract_and_match_key_info(expected_answer: str, agent_answer: str) -> float:
    """Extract and match key information from expected answer"""
    score = 0.0
    
    # Extract dates, numbers, addresses, etc. from expected answer
    import re
    
    # Date patterns
    date_patterns = [
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD or YYYY-MM-DD
        r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',  # MM/DD/YYYY or MM-DD-YYYY
        r'\d{4}-\d{1,2}-\d{1,2}',        # YYYY-MM-DD
    ]
    
    # Number patterns
    number_patterns = [
        r'\$[\d,]+\.?\d*',  # Dollar amounts
        r'\d+\.?\d*',       # General numbers
    ]
    
    # Address patterns
    address_patterns = [
        r'0x[a-fA-F0-9]{40}',  # Ethereum addresses
        r'0x[a-fA-F0-9]{64}',  # Transaction hashes
    ]
    
    all_patterns = date_patterns + number_patterns + address_patterns
    
    expected_matches = set()
    for pattern in all_patterns:
        matches = re.findall(pattern, expected_answer, re.IGNORECASE)
        expected_matches.update(matches)
    
    agent_matches = set()
    for pattern in all_patterns:
        matches = re.findall(pattern, agent_answer, re.IGNORECASE)
        agent_matches.update(matches)
    
    if expected_matches:
        match_ratio = len(expected_matches.intersection(agent_matches)) / len(expected_matches)
        score += match_ratio * 0.5
    
    # Check for key terms
    expected_terms = set(re.findall(r'\b\w+\b', expected_answer.lower()))
    agent_terms = set(re.findall(r'\b\w+\b', agent_answer.lower()))
    
    if expected_terms:
        term_match_ratio = len(expected_terms.intersection(agent_terms)) / len(expected_terms)
        score += term_match_ratio * 0.5
    
    return min(1.0, score)

def _check_factual_accuracy(expected_answer: str, agent_answer: str) -> float:
    """Check factual accuracy against expected answer"""
    score = 0.0
    
    # Check for contradictory information
    expected_lower = expected_answer.lower()
    agent_lower = agent_answer.lower()
    
    # Look for negation patterns that might indicate contradiction
    negation_words = ['not', 'no', 'never', 'none', 'false', 'incorrect', 'wrong']
    
    expected_has_negation = any(word in expected_lower for word in negation_words)
    agent_has_negation = any(word in agent_lower for word in negation_words)
    
    if expected_has_negation == agent_has_negation:
        score += 0.3
    
    # Check for specific factual claims
    if 'true' in expected_lower and 'true' in agent_lower:
        score += 0.3
    elif 'false' in expected_lower and 'false' in agent_lower:
        score += 0.3
    
    # Check for numerical accuracy
    import re
    expected_numbers = re.findall(r'\d+\.?\d*', expected_answer)
    agent_numbers = re.findall(r'\d+\.?\d*', agent_answer)
    
    if expected_numbers and agent_numbers:
        # Check if any expected numbers appear in agent answer
        number_matches = sum(1 for num in expected_numbers if num in agent_answer)
        number_accuracy = number_matches / len(expected_numbers)
        score += number_accuracy * 0.4
    
    return min(1.0, score)
