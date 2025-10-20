"""
Hybrid Evaluator for DAB Evaluation SDK
Hybrid evaluator for DAB Evaluation SDK
"""

from typing import Dict, Any, List, Optional
from .base_evaluator import BaseEvaluator
from .llm_evaluator import LLMEvaluator

class HybridEvaluator(BaseEvaluator):
    """Hybrid evaluator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.use_llm_evaluation = config.get("use_llm_evaluation", True) if config else True
        self.llm_evaluation_threshold = config.get("llm_evaluation_threshold", 0.5) if config else 0.5
        self.rule_based_weight = config.get("rule_based_weight", 0.3) if config else 0.3
        self.llm_based_weight = config.get("llm_based_weight", 0.7) if config else 0.7
        
        # Initialize LLM evaluator
        llm_config = config.get("llm_config", {}) if config else {}
        self.llm_evaluator = LLMEvaluator(llm_config)
    
    async def evaluate(self, 
                      question: str,
                      agent_response: str,
                      expected_answer: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Hybrid evaluation of Agent response"""
        
        # Rule-based evaluation
        rule_score = self._rule_based_evaluation(question, agent_response, expected_answer, context)
        
        # LLM evaluation
        llm_score = 0.0
        llm_reasoning = ""
        if self.use_llm_evaluation:
            try:
                llm_result = await self.llm_evaluator.evaluate(
                    question, agent_response, expected_answer, context
                )
                llm_score = llm_result.get("score", 0.0)
                llm_reasoning = llm_result.get("reasoning", "")
            except Exception as e:
                llm_score = 0.0
                llm_reasoning = f"LLM evaluation failed: {str(e)}"
        
        # Calculate comprehensive score
        final_score = (rule_score * self.rule_based_weight + 
                      llm_score * self.llm_based_weight)
        
        # Build comprehensive evaluation reasoning
        reasoning = f"Rule-based evaluation: {rule_score:.2f} (weight: {self.rule_based_weight}); "
        reasoning += f"LLM evaluation: {llm_score:.2f} (weight: {self.llm_based_weight}); "
        reasoning += f"Comprehensive score: {final_score:.2f}"
        
        return {
            "score": min(1.0, max(0.0, final_score)),
            "reasoning": reasoning,
            "details": {
                "rule_based_score": rule_score,
                "llm_based_score": llm_score,
                "rule_based_weight": self.rule_based_weight,
                "llm_based_weight": self.llm_based_weight,
                "llm_reasoning": llm_reasoning,
                "evaluation_method": "hybrid"
            }
        }
    
    def _rule_based_evaluation(self, 
                              question: str, 
                              agent_response: str, 
                              expected_answer: Optional[str] = None,
                              context: Optional[Dict[str, Any]] = None) -> float:
        """Rule-based evaluation"""
        
        score = 0.0
        
        # Length evaluation
        if len(agent_response) > 100:
            score += 0.2
        elif len(agent_response) > 50:
            score += 0.1
        
        # Keyword matching evaluation
        question_lower = question.lower()
        response_lower = agent_response.lower()
        
        # Web3 related keywords
        web3_keywords = ["blockchain", "ethereum", "smart contract", "defi", "nft", "web3", "crypto", "bitcoin"]
        if any(keyword in response_lower for keyword in web3_keywords):
            score += 0.2
        
        # Technical keywords
        tech_keywords = ["analysis", "technology", "protocol", "algorithm", "mechanism", "architecture"]
        if any(keyword in response_lower for keyword in tech_keywords):
            score += 0.2
        
        # Detail level keywords
        detail_keywords = ["specific", "detailed", "steps", "method", "principle", "implementation"]
        if any(keyword in response_lower for keyword in detail_keywords):
            score += 0.2
        
        # Expected answer matching
        if expected_answer:
            expected_lower = expected_answer.lower()
            if expected_lower in response_lower:
                score += 0.2
        
        # Context matching
        if context:
            category = context.get("category", "").lower()
            if category == "web_retrieval" and any(keyword in response_lower for keyword in ["source", "reference", "according"]):
                score += 0.1
            elif category == "onchain_retrieval" and any(keyword in response_lower for keyword in ["contract", "address", "hash", "0x"]):
                score += 0.1
        
        return min(1.0, score)
    
    def get_capabilities(self) -> List[str]:
        """Get evaluator capabilities"""
        return [
            "web3_analysis",
            "blockchain_exploration",
            "defi_analysis", 
            "nft_evaluation",
            "smart_contract",
            "technical_knowledge",
            "hybrid_evaluation",
            "rule_based_evaluation",
            "llm_based_evaluation"
        ]
