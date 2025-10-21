"""
Hybrid Evaluator for DAB Evaluation SDK
Enhanced hybrid evaluator with improved scoring system
"""

from typing import Dict, Any, List, Optional
from .base_evaluator import BaseEvaluator
from .llm_evaluator import LLMEvaluator
from .enhanced_scoring import EnhancedScoringSystem, ScoringMethod


class HybridEvaluator(BaseEvaluator):
    """Hybrid evaluator"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        config = config or {}

        self.use_llm_evaluation = config.get("use_llm_evaluation", True)
        self.llm_evaluation_threshold = config.get("llm_evaluation_threshold", 0.5)
        self.rule_based_weight = config.get("rule_based_weight", 0.3)
        self.llm_based_weight = config.get("llm_based_weight", 0.7)
        self.use_enhanced_scoring = config.get("use_enhanced_scoring", True)

        llm_config = config.get("llm_config", {})
        self.llm_evaluator = LLMEvaluator(llm_config)
        self.enhanced_scoring = EnhancedScoringSystem()

    async def evaluate(
        self,
        question: str,
        agent_response: str,
        expected_answer: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enhanced hybrid evaluation of Agent response"""

        rule_score, enhanced_reasoning = self._evaluate_rule_component(
            question, agent_response, expected_answer, context
        )

        llm_score, llm_reasoning = await self._evaluate_llm_component(
            question, agent_response, expected_answer, context
        )

        effective_llm_weight = self.llm_based_weight
        threshold_note = ""
        if self.use_llm_evaluation and llm_score < self.llm_evaluation_threshold:
            effective_llm_weight = 0.0
            threshold_note = (
                f"LLM score {llm_score:.2f} below threshold {self.llm_evaluation_threshold:.2f}, ignored in final mix."
            )

        weight_sum = self.rule_based_weight + effective_llm_weight
        if weight_sum == 0:
            final_score = 0.0
        else:
            final_score = (
                rule_score * self.rule_based_weight + llm_score * effective_llm_weight
            ) / weight_sum

        reasoning_lines = [
            f"Rule-based evaluation: {rule_score:.2f} (weight: {self.rule_based_weight:.2f})",
            f"LLM evaluation: {llm_score:.2f} (weight: {effective_llm_weight:.2f})",
            f"Comprehensive score: {final_score:.2f}",
        ]
        if threshold_note:
            reasoning_lines.append(threshold_note)
        if self.use_enhanced_scoring and expected_answer:
            reasoning_lines.append(f"Enhanced scoring details: {enhanced_reasoning}")

        return {
            "score": min(1.0, max(0.0, final_score)),
            "reasoning": "\n".join(reasoning_lines),
            "details": {
                "rule_based_score": rule_score,
                "llm_based_score": llm_score,
                "rule_based_weight": self.rule_based_weight,
                "llm_based_weight": self.llm_based_weight,
                "effective_llm_weight": effective_llm_weight,
                "llm_reasoning": llm_reasoning,
                "enhanced_scoring": self.use_enhanced_scoring,
                "enhanced_reasoning": enhanced_reasoning,
                "evaluation_method": "enhanced_hybrid",
            },
        }

    def _evaluate_rule_component(
        self,
        question: str,
        agent_response: str,
        expected_answer: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> (float, str):
        if self.use_enhanced_scoring and expected_answer:
            try:
                scoring_method = self._select_optimal_scoring_method(question, expected_answer, context)
                enhanced_result = self.enhanced_scoring.score_answer(
                    expected=expected_answer,
                    agent=agent_response,
                    method=scoring_method,
                    context={"question": question, "context": context or {}},
                )
                score = max(0.0, min(1.0, enhanced_result.score * enhanced_result.confidence))
                return score, enhanced_result.reasoning
            except Exception as exc:
                fallback_score = self._rule_based_evaluation(
                    question, agent_response, expected_answer, context
                )
                return fallback_score, f"Enhanced scoring failed, fallback triggered: {exc}"

        score = self._rule_based_evaluation(question, agent_response, expected_answer, context)
        return score, "Enhanced scoring disabled; using traditional rules"

    async def _evaluate_llm_component(
        self,
        question: str,
        agent_response: str,
        expected_answer: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> (float, str):
        if not self.use_llm_evaluation:
            return 0.0, "LLM evaluation disabled"

        try:
            llm_result = await self.llm_evaluator.evaluate(
                question, agent_response, expected_answer, context
            )
            score = float(llm_result.get("score", 0.0))
            reasoning = llm_result.get("reasoning", "")
            return max(0.0, min(1.0, score)), reasoning
        except Exception as exc:
            return 0.0, f"LLM evaluation failed: {exc}"

    def _rule_based_evaluation(
        self,
        question: str,
        agent_response: str,
        expected_answer: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Rule-based evaluation fallback"""

        score = 0.0
        response_lower = (agent_response or "").lower()

        if len(agent_response or "") > 100:
            score += 0.2
        elif len(agent_response or "") > 50:
            score += 0.1

        web3_keywords = [
            "blockchain",
            "ethereum",
            "smart contract",
            "defi",
            "nft",
            "web3",
            "crypto",
            "bitcoin",
        ]
        if any(keyword in response_lower for keyword in web3_keywords):
            score += 0.2

        tech_keywords = ["analysis", "technology", "protocol", "algorithm", "mechanism", "architecture"]
        if any(keyword in response_lower for keyword in tech_keywords):
            score += 0.2

        detail_keywords = ["specific", "detailed", "steps", "method", "principle", "implementation"]
        if any(keyword in response_lower for keyword in detail_keywords):
            score += 0.2

        if expected_answer:
            expected_lower = expected_answer.lower()
            if expected_lower in response_lower:
                score += 0.2

        if context:
            category = context.get("category", "").lower()
            if category == "web_retrieval" and any(
                keyword in response_lower for keyword in ["source", "reference", "according"]
            ):
                score += 0.1
            elif category == "onchain_retrieval" and any(
                keyword in response_lower for keyword in ["contract", "address", "hash", "0x"]
            ):
                score += 0.1

        return min(1.0, score)

    def _select_optimal_scoring_method(
        self,
        question: str,
        expected_answer: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ScoringMethod:
        question_text = " ".join(
            filter(None, [question or "", expected_answer or "", (context or {}).get("category") or ""])
        ).lower()

        if any(keyword in question_text for keyword in ["date", "when", "time", "timestamp"]):
            return ScoringMethod.FORMAT_STRICT
        if any(keyword in question_text for keyword in ["price", "amount", "value", "cost", "quantity", "total"]):
            return ScoringMethod.FORMAT_STRICT
        if any(keyword in question_text for keyword in ["which", "where", "list", "compare"]):
            return ScoringMethod.CONTENT_FOCUSED
        if any(
            keyword in question_text
            for keyword in ["how", "what is", "explain", "describe"]
        ):
            return ScoringMethod.CONTENT_FOCUSED
        return ScoringMethod.HYBRID_BALANCED

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
            "llm_based_evaluation",
            "enhanced_scoring",
            "format_normalization",
            "semantic_similarity",
            "content_focused_evaluation",
            "multi_dimensional_scoring",
        ]
