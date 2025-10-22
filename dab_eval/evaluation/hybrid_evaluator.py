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

        weight_context = self._calculate_dynamic_weights(rule_score, llm_score)
        effective_llm_weight = weight_context["llm_weight"]
        threshold_note = weight_context["threshold_note"]

        weight_sum = weight_context["rule_weight"] + effective_llm_weight
        if weight_sum == 0:
            final_score = 0.0
        else:
            final_score = (
                rule_score * weight_context["rule_weight"] + llm_score * effective_llm_weight
            ) / weight_sum

        reasoning_lines = [
            f"Rule-based evaluation: {rule_score:.2f} (weight: {weight_context['rule_weight']:.2f})",
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
                "rule_based_weight": weight_context["rule_weight"],
                "llm_based_weight": self.llm_based_weight,
                "effective_llm_weight": effective_llm_weight,
                "llm_reasoning": llm_reasoning,
                "enhanced_scoring": self.use_enhanced_scoring,
                "enhanced_reasoning": enhanced_reasoning,
                "weighting_notes": weight_context["notes"],
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
            confidence = float(llm_result.get("details", {}).get("confidence", 1.0))
            adjusted_score = max(0.0, min(1.0, score))
            return adjusted_score * min(1.0, max(0.0, confidence)), reasoning
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

        if not agent_response:
            return 0.0

        response_lower = agent_response.lower()
        response_norm = self.enhanced_scoring.normalize_text(agent_response)
        expected_norm = self.enhanced_scoring.normalize_text(expected_answer) if expected_answer else ""

        score = 0.0
        penalties = 0.0

        # Length and structure heuristics
        response_length = len(agent_response.strip())
        if response_length >= 120:
            score += 0.15
        elif response_length >= 60:
            score += 0.1
        elif response_length < 25:
            penalties += 0.1

        sentences = [s.strip() for s in agent_response.split(".") if s.strip()]
        if len(sentences) >= 2:
            score += 0.05

        # Domain specific terms
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
            score += 0.15

        tech_keywords = ["analysis", "technology", "protocol", "algorithm", "mechanism", "architecture"]
        if any(keyword in response_lower for keyword in tech_keywords):
            score += 0.1

        detail_keywords = ["specific", "detailed", "steps", "method", "principle", "implementation"]
        if any(keyword in response_lower for keyword in detail_keywords):
            score += 0.1

        # Key information overlap
        if expected_answer:
            expected_info = self.enhanced_scoring.extract_key_information(expected_answer)
            agent_info = self.enhanced_scoring.extract_key_information(agent_response)
            if expected_info:
                matches = sum(
                    1
                    for info in expected_info
                    if any(self.enhanced_scoring.compare_key_information(info, candidate) for candidate in agent_info)
                )
                score += 0.4 * (matches / len(expected_info))

            if expected_norm and expected_norm in response_norm:
                score += 0.15
            else:
                penalties += 0.1

        # Context-aware adjustments
        question_lower = (question or "").lower()
        is_numeric_question = any(keyword in question_lower for keyword in ["price", "amount", "value", "cost", "quantity", "total"])
        is_temporal_question = any(keyword in question_lower for keyword in ["date", "when", "time", "timestamp"])

        if context:
            category = context.get("category", "").lower()
            if category == "web_retrieval" and any(
                keyword in response_lower for keyword in ["source", "reference", "according", "cited"]
            ):
                score += 0.1
            elif category == "onchain_retrieval" and any(
                keyword in response_lower for keyword in ["contract", "address", "hash", "0x", "transaction"]
            ):
                score += 0.1

        if is_numeric_question:
            digits_in_response = any(char.isdigit() for char in agent_response)
            if digits_in_response:
                score += 0.1
            else:
                penalties += 0.1

        if is_temporal_question:
            if any(token in response_lower for token in ["202", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]):
                score += 0.1
            else:
                penalties += 0.1

        # Penalties for uncertain or contradictory language
        uncertainty_markers = ["not sure", "no idea", "unknown", "cannot", "unable", "uncertain", "n/a"]
        if any(marker in response_lower for marker in uncertainty_markers):
            penalties += 0.15

        contradiction_markers = ["does not", "cannot be", "no evidence", "incorrect", "wrong"]
        if any(marker in response_lower for marker in contradiction_markers) and expected_answer:
            penalties += 0.1

        final_score = max(0.0, min(1.0, score - penalties))
        return final_score

    def _calculate_dynamic_weights(self, rule_score: float, llm_score: float) -> Dict[str, Any]:
        """Adjust weights based on confidence and threshold signals."""

        notes: List[str] = []
        rule_weight = self.rule_based_weight
        llm_weight = self.llm_based_weight
        threshold_note = ""

        if not self.use_llm_evaluation:
            return {
                "rule_weight": 1.0,
                "llm_weight": 0.0,
                "threshold_note": "LLM evaluation disabled; full weight on rule component.",
                "notes": notes,
            }

        if llm_score < self.llm_evaluation_threshold:
            llm_weight = 0.0
            threshold_note = (
                f"LLM score {llm_score:.2f} < threshold {self.llm_evaluation_threshold:.2f}; ignored."
            )
        elif llm_score >= 0.85:
            notes.append("High-confidence LLM score detected; boosting LLM weight.")
            llm_weight = min(1.0, llm_weight + 0.1)
            rule_weight = max(0.0, rule_weight - 0.1)

        if rule_score < 0.3 and llm_weight > 0:
            notes.append("Low rule score detected; shifting weight towards LLM.")
            llm_weight = min(1.0, llm_weight + 0.05)
            rule_weight = max(0.0, rule_weight - 0.05)

        normalization_factor = rule_weight + llm_weight
        if normalization_factor == 0:
            rule_weight = 0.0
            llm_weight = 0.0
        else:
            rule_weight /= normalization_factor
            llm_weight /= normalization_factor

        return {
            "rule_weight": rule_weight,
            "llm_weight": llm_weight,
            "threshold_note": threshold_note,
            "notes": notes,
        }

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
