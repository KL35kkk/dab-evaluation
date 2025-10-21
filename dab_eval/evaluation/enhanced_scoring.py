"""
Enhanced Scoring System for DAB Evaluation SDK
Addresses format sensitivity and improves scoring accuracy.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ScoringMethod(Enum):
    """Available scoring strategies."""
    FORMAT_STRICT = "format_strict"          # Strict format comparison
    SEMANTIC_FLEXIBLE = "semantic_flexible"  # Semantic-focused comparison
    HYBRID_BALANCED = "hybrid_balanced"      # Balanced blend of signals
    CONTENT_FOCUSED = "content_focused"      # Content-oriented comparison


@dataclass
class ScoringResult:
    """Container for scoring results."""
    score: float
    confidence: float
    method: ScoringMethod
    breakdown: Dict[str, float]
    reasoning: str
    suggestions: List[str]


class EnhancedScoringSystem:
    """Scoring helper that blends multiple comparison strategies."""

    def __init__(self):
        self.scoring_methods = {
            ScoringMethod.FORMAT_STRICT: self._format_strict_scoring,
            ScoringMethod.SEMANTIC_FLEXIBLE: self._semantic_flexible_scoring,
            ScoringMethod.HYBRID_BALANCED: self._hybrid_balanced_scoring,
            ScoringMethod.CONTENT_FOCUSED: self._content_focused_scoring,
        }

    def score_answer(
        self,
        expected: str,
        agent: str,
        method: ScoringMethod = ScoringMethod.HYBRID_BALANCED,
        context: Optional[Dict[str, Any]] = None,
    ) -> ScoringResult:
        """Score an answer using the selected method."""
        if not expected or not agent:
            return ScoringResult(
                score=0.0,
                confidence=0.0,
                method=method,
                breakdown={},
                reasoning="Empty expected or agent answer",
                suggestions=["Provide valid expected and agent answers"],
            )

        scoring_func = self.scoring_methods.get(method, self._hybrid_balanced_scoring)
        result = scoring_func(expected, agent, context)
        result.suggestions = self._generate_suggestions(result, expected, agent)
        return result

    # --- Core scoring strategies -------------------------------------------------

    def _format_strict_scoring(
        self, expected: str, agent: str, context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        """Evaluate answers that require strict formatting."""
        if expected.strip() == agent.strip():
            return ScoringResult(
                score=1.0,
                confidence=1.0,
                method=ScoringMethod.FORMAT_STRICT,
                breakdown={"exact_match": 1.0},
                reasoning="Exact format match",
                suggestions=[],
            )

        expected_norm, agent_norm = self._normalize_text(expected), self._normalize_text(agent)

        if expected_norm == agent_norm:
            return ScoringResult(
                score=0.95,
                confidence=0.95,
                method=ScoringMethod.FORMAT_STRICT,
                breakdown={"normalized_match": 0.95},
                reasoning="Normalized exact match",
                suggestions=[],
            )

        if expected_norm in agent_norm:
            return ScoringResult(
                score=0.8,
                confidence=0.9,
                method=ScoringMethod.FORMAT_STRICT,
                breakdown={"substring_match": 0.8},
                reasoning="Substring match found after normalization",
                suggestions=[],
            )

        normalized_score = self._normalized_format_match(expected_norm, agent_norm)
        return ScoringResult(
            score=normalized_score,
            confidence=0.8,
            method=ScoringMethod.FORMAT_STRICT,
            breakdown={"normalized_match": normalized_score},
            reasoning=f"Normalized format match: {normalized_score:.3f}",
            suggestions=[],
        )

    def _semantic_flexible_scoring(
        self, expected: str, agent: str, context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        """Evaluate answers with a semantic-focused strategy."""
        semantic_score = self._calculate_semantic_similarity(expected, agent)
        completeness_score = self._calculate_completeness(expected, agent)
        factual_score = self._calculate_factual_accuracy(expected, agent)
        relevance_score = self._calculate_relevance(expected, agent)

        weights = {
            "semantic": 0.4,
            "completeness": 0.3,
            "factual": 0.2,
            "relevance": 0.1,
        }

        total_score = (
            semantic_score * weights["semantic"]
            + completeness_score * weights["completeness"]
            + factual_score * weights["factual"]
            + relevance_score * weights["relevance"]
        )

        return ScoringResult(
            score=total_score,
            confidence=0.8,
            method=ScoringMethod.SEMANTIC_FLEXIBLE,
            breakdown={
                "semantic": semantic_score,
                "completeness": completeness_score,
                "factual": factual_score,
                "relevance": relevance_score,
            },
            reasoning=(
                "Semantic flexible scoring: "
                f"semantic={semantic_score:.3f}, completeness={completeness_score:.3f}, "
                f"factual={factual_score:.3f}, relevance={relevance_score:.3f}"
            ),
            suggestions=[],
        )

    def _hybrid_balanced_scoring(
        self, expected: str, agent: str, context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        """Blend format, semantic, content, and factual signals."""
        format_score = self._normalized_format_match(
            self._normalize_text(expected), self._normalize_text(agent)
        )
        semantic_score = self._calculate_semantic_similarity(expected, agent)
        content_score = self._calculate_content_quality(agent)
        factual_score = self._calculate_factual_accuracy(expected, agent)

        weights = self._calculate_dynamic_weights(expected, agent, context)
        total_score = (
            format_score * weights["format"]
            + semantic_score * weights["semantic"]
            + content_score * weights["content"]
            + factual_score * weights["factual"]
        )

        return ScoringResult(
            score=total_score,
            confidence=0.85,
            method=ScoringMethod.HYBRID_BALANCED,
            breakdown={
                "format": format_score,
                "semantic": semantic_score,
                "content": content_score,
                "factual": factual_score,
            },
            reasoning=(
                "Hybrid balanced scoring with dynamic weights: "
                f"format={format_score:.3f}, semantic={semantic_score:.3f}, "
                f"content={content_score:.3f}, factual={factual_score:.3f}"
            ),
            suggestions=[],
        )

    def _content_focused_scoring(
        self, expected: str, agent: str, context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        """Prioritise key information coverage and professionalism."""
        key_info_score = self._calculate_key_info_match(expected, agent)
        factual_score = self._calculate_factual_accuracy(expected, agent)
        completeness_score = self._calculate_completeness(expected, agent)
        professionalism_score = self._calculate_professionalism(agent)

        weights = {
            "key_info": 0.4,
            "factual": 0.3,
            "completeness": 0.2,
            "professionalism": 0.1,
        }

        total_score = (
            key_info_score * weights["key_info"]
            + factual_score * weights["factual"]
            + completeness_score * weights["completeness"]
            + professionalism_score * weights["professionalism"]
        )

        return ScoringResult(
            score=total_score,
            confidence=0.9,
            method=ScoringMethod.CONTENT_FOCUSED,
            breakdown={
                "key_info": key_info_score,
                "factual": factual_score,
                "completeness": completeness_score,
                "professionalism": professionalism_score,
            },
            reasoning=(
                "Content focused scoring: "
                f"key_info={key_info_score:.3f}, factual={factual_score:.3f}, "
                f"completeness={completeness_score:.3f}, professionalism={professionalism_score:.3f}"
            ),
            suggestions=[],
        )

    # --- Normalization utilities -------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        """Normalise text to improve robust comparisons."""
        if not text:
            return ""

        text = text.lower().strip()
        text = self._normalize_dates(text)
        text = self._normalize_numbers(text)
        text = self._normalize_currency(text)
        text = self._normalize_addresses(text)
        return re.sub(r"\s+", " ", text)

    def _normalize_dates(self, text: str) -> str:
        """Normalise common date formats."""
        month_map = {
            "january": "01",
            "february": "02",
            "march": "03",
            "april": "04",
            "may": "05",
            "june": "06",
            "july": "07",
            "august": "08",
            "september": "09",
            "october": "10",
            "november": "11",
            "december": "12",
        }

        for month_name, month_num in month_map.items():
            pattern = rf"{month_name}\s+(\d{{1,2}}),?\s+(\d{{4}})"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                day = match.group(1).zfill(2)
                year = match.group(2)
                text = re.sub(pattern, f"{year}-{month_num}-{day}", text, flags=re.IGNORECASE)

        date_patterns = [
            (r"(\d{4})/(\d{1,2})/(\d{1,2})", r"\1-\2-\3"),
            (r"(\d{1,2})/(\d{1,2})/(\d{4})", r"\3-\1-\2"),
        ]
        for pattern, replacement in date_patterns:
            text = re.sub(pattern, replacement, text)
        return text

    def _normalize_numbers(self, text: str) -> str:
        """Normalise number formats."""
        return re.sub(r"(\d),(\d)", r"\1\2", text)

    def _normalize_currency(self, text: str) -> str:
        """Normalise currency formats."""
        text = re.sub(r"\$(\d+(?:\.\d+)?)", r"$\1", text)
        text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:usd|dollars?)", r"$\1", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", lambda m: "$" + m.group(1).replace(",", ""), text
        )
        return text

    def _normalize_addresses(self, text: str) -> str:
        """Normalise hexadecimal addresses."""
        return re.sub(r"0x([a-fA-F0-9]{40})", lambda m: "0x" + m.group(1).lower(), text)

    def _normalized_format_match(self, expected: str, agent: str) -> float:
        """Compute similarity after normalisation."""
        if not expected and not agent:
            return 1.0
        if not expected or not agent:
            return 0.0
        if expected == agent:
            return 1.0
        if expected in agent:
            return 0.8

        edit_distance = self._calculate_edit_distance(expected, agent)
        max_length = max(len(expected), len(agent))
        if max_length == 0:
            return 0.0
        similarity = 1.0 - (edit_distance / max_length)
        return max(0.0, similarity)

    # --- Similarity helpers ------------------------------------------------------

    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    def _calculate_semantic_similarity(self, expected: str, agent: str) -> float:
        key_info_score = self._calculate_key_info_similarity(expected, agent)
        if key_info_score > 0.8:
            return key_info_score

        expected_words = set(re.findall(r"\b\w+\b", expected.lower()))
        agent_words = set(re.findall(r"\b\w+\b", agent.lower()))
        if not expected_words:
            return 0.0

        intersection = len(expected_words.intersection(agent_words))
        union = len(expected_words.union(agent_words))
        jaccard_similarity = intersection / union if union > 0 else 0.0

        word_order_similarity = self._calculate_word_order_similarity(expected, agent)
        semantic_similarity = 0.7 * jaccard_similarity + 0.3 * word_order_similarity
        return min(1.0, semantic_similarity)

    def _calculate_key_info_similarity(self, expected: str, agent: str) -> float:
        expected_info = self._extract_key_information(expected)
        agent_info = self._extract_key_information(agent)
        if not expected_info:
            return 0.0

        matched_info = 0
        for expected_item in expected_info:
            for agent_item in agent_info:
                if self._is_similar_key_info(expected_item, agent_item):
                    matched_info += 1
                    break
        return matched_info / len(expected_info)

    def _extract_key_information(self, text: str) -> List[str]:
        info: List[str] = []
        if not text:
            return info

        date_patterns = [
            r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
        ]
        for pattern in date_patterns:
            info.extend(re.findall(pattern, text, re.IGNORECASE))

        number_patterns = [r"\$[\d,]+\.?\d*", r"\d+\.?\d*%", r"\d+\.?\d*"]
        for pattern in number_patterns:
            info.extend(re.findall(pattern, text))

        address_patterns = [r"0x[a-fA-F0-9]{40}", r"0x[a-fA-F0-9]{64}"]
        for pattern in address_patterns:
            info.extend(re.findall(pattern, text))
        return info

    def _is_similar_key_info(self, info1: str, info2: str) -> bool:
        norm1 = self._normalize_text(info1)
        norm2 = self._normalize_text(info2)
        if norm1 == norm2:
            return True
        if self._is_date_info(info1) and self._is_date_info(info2):
            return self._compare_dates(info1, info2)
        if self._is_numeric_info(info1) and self._is_numeric_info(info2):
            return self._compare_numbers(info1, info2)
        if self._is_address_info(info1) and self._is_address_info(info2):
            return norm1.lower() == norm2.lower()
        return False

    def _is_date_info(self, info: str) -> bool:
        date_patterns = [
            r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
        ]
        return any(re.search(pattern, info, re.IGNORECASE) for pattern in date_patterns)

    def _is_numeric_info(self, info: str) -> bool:
        return bool(re.search(r"\d", info))

    def _is_address_info(self, info: str) -> bool:
        return bool(re.search(r"0x[a-fA-F0-9]{40,64}", info))

    def _compare_dates(self, date1: str, date2: str) -> bool:
        try:
            norm1 = self._normalize_dates(date1)
            norm2 = self._normalize_dates(date2)

            pattern = r"(\d{4})-(\d{1,2})-(\d{1,2})"
            match1 = re.search(pattern, norm1)
            match2 = re.search(pattern, norm2)
            if match1 and match2:
                year1, month1, day1 = match1.groups()
                year2, month2, day2 = match2.groups()
                return (year1, month1.zfill(2), day1.zfill(2)) == (
                    year2,
                    month2.zfill(2),
                    day2.zfill(2),
                )
        except Exception:
            pass
        return False

    def _compare_numbers(self, num1: str, num2: str) -> bool:
        try:
            val1 = float(re.sub(r"[^\d.]", "", num1))
            val2 = float(re.sub(r"[^\d.]", "", num2))
            return abs(val1 - val2) < 0.01
        except Exception:
            return False

    def _calculate_word_order_similarity(self, expected: str, agent: str) -> float:
        expected_words = re.findall(r"\b\w+\b", expected.lower())
        agent_words = re.findall(r"\b\w+\b", agent.lower())
        if not expected_words or not agent_words:
            return 0.0

        lcs_length = self._calculate_lcs_length(expected_words, agent_words)
        max_length = max(len(expected_words), len(agent_words))
        return lcs_length / max_length if max_length > 0 else 0.0

    def _calculate_lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    # --- Component scores --------------------------------------------------------

    def _calculate_completeness(self, expected: str, agent: str) -> float:
        expected_words = set(re.findall(r"\b\w+\b", expected.lower()))
        agent_words = set(re.findall(r"\b\w+\b", agent.lower()))
        if not expected_words:
            return 0.0
        covered_words = len(expected_words.intersection(agent_words))
        return covered_words / len(expected_words)

    def _calculate_factual_accuracy(self, expected: str, agent: str) -> float:
        expected_info = self._extract_key_information(expected)
        if not expected_info:
            return 0.5

        agent_info = self._extract_key_information(agent)
        if not agent_info:
            return 0.0

        matched_info = 0
        for info in expected_info:
            if any(self._is_similar_key_info(info, agent_item) for agent_item in agent_info):
                matched_info += 1
        return matched_info / len(expected_info)

    def _calculate_relevance(self, expected: str, agent: str) -> float:
        if not agent:
            return 0.0
        expected_words = set(re.findall(r"\b\w+\b", expected.lower()))
        agent_words = set(re.findall(r"\b\w+\b", agent.lower()))
        if not agent_words:
            return 0.0
        intersection = len(expected_words.intersection(agent_words))
        return intersection / len(agent_words)

    def _calculate_content_quality(self, agent: str) -> float:
        if not agent:
            return 0.0
        length_score = min(len(agent) / 500.0, 1.0)

        structure_indicators = ["first", "second", "finally", "overall", "in summary", "step"]
        structure_score = 0.2 if any(indicator in agent.lower() for indicator in structure_indicators) else 0.0

        reference_indicators = ["according to", "based on", "source", "reference", "data shows", "report"]
        reference_score = 0.2 if any(indicator in agent.lower() for indicator in reference_indicators) else 0.0

        return min(1.0, length_score * 0.6 + structure_score + reference_score)

    def _calculate_key_info_match(self, expected: str, agent: str) -> float:
        expected_info = self._extract_key_information(expected)
        agent_info = self._extract_key_information(agent)
        if not expected_info:
            return 0.5
        matched_info = 0
        for info in expected_info:
            if any(self._is_similar_key_info(info, agent_item) for agent_item in agent_info):
                matched_info += 1
        return matched_info / len(expected_info)

    def _calculate_professionalism(self, agent: str) -> float:
        if not agent:
            return 0.0
        score = 0.0
        technical_terms = [
            "blockchain",
            "ethereum",
            "bitcoin",
            "contract",
            "transaction",
            "hash",
            "address",
            "token",
            "defi",
            "nft",
            "smart contract",
            "protocol",
            "consensus",
            "mining",
            "staking",
            "liquidity",
        ]
        technical_count = sum(1 for term in technical_terms if term in agent.lower())
        if technical_count > 0:
            score += min(0.4, technical_count * 0.1)

        data_indicators = [
            r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
            r"\$[\d,]+\.?\d*",
            r"0x[a-fA-F0-9]{40,64}",
            r"\d+\.?\d*%",
        ]
        data_count = 0
        for pattern in data_indicators:
            data_count += len(re.findall(pattern, agent))
        if data_count > 0:
            score += min(0.3, data_count * 0.1)

        professional_indicators = [
            "according to",
            "based on",
            "analysis",
            "research",
            "data shows",
            "evidence",
            "verified",
            "confirmed",
        ]
        professional_count = sum(1 for indicator in professional_indicators if indicator in agent.lower())
        if professional_count > 0:
            score += min(0.3, professional_count * 0.1)
        return min(1.0, score)

    # --- Dynamic weights ---------------------------------------------------------

    def _calculate_dynamic_weights(
        self, expected: str, agent: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        weights = {"format": 0.2, "semantic": 0.4, "content": 0.2, "factual": 0.2}

        question_text = ""
        if context:
            question_text = context.get("question", "") or ""
        combined = f"{question_text} {expected}"

        if self._is_date_question(combined):
            weights.update({"format": 0.3, "semantic": 0.3, "factual": 0.4})
        elif self._is_numeric_question(combined):
            weights.update({"format": 0.4, "semantic": 0.2, "factual": 0.4})
        elif self._is_technical_question(combined):
            weights.update({"semantic": 0.5, "content": 0.3, "factual": 0.2})
        return weights

    def _is_date_question(self, text: str) -> bool:
        date_keywords = ["date", "when", "time", "timestamp"]
        return any(keyword in text.lower() for keyword in date_keywords)

    def _is_numeric_question(self, text: str) -> bool:
        numeric_keywords = ["price", "amount", "value", "cost", "quantity", "total"]
        return any(keyword in text.lower() for keyword in numeric_keywords)

    def _is_technical_question(self, text: str) -> bool:
        technical_keywords = ["how", "what is", "explain", "describe", "which", "where", "compare"]
        return any(keyword in text.lower() for keyword in technical_keywords)

    # --- Suggestions -------------------------------------------------------------

    def _generate_suggestions(self, result: ScoringResult, expected: str, agent: str) -> List[str]:
        suggestions: List[str] = []

        if result.score < 0.5:
            suggestions.append("Consider improving answer accuracy and completeness")
        if result.breakdown.get("format", 1.0) < 0.5:
            suggestions.append("Check format consistency with expected answer")
        if result.breakdown.get("semantic", 1.0) < 0.5:
            suggestions.append("Improve semantic similarity to expected answer")
        if result.breakdown.get("factual", 1.0) < 0.5:
            suggestions.append("Verify factual accuracy of the answer")
        if result.breakdown.get("completeness", 1.0) < 0.5:
            suggestions.append("Ensure all key information is covered")
        return suggestions
