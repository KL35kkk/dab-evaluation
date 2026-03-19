"""
DeterministicTestsGrader – code-based checks on the Outcome.

Runs deterministic assertions: exact match, substring containment,
regex patterns, numeric range checks, and Web3-specific format
validation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..trajectory import GradeResult, Outcome, Trajectory
from .base import BaseGrader
from .web3_validators import (
    normalize_number,
    validate_address,
    validate_ens,
    validate_tx_hash,
)


class DeterministicTestsGrader(BaseGrader):
    """Runs deterministic (code-based) tests against the Outcome."""

    name = "deterministic_tests"

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _exact_match(expected: str, actual: str) -> bool:
        return DeterministicTestsGrader._normalize(expected) == DeterministicTestsGrader._normalize(actual)

    @staticmethod
    def _contains(expected: str, actual: str) -> bool:
        return DeterministicTestsGrader._normalize(expected) in DeterministicTestsGrader._normalize(actual)

    # ── grading ───────────────────────────────────────────────────────────────

    async def grade(
        self,
        outcome: Outcome,
        trajectory: Optional[Trajectory] = None,
        task: Optional[Any] = None,
    ) -> GradeResult:
        assertions: List[Dict[str, Any]] = []
        expected = (task.expected_answer if task else None) or ""
        answer = outcome.answer
        context = (task.context if task else {}) or {}

        # 1. Exact match
        if expected:
            assertions.append(
                {"name": "exact_match", "passed": self._exact_match(expected, answer)}
            )

        # 2. Substring containment
        if expected:
            assertions.append(
                {"name": "contains_answer", "passed": self._contains(expected, answer)}
            )

        # 3. Regex patterns defined in grader config or task context
        for pattern in self.config.get("regex_patterns", []):
            match = bool(re.search(pattern, answer))
            assertions.append({"name": f"regex:{pattern[:30]}", "passed": match})

        # 4. Numeric range check
        num_range = self.config.get("numeric_range")
        if num_range:
            parsed = normalize_number(answer)
            in_range = (
                parsed is not None
                and num_range.get("min", float("-inf")) <= parsed <= num_range.get("max", float("inf"))
            )
            assertions.append({"name": "numeric_range", "passed": in_range})

        # 5. Web3-specific format checks (driven by context)
        if context.get("check_address"):
            ok = validate_address(answer) or validate_ens(answer)
            assertions.append({"name": "valid_web3_address", "passed": ok})

        if context.get("check_tx_hash"):
            assertions.append({"name": "valid_tx_hash", "passed": validate_tx_hash(answer)})

        # 6. Multiple-choice option check (for DMind-style tasks)
        mc_options = context.get("options") or task_options(task)
        if mc_options:
            answer_clean = answer.strip().upper()
            assertions.append(
                {
                    "name": "valid_mc_choice",
                    "passed": answer_clean in [o.upper() for o in mc_options],
                }
            )

        if not assertions:
            # Nothing to assert – neutral score
            return GradeResult(
                grader_name=self.name,
                score=0.5,
                passed=True,
                reasoning="No deterministic assertions defined for this task.",
            )

        passed_count = sum(1 for a in assertions if a["passed"])
        score = passed_count / len(assertions)
        passed = score >= 0.5

        return GradeResult(
            grader_name=self.name,
            score=round(score, 4),
            passed=passed,
            reasoning=f"Passed {passed_count}/{len(assertions)} deterministic assertions.",
            assertions=assertions,
        )

    def get_assertions(self) -> List[str]:
        base = ["exact_match", "contains_answer"]
        for p in self.config.get("regex_patterns", []):
            base.append(f"regex:{p[:30]}")
        return base


# ── helpers ───────────────────────────────────────────────────────────────────


def task_options(task: Optional[Any]) -> List[str]:
    """Extract multiple-choice options from a task object if present."""
    if task is None:
        return []
    ctx = getattr(task, "context", {}) or {}
    return ctx.get("options", [])
