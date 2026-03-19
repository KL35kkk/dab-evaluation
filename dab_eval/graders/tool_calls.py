"""
ToolCallsGrader – evaluates the agent's tool usage within a Trajectory.

Checks: tool call success rate, selection relevance, absence of
redundant calls, loop detection, and Web3 parameter correctness.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..trajectory import GradeResult, Outcome, StepType, Trajectory
from .base import BaseGrader
from .web3_validators import get_expected_tools, validate_web3_params


class ToolCallsGrader(BaseGrader):
    """Grades tool usage quality from the Trajectory."""

    name = "tool_calls"

    async def grade(
        self,
        outcome: Outcome,
        trajectory: Optional[Trajectory] = None,
        task: Optional[Any] = None,
    ) -> GradeResult:
        if not trajectory or not trajectory.steps:
            return self._make_degraded_result("No trajectory available for tool call analysis.")

        tool_steps = [s for s in trajectory.steps if s.step_type == StepType.TOOL_CALL]

        if not tool_steps:
            # Agent produced an answer without any tool calls – this may be fine
            # for knowledge-based tasks, so return a neutral score.
            return GradeResult(
                grader_name=self.name,
                score=0.7,
                passed=True,
                reasoning="No tool calls in trajectory (acceptable for knowledge-based tasks).",
            )

        assertions: List[Dict[str, Any]] = []
        category = str(getattr(task, "category", "")) if task else ""
        if hasattr(category, "value"):
            category = category.value  # handle TaskCategory enum

        # 1. Tool call success rate
        success_rate = sum(1 for s in tool_steps if s.tool_success) / len(tool_steps)
        assertions.append(
            {
                "name": "tool_success_rate",
                "passed": success_rate >= self.config.get("min_success_rate", 0.8),
                "value": success_rate,
            }
        )

        # 2. Tool selection relevance
        expected_tools = get_expected_tools(category)
        if expected_tools:
            used_tools = {s.tool_name for s in tool_steps if s.tool_name}
            if used_tools:
                relevance = len(used_tools & expected_tools) / len(used_tools)
            else:
                relevance = 0.0
            assertions.append(
                {
                    "name": "tool_selection_relevance",
                    "passed": relevance >= self.config.get("min_relevance", 0.5),
                    "value": relevance,
                }
            )

        # 3. Redundancy / loop detection
        call_signatures = [(s.tool_name, str(s.tool_input)) for s in tool_steps]
        unique_ratio = (
            len(set(call_signatures)) / len(call_signatures) if call_signatures else 1.0
        )
        assertions.append(
            {
                "name": "no_redundant_calls",
                "passed": unique_ratio >= self.config.get("min_unique_ratio", 0.5),
                "value": unique_ratio,
            }
        )

        # 4. Loop detection (trajectory-level)
        assertions.append(
            {"name": "no_infinite_loop", "passed": not trajectory.has_loops}
        )

        # 5. Web3 parameter format validation
        param_failures: List[str] = []
        for step in tool_steps:
            if step.tool_input and not validate_web3_params(step.tool_input):
                param_failures.append(f"step_{step.step_id}")

        assertions.append(
            {
                "name": "valid_web3_params",
                "passed": len(param_failures) == 0,
                "failed_steps": param_failures,
            }
        )

        # 6. Step count efficiency (configurable)
        max_steps = self.config.get("max_tool_calls")
        if max_steps:
            assertions.append(
                {
                    "name": "step_count_within_limit",
                    "passed": len(tool_steps) <= max_steps,
                    "value": len(tool_steps),
                }
            )

        passed_count = sum(1 for a in assertions if a["passed"])
        score = passed_count / len(assertions)

        return GradeResult(
            grader_name=self.name,
            score=round(score, 4),
            passed=score >= self.config.get("pass_threshold", 0.6),
            reasoning=(
                f"Tool calls: {passed_count}/{len(assertions)} checks passed. "
                f"{len(tool_steps)} tool calls, success rate {success_rate:.0%}."
            ),
            assertions=assertions,
            details={
                "n_tool_calls": len(tool_steps),
                "tool_success_rate": success_rate,
                "unique_ratio": unique_ratio,
            },
        )
