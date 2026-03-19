"""
StateCheckGrader – validates the final environment state in the Outcome.

Checks source reliability, state consistency, and Web3-specific state
items (transactions, contract addresses, etc.).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..trajectory import GradeResult, Outcome, Trajectory
from .base import BaseGrader
from .web3_validators import validate_address, validate_tx_hash

# Trusted sources for Web3 information
_TRUSTED_DOMAINS = {
    "etherscan.io",
    "bscscan.com",
    "polygonscan.com",
    "arbiscan.io",
    "optimistic.etherscan.io",
    "sec.gov",
    "ethereum.org",
    "eips.ethereum.org",
    "docs.uniswap.org",
    "compound.finance",
    "aave.com",
    "reuters.com",
    "coindesk.com",
    "cointelegraph.com",
    "theblock.co",
}

_DOMAIN_RE = re.compile(r"(?:https?://)?(?:www\.)?([^/\s]+)")


def _extract_domain(url: str) -> str:
    m = _DOMAIN_RE.match(url.strip())
    return m.group(1).lower() if m else url.lower()


def _is_trusted(url: str) -> bool:
    domain = _extract_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in _TRUSTED_DOMAINS)


class StateCheckGrader(BaseGrader):
    """Checks the final state captured in the Outcome."""

    name = "state_check"

    async def grade(
        self,
        outcome: Outcome,
        trajectory: Optional[Trajectory] = None,
        task: Optional[Any] = None,
    ) -> GradeResult:
        assertions: List[Dict[str, Any]] = []

        # 1. Source reliability
        if outcome.sources:
            trusted_count = sum(1 for s in outcome.sources if _is_trusted(s))
            reliable = trusted_count / len(outcome.sources) >= 0.5
            assertions.append(
                {
                    "name": "reliable_sources",
                    "passed": reliable,
                    "detail": {
                        "trusted": trusted_count,
                        "total": len(outcome.sources),
                    },
                }
            )

        # 2. State ↔ answer consistency (answer text should not contradict state)
        if outcome.state and outcome.answer:
            consistent = self._check_state_consistency(outcome.state, outcome.answer)
            assertions.append({"name": "state_consistency", "passed": consistent})

        # 3. Web3-specific state checks
        if isinstance(outcome.state, dict):
            tx = outcome.state.get("transaction")
            if tx:
                valid_tx = isinstance(tx, str) and validate_tx_hash(tx)
                assertions.append({"name": "valid_transaction_hash", "passed": valid_tx})

            contract = outcome.state.get("contract_address")
            if contract:
                assertions.append(
                    {"name": "valid_contract_address", "passed": validate_address(contract)}
                )

        # 4. Required state keys (configurable)
        required_keys = self.config.get("required_state_keys", [])
        for key in required_keys:
            assertions.append(
                {
                    "name": f"state_has_{key}",
                    "passed": key in (outcome.state or {}),
                }
            )

        if not assertions:
            return GradeResult(
                grader_name=self.name,
                score=0.5,
                passed=True,
                reasoning="No state to check – returning neutral score.",
            )

        passed_count = sum(1 for a in assertions if a["passed"])
        score = passed_count / len(assertions)

        return GradeResult(
            grader_name=self.name,
            score=round(score, 4),
            passed=score >= 0.6,
            reasoning=f"State check: {passed_count}/{len(assertions)} assertions passed.",
            assertions=assertions,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _check_state_consistency(state: Dict[str, Any], answer: str) -> bool:
        """Light heuristic: no blatant contradiction between state values and answer."""
        answer_lower = answer.lower()
        for key, value in state.items():
            if not isinstance(value, str):
                continue
            # If state claims something is X but the answer says "not X", flag it
            if value.lower() in answer_lower:
                return True
            if f"not {value.lower()}" in answer_lower or f"no {value.lower()}" in answer_lower:
                return False
        return True
