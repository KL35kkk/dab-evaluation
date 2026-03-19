"""
Web3-specific validation utilities used across multiple graders.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Set


# ── Regex patterns ─────────────────────────────────────────────────────────────

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_TX_HASH_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")
_ENS_RE = re.compile(r"^[a-zA-Z0-9-]+\.eth$")
_BYTES32_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")

# ── Known tool sets per task category ──────────────────────────────────────────

WEB3_TOOL_PATTERNS: Dict[str, Dict[str, Any]] = {
    "onchain_retrieval": {
        "expected_tools": {
            "blockchain_query",
            "contract_read",
            "transaction_fetch",
            "etherscan_api",
            "get_block",
        },
        "required_params": ["chain_id", "address"],
    },
    "defi_analysis": {
        "expected_tools": {
            "price_query",
            "liquidity_check",
            "protocol_info",
            "dex_api",
        },
        "required_params": ["token_address", "protocol"],
    },
    "web_retrieval": {
        "expected_tools": {
            "web_search",
            "api_call",
            "document_fetch",
        },
        "required_params": ["query"],
    },
    "web_onchain_retrieval": {
        "expected_tools": {
            "web_search",
            "blockchain_query",
            "contract_read",
        },
        "required_params": ["query"],
    },
}

# ── Validation helpers ──────────────────────────────────────────────────────────


def validate_address(value: str) -> bool:
    return bool(_ADDRESS_RE.match(value.strip()))


def validate_tx_hash(value: str) -> bool:
    return bool(_TX_HASH_RE.match(value.strip()))


def validate_ens(value: str) -> bool:
    return bool(_ENS_RE.match(value.strip()))


def validate_bytes32(value: str) -> bool:
    return bool(_BYTES32_RE.match(value.strip()))


def get_expected_tools(category: str) -> Set[str]:
    pattern = WEB3_TOOL_PATTERNS.get(category.lower(), {})
    return pattern.get("expected_tools", set())


def validate_web3_params(params: Dict[str, Any]) -> bool:
    """Return False if any param that looks like an address/hash is malformed."""
    for key, value in params.items():
        if not isinstance(value, str):
            continue
        key_lower = key.lower()
        if "address" in key_lower:
            if not (validate_address(value) or validate_ens(value)):
                return False
        if ("hash" in key_lower or key_lower in ("tx", "txhash")) and value.startswith("0x"):
            if not validate_tx_hash(value):
                return False
    return True


def normalize_number(value: str) -> Optional[int]:
    """Strip common separators and parse as int; return None on failure."""
    try:
        cleaned = str(value).replace(",", "").replace("_", "").replace(" ", "")
        return int(cleaned)
    except (ValueError, TypeError):
        return None
