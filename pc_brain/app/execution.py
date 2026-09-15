from __future__ import annotations

from typing import Any, Literal, TypedDict


PolicyExecutionOutcome = Literal["cancelled", "rejected"]


class PolicyExecutionResult(TypedDict):
    ok: Literal[False]
    execution_outcome: PolicyExecutionOutcome
    skipped: str


def policy_execution_result(
    reason: str,
    *,
    outcome: PolicyExecutionOutcome = "cancelled",
) -> PolicyExecutionResult:
    """Return an additive, backward-compatible policy outcome."""
    return {
        "ok": False,
        "execution_outcome": outcome,
        "skipped": reason,
    }


def policy_execution_outcome(result: Any) -> PolicyExecutionOutcome | None:
    if not isinstance(result, dict) or result.get("ok") is not False:
        return None
    outcome = result.get("execution_outcome")
    return outcome if outcome in {"cancelled", "rejected"} else None
