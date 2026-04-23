"""Phase 4 gate: retrieval frontier behavior on a fixture graph.

Skeleton: needs LLM backend. Describes the contract Traverser must meet.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(
    reason="Phase 4 gate requires LLM client (see skillmap/llm/client.py TODO)",
    strict=False,
)


@pytest.mark.asyncio
async def test_frontier_stops_at_deepest_matching_node() -> None:
    """Graph A -> B -> C, all in the same category. Query matches all three.
    Expected frontier: [C] only (deepest match replaces ancestors).
    """
    raise NotImplementedError("Phase 4 skeleton")


@pytest.mark.asyncio
async def test_frontier_includes_multiple_roots_when_parallel_branches_match() -> None:
    """Graph A -> B, C -> D. Query matches A+B and C. Expected frontier:
    [B, C] - B replaces A, C has no matching child."""
    raise NotImplementedError("Phase 4 skeleton")


@pytest.mark.asyncio
async def test_tentative_skills_skipped_by_default() -> None:
    raise NotImplementedError("Phase 4 skeleton")
