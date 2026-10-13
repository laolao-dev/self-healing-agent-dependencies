from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any, ClassVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dependency_killer import (
    TOOL_ID,
    canonical_hash,
    dispatch_mcp_tool,
    load_manifest,
    manifest_changes,
)


class UnitTests(unittest.TestCase):
    manifest: ClassVar[dict[str, Any]] = {}

    def setUp(self) -> None:
        type(self).manifest = load_manifest(Path(os.environ["TOOL_MANIFEST"]))

    def test_manifest_is_valid_mcp_tool_definition(self) -> None:
        self.assertEqual(self.manifest["name"], TOOL_ID)
        self.assertEqual(self.manifest["inputSchema"]["required"], ["ticket_id"])

    def test_transport_returns_bounded_record(self) -> None:
        self.assertEqual(
            dispatch_mcp_tool(self.manifest, {"ticket_id": "TCK-100"}),
            {"ticket_id": "TCK-100", "status": "open", "priority": "high"},
        )

    def test_dependency_diff_is_deterministic(self) -> None:
        baseline = load_manifest(ROOT / "fixtures" / "tool_baseline.json")
        changed = load_manifest(ROOT / "fixtures" / "tool_changed.json")
        first = manifest_changes(baseline, changed)
        second = manifest_changes(baseline, changed)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["path"], f"tools.{TOOL_ID}.description")
        self.assertEqual(len(canonical_hash(first)), 64)


if __name__ == "__main__":
    unittest.main()
