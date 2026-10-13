from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from dependency_killer import dispatch_mcp_tool, load_manifest


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: mcp_tool_server.py MANIFEST")
    manifest = load_manifest(Path(sys.argv[1]))
    server: MCPServer[object] = MCPServer("dependency-killer-demo", version="1.0.0")

    @server.tool(
        name=manifest["name"],
        description=manifest["description"],
        structured_output=True,
    )
    def lookup_ticket(ticket_id: str) -> dict[str, str]:
        return dispatch_mcp_tool(manifest, {"ticket_id": ticket_id})

    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
