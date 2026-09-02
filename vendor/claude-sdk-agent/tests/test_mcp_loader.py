from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import McpSettings
from src.mcp_loader import build_mcp_servers


class McpLoaderTests(unittest.TestCase):
    def test_build_mcp_servers_loads_http_configs_and_fills_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mcps").mkdir(parents=True, exist_ok=True)
            (root / "mcps" / "tool_service.json").write_text(
                json.dumps(
                    {
                        "name": "通用工具服务",
                        "url": "http://10.2.67.41:30010/mcp",
                        "transport": "http",
                        "headers": {
                            "x-api-key": "",
                            "uac-user-id": "",
                            "uac-user-token": "",
                        },
                    }
                ),
                encoding="utf-8",
            )

            servers = build_mcp_servers(
                McpSettings(config_dir=root / "mcps", extra_config_dirs=[], auto_load=True),
                request_headers={
                    "uac-user-id": "10154402",
                    "uac-user-token": "uac-token-1",
                },
                fallback_tdl_api_key="tdl_shared_key",
            )

            self.assertEqual(servers["通用工具服务"]["type"], "http")
            self.assertEqual(servers["通用工具服务"]["url"], "http://10.2.67.41:30010/mcp")
            self.assertEqual(servers["通用工具服务"]["headers"]["x-api-key"], "tdl_shared_key")
            self.assertEqual(servers["通用工具服务"]["headers"]["uac-user-id"], "10154402")
            self.assertEqual(servers["通用工具服务"]["headers"]["uac-user-token"], "uac-token-1")

    def test_build_mcp_servers_skips_examples_and_invalid_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mcps").mkdir(parents=True, exist_ok=True)
            (root / "mcps" / "demo.example.json").write_text("{}", encoding="utf-8")
            (root / "mcps" / "invalid.json").write_text(json.dumps({"name": "bad"}), encoding="utf-8")

            servers = build_mcp_servers(
                McpSettings(config_dir=root / "mcps", extra_config_dirs=[], auto_load=True),
                request_headers={},
                fallback_tdl_api_key="",
            )

            self.assertEqual(servers, {})

    def test_build_mcp_servers_injects_x_api_key_when_header_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mcps").mkdir(parents=True, exist_ok=True)
            (root / "mcps" / "haha.json").write_text(
                json.dumps(
                    {
                        "name": "haha",
                        "url": "http://10.2.67.41:30006/mcp",
                        "transport": "http",
                        "headers": {},
                    }
                ),
                encoding="utf-8",
            )

            servers = build_mcp_servers(
                McpSettings(config_dir=root / "mcps", extra_config_dirs=[], auto_load=True),
                request_headers={},
                fallback_tdl_api_key="tdl_shared_key",
            )

            self.assertEqual(servers["haha"]["type"], "http")
            self.assertEqual(servers["haha"]["headers"]["x-api-key"], "tdl_shared_key")

    def test_build_mcp_servers_loads_stdio_configs_and_fills_minimax_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mcps").mkdir(parents=True, exist_ok=True)
            (root / "mcps" / "minimax_web_search.json").write_text(
                json.dumps(
                    {
                        "name": "MiniMax",
                        "transport": "stdio",
                        "command": "uvx",
                        "args": ["minimax-coding-plan-mcp", "-y"],
                        "env": {
                            "MINIMAX_API_KEY": "",
                            "MINIMAX_API_HOST": "https://api.minimaxi.com",
                        },
                    }
                ),
                encoding="utf-8",
            )

            servers = build_mcp_servers(
                McpSettings(config_dir=root / "mcps", extra_config_dirs=[], auto_load=True),
                request_headers={"x-api-key": "tdl_request_key"},
                fallback_tdl_api_key="tdl_shared_key",
            )

            self.assertEqual(servers["MiniMax"]["command"], "uvx")
            self.assertEqual(servers["MiniMax"]["args"], ["minimax-coding-plan-mcp", "-y"])
            self.assertEqual(servers["MiniMax"]["env"]["MINIMAX_API_KEY"], "tdl_request_key")
            self.assertEqual(servers["MiniMax"]["env"]["MINIMAX_API_HOST"], "https://api.minimaxi.com")

    def test_build_mcp_servers_loads_extra_config_dirs_after_shared_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shared").mkdir(parents=True, exist_ok=True)
            (root / "private").mkdir(parents=True, exist_ok=True)
            (root / "shared" / "tool_service.json").write_text(
                json.dumps(
                    {
                        "name": "通用工具服务",
                        "url": "http://10.2.67.41:30010/mcp",
                        "transport": "http",
                    }
                ),
                encoding="utf-8",
            )
            (root / "private" / "minimax_web_search.json").write_text(
                json.dumps(
                    {
                        "name": "MiniMax",
                        "transport": "stdio",
                        "command": "uvx",
                        "args": ["minimax-coding-plan-mcp", "-y"],
                        "env": {"MINIMAX_API_KEY": ""},
                    }
                ),
                encoding="utf-8",
            )

            servers = build_mcp_servers(
                McpSettings(config_dir=root / "shared", extra_config_dirs=[root / "private"], auto_load=True),
                request_headers={"x-api-key": "tdl_request_key"},
                fallback_tdl_api_key="",
            )

            self.assertEqual(servers["通用工具服务"]["type"], "http")
            self.assertEqual(servers["MiniMax"]["command"], "uvx")
            self.assertEqual(servers["MiniMax"]["env"]["MINIMAX_API_KEY"], "tdl_request_key")
