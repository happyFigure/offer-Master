from __future__ import annotations

import asyncio
import base64
import io
import unittest

from openpyxl import Workbook

from src.multimodal.sdk_input import build_sdk_prompt_input


class MultimodalSdkInputTests(unittest.TestCase):
    def test_build_sdk_prompt_input_converts_image_url_and_deduplicates_attachment(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "analyze image"},
                        {"type": "image_url", "image_url": {"url": "https://files.example.local/image.png"}},
                    ],
                }
            ],
            "metadata": {
                "attachments": [
                    {
                        "kind": "image",
                        "name": "image.png",
                        "fileId": "file-1",
                        "downloadUrl": "https://files.example.local/image.png",
                        "mimeType": "image/png",
                    }
                ]
            },
        }

        async def fetcher(url: str, headers):  # type: ignore[no-untyped-def]
            self.assertEqual(url, "https://files.example.local/image.png")
            self.assertIn("authorization", headers)
            return (b"\x89PNGdemo", "image/png")

        sdk_input = asyncio.run(
            build_sdk_prompt_input(
                payload,
                include_history=False,
                request_headers={"authorization": "Bearer demo"},
                timeout_sec=30.0,
                attachment_text_char_limit=96000,
                fetcher=fetcher,
            )
        )

        self.assertEqual(len(sdk_input.content_blocks), 2)
        self.assertEqual(sdk_input.content_blocks[0]["type"], "text")
        self.assertEqual(sdk_input.content_blocks[1]["type"], "image")
        self.assertEqual(
            sdk_input.content_blocks[1]["source"]["data"],
            base64.b64encode(b"\x89PNGdemo").decode("ascii"),
        )

    def test_build_sdk_prompt_input_extracts_xlsx_attachment_into_text_block(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Summary"
        sheet.append(["name", "score"])
        sheet.append(["Alice", 95])
        sheet.append(["Bob", 88])
        buffer = io.BytesIO()
        workbook.save(buffer)

        payload = {
            "messages": [{"role": "user", "content": "帮我解析这个xlsx"}],
            "metadata": {
                "attachments": [
                    {
                        "kind": "file",
                        "name": "score.xlsx",
                        "fileId": "file-2",
                        "downloadUrl": "https://files.example.local/score.xlsx",
                        "mimeType": "application/wps-office.xlsx",
                    }
                ]
            },
        }

        async def fetcher(url: str, headers):  # type: ignore[no-untyped-def]
            self.assertEqual(url, "https://files.example.local/score.xlsx")
            return (buffer.getvalue(), "application/wps-office.xlsx")

        sdk_input = asyncio.run(
            build_sdk_prompt_input(
                payload,
                include_history=False,
                request_headers={},
                timeout_sec=30.0,
                attachment_text_char_limit=96000,
                fetcher=fetcher,
            )
        )

        self.assertEqual(sdk_input.content_blocks[0]["text"], "帮我解析这个xlsx")
        attachment_block = sdk_input.content_blocks[1]
        self.assertEqual(attachment_block["type"], "text")
        self.assertIn("Spreadsheet: score.xlsx", attachment_block["text"])
        self.assertIn("Summary", attachment_block["text"])
        self.assertIn("Alice", attachment_block["text"])
        self.assertIn("95", attachment_block["text"])

    def test_build_sdk_prompt_input_preserves_history_in_first_turn_text_block(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "new question"},
            ]
        }

        sdk_input = asyncio.run(
            build_sdk_prompt_input(
                payload,
                include_history=True,
                request_headers={},
                timeout_sec=30.0,
                attachment_text_char_limit=96000,
            )
        )

        self.assertEqual(len(sdk_input.content_blocks), 1)
        self.assertIn("[Chat messages since your last reply - for context]", sdk_input.content_blocks[0]["text"])
        self.assertIn("assistant: old answer", sdk_input.content_blocks[0]["text"])
        self.assertIn("[Current message - respond to this]", sdk_input.content_blocks[0]["text"])

    def test_build_sdk_prompt_input_uses_configured_attachment_text_limit(self) -> None:
        payload = {
            "messages": [{"role": "user", "content": "read this code"}],
            "metadata": {
                "attachments": [
                    {
                        "kind": "file",
                        "name": "ParamTable.py",
                        "fileId": "file-3",
                        "downloadUrl": "https://files.example.local/ParamTable.py",
                        "mimeType": "text/x-python",
                    }
                ]
            },
        }
        large_text = "a" * 20000

        async def fetcher(url: str, headers):  # type: ignore[no-untyped-def]
            self.assertEqual(url, "https://files.example.local/ParamTable.py")
            return (large_text.encode("utf-8"), "text/x-python")

        sdk_input = asyncio.run(
            build_sdk_prompt_input(
                payload,
                include_history=False,
                request_headers={},
                timeout_sec=30.0,
                attachment_text_char_limit=40000,
                fetcher=fetcher,
            )
        )

        self.assertNotIn("[Content truncated]", sdk_input.content_blocks[1]["text"])
        self.assertIn("a" * 1000, sdk_input.content_blocks[1]["text"])
