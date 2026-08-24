from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from src.session.checkpoint_store import SessionCheckpointStore


class CheckpointStoreTests(unittest.TestCase):
    def test_store_persists_and_lists_checkpoints(self) -> None:
        async def scenario(root: Path) -> None:
            store = SessionCheckpointStore(root / "checkpoints.json")
            await store.put("session-1", "claude-1", "cp-1", prompt_excerpt="first", affected_files=["/tmp/first.txt"])
            await store.put("session-1", "claude-1", "cp-2", prompt_excerpt="second", affected_files=["/tmp/second.txt"])
            items = await store.list("session-1")
            self.assertEqual([item.checkpoint_id for item in items], ["cp-1", "cp-2"])
            self.assertEqual(items[1].prompt_excerpt, "second")
            self.assertEqual((await store.get("session-1", "cp-2")).prompt_excerpt, "second")  # type: ignore[union-attr]
            self.assertIsNone(await store.get("session-1", "missing"))

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(scenario(Path(tmp)))

    def test_store_lists_legacy_checkpoints_without_claude_session_id(self) -> None:
        async def scenario(root: Path) -> None:
            path = root / "checkpoints.json"
            path.write_text(
                """
{
  "session-1": [
    {
      "frontend_session_id": "session-1",
      "claude_session_id": "",
      "checkpoint_id": "cp-legacy",
      "created_at": 1.0,
      "prompt_excerpt": "legacy"
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )
            store = SessionCheckpointStore(path)
            items = await store.list("session-1")
            self.assertEqual(items, [])
            raw_items = await store.list_raw("session-1")
            self.assertEqual(len(raw_items), 1)
            self.assertEqual(raw_items[0].checkpoint_id, "cp-legacy")
            self.assertEqual(raw_items[0].claude_session_id, "")

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(scenario(Path(tmp)))

    def test_store_hides_non_file_checkpoints_from_ui_list(self) -> None:
        async def scenario(root: Path) -> None:
            store = SessionCheckpointStore(root / "checkpoints.json")
            await store.put("session-1", "claude-1", "cp-no-file", prompt_excerpt="/compact")
            await store.put(
                "session-1",
                "claude-1",
                "cp-file",
                prompt_excerpt="edit file",
                affected_files=["/tmp/demo.txt"],
            )

            items = await store.list("session-1")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].checkpoint_id, "cp-file")

            raw_items = await store.list_raw("session-1")
            self.assertEqual([item.checkpoint_id for item in raw_items], ["cp-no-file", "cp-file"])

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(scenario(Path(tmp)))

    def test_store_keeps_rewound_checkpoint_without_file_list(self) -> None:
        async def scenario(root: Path) -> None:
            store = SessionCheckpointStore(root / "checkpoints.json")
            await store.put("session-1", "claude-1", "cp-no-file", prompt_excerpt="edit file")
            await store.update_metadata(
                "session-1",
                "cp-no-file",
                rewound_checkpoint_id="cp-no-file",
                rewound_at=123.0,
            )

            items = await store.list("session-1")
            self.assertEqual([item.checkpoint_id for item in items], ["cp-no-file"])
            self.assertEqual(items[0].rewound_checkpoint_id, "cp-no-file")
            self.assertEqual(items[0].rewound_at, 123.0)

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(scenario(Path(tmp)))

    def test_store_compacts_adjacent_duplicate_prompts_for_ui(self) -> None:
        async def scenario(root: Path) -> None:
            store = SessionCheckpointStore(root / "checkpoints.json")
            await store.put("session-1", "claude-1", "cp-1", prompt_excerpt="same prompt")
            await store.put("session-1", "claude-2", "cp-2", prompt_excerpt="same   prompt", affected_files=["/tmp/demo.txt"])
            await store.put("session-1", "claude-1", "cp-3", prompt_excerpt="next prompt", affected_files=["/tmp/next.txt"])

            items = await store.list("session-1")
            self.assertEqual([item.checkpoint_id for item in items], ["cp-1", "cp-3"])
            self.assertEqual(items[0].affected_files, ["/tmp/demo.txt"])

            await store.update_metadata("session-1", "cp-2", rewound_checkpoint_id="cp-2", rewound_at=456.0)
            items = await store.list("session-1")
            self.assertEqual(items[0].rewound_checkpoint_id, "cp-2")
            self.assertEqual(items[0].rewound_at, 456.0)

            raw_items = await store.list_raw("session-1")
            self.assertEqual([item.checkpoint_id for item in raw_items], ["cp-1", "cp-2", "cp-3"])
            hidden_checkpoint = await store.get("session-1", "cp-1")
            self.assertIsNotNone(hidden_checkpoint)
            self.assertEqual(hidden_checkpoint.checkpoint_id, "cp-1")  # type: ignore[union-attr]

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(scenario(Path(tmp)))

    def test_store_hides_unavailable_checkpoints_from_ui_list(self) -> None:
        async def scenario(root: Path) -> None:
            store = SessionCheckpointStore(root / "checkpoints.json")
            await store.put("session-1", "claude-1", "cp-1", prompt_excerpt="no file edit", affected_files=["/tmp/old.txt"])
            await store.put("session-1", "claude-1", "cp-2", prompt_excerpt="file edit", affected_files=["/tmp/new.txt"])

            await store.mark_unavailable("session-1", "cp-1", reason="no_file_checkpoint")

            items = await store.list("session-1")
            self.assertEqual([item.checkpoint_id for item in items], ["cp-2"])

            raw_items = await store.list_raw("session-1")
            self.assertEqual([item.checkpoint_id for item in raw_items], ["cp-1", "cp-2"])
            self.assertEqual(raw_items[0].unavailable_reason, "no_file_checkpoint")
            self.assertEqual(raw_items[1].unavailable_reason, "")

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(scenario(Path(tmp)))
