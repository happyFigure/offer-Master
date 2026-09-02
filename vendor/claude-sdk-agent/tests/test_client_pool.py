from __future__ import annotations

import asyncio
import unittest

from src.claude_sdk.client_pool import ClaudeClientPool, SessionClientRecord


class _FakeClient:
    def __init__(self) -> None:
        self.disconnect_calls = 0

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


class ClientPoolTests(unittest.TestCase):
    def test_snapshot_redacts_internal_signature(self) -> None:
        record = SessionClientRecord(
            frontend_session_id="session-1",
            claude_session_id="claude-1",
            model="MiniMax-RAN3",
            resumed=False,
            signature='{"env":{"API_TOKEN":"secret-value"}}',
            client=_FakeClient(),
        )

        snapshot = record.snapshot()

        self.assertNotIn("signature", snapshot)
        self.assertNotIn("secret-value", str(snapshot))
        self.assertEqual(len(snapshot["signatureDigest"]), 24)

    def test_get_or_create_reuses_same_signature(self) -> None:
        async def scenario() -> None:
            pool = ClaudeClientPool()
            created: list[_FakeClient] = []

            async def factory():
                client = _FakeClient()
                created.append(client)
                return client, {"commands": []}

            first = await pool.get_or_create(
                "session-1",
                claude_session_id="claude-1",
                model="MiniMax-RAN3",
                resumed=False,
                signature="sig-1",
                factory=factory,
            )
            second = await pool.get_or_create(
                "session-1",
                claude_session_id="claude-1",
                model="MiniMax-RAN3",
                resumed=False,
                signature="sig-1",
                factory=factory,
            )

            self.assertIs(first, second)
            self.assertEqual(len(created), 1)

        asyncio.run(scenario())

    def test_get_or_create_replaces_client_when_signature_changes(self) -> None:
        async def scenario() -> None:
            pool = ClaudeClientPool()
            created: list[_FakeClient] = []

            async def factory():
                client = _FakeClient()
                created.append(client)
                return client, None

            first = await pool.get_or_create(
                "session-1",
                claude_session_id="claude-1",
                model="MiniMax-RAN3",
                resumed=False,
                signature="sig-1",
                factory=factory,
            )
            second = await pool.get_or_create(
                "session-1",
                claude_session_id="claude-1",
                model="MiniMax-RAN3",
                resumed=True,
                signature="sig-2",
                factory=factory,
            )

            self.assertIsNot(first, second)
            self.assertEqual(first.client.disconnect_calls, 1)
            self.assertEqual(len(created), 2)

        asyncio.run(scenario())
