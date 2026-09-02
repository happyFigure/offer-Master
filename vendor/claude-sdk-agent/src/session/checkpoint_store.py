from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .models import SessionCheckpoint


class SessionCheckpointStore:
    def __init__(self, path: Path, *, max_items_per_session: int = 200) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._max_items_per_session = max(1, int(max_items_per_session))

    async def list(self, frontend_session_id: str) -> list[SessionCheckpoint]:
        available = [item for item in await self.list_raw(frontend_session_id) if not item.unavailable_reason]
        compacted = self._compact_adjacent_prompt_replays(available)
        return [item for item in compacted if item.affected_files or item.rewound_at > 0]

    async def list_raw(self, frontend_session_id: str) -> list[SessionCheckpoint]:
        session_id = str(frontend_session_id or "").strip()
        if not session_id:
            return []
        async with self._lock:
            data = self._read_all()
        items = data.get(session_id)
        if not isinstance(items, list):
            return []
        checkpoints: list[SessionCheckpoint] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            checkpoint_id = str(item.get("checkpoint_id") or "").strip()
            claude_session_id = str(item.get("claude_session_id") or "").strip()
            if not checkpoint_id:
                continue
            checkpoints.append(
                SessionCheckpoint(
                    frontend_session_id=session_id,
                    claude_session_id=claude_session_id,
                    checkpoint_id=checkpoint_id,
                    created_at=float(item.get("created_at") or 0.0),
                    prompt_excerpt=str(item.get("prompt_excerpt") or "").strip(),
                    unavailable_reason=str(item.get("unavailable_reason") or "").strip(),
                    affected_files=_normalize_files(item.get("affected_files")),
                    rewound_at=float(item.get("rewound_at") or 0.0),
                    rewound_checkpoint_id=str(item.get("rewound_checkpoint_id") or "").strip(),
                )
            )
        checkpoints.sort(key=lambda item: item.created_at)
        return checkpoints

    async def get(self, frontend_session_id: str, checkpoint_id: str) -> SessionCheckpoint | None:
        wanted = str(checkpoint_id or "").strip()
        if not wanted:
            return None
        for checkpoint in await self.list_raw(frontend_session_id):
            if checkpoint.checkpoint_id == wanted:
                return checkpoint
        return None

    @staticmethod
    def _compact_adjacent_prompt_replays(checkpoints: list[SessionCheckpoint]) -> list[SessionCheckpoint]:
        compacted: list[SessionCheckpoint] = []
        for checkpoint in checkpoints:
            prompt = " ".join(str(checkpoint.prompt_excerpt or "").split())
            previous_prompt = " ".join(str(compacted[-1].prompt_excerpt or "").split()) if compacted else ""
            if prompt and prompt == previous_prompt:
                previous = compacted[-1]
                if not previous.claude_session_id and checkpoint.claude_session_id:
                    previous.claude_session_id = checkpoint.claude_session_id
                previous.affected_files = _normalize_files([*previous.affected_files, *checkpoint.affected_files])
                if checkpoint.rewound_at > 0 and previous.rewound_at <= 0:
                    previous.rewound_at = checkpoint.rewound_at
                if checkpoint.rewound_checkpoint_id and not previous.rewound_checkpoint_id:
                    previous.rewound_checkpoint_id = checkpoint.rewound_checkpoint_id
                continue
            compacted.append(checkpoint)
        return compacted

    async def put(
        self,
        frontend_session_id: str,
        claude_session_id: str,
        checkpoint_id: str,
        *,
        prompt_excerpt: str = "",
        affected_files: list[str] | None = None,
    ) -> SessionCheckpoint:
        checkpoint = SessionCheckpoint(
            frontend_session_id=frontend_session_id,
            claude_session_id=claude_session_id,
            checkpoint_id=checkpoint_id,
            created_at=time.time(),
            prompt_excerpt=str(prompt_excerpt or "").strip(),
            affected_files=_normalize_files(affected_files),
        )
        async with self._lock:
            data = self._read_all()
            items = data.setdefault(frontend_session_id, [])
            if not isinstance(items, list):
                items = []
                data[frontend_session_id] = items
            items = [item for item in items if not (isinstance(item, dict) and str(item.get("checkpoint_id") or "").strip() == checkpoint_id)]
            items.append(checkpoint.to_dict())
            items = items[-self._max_items_per_session :]
            data[frontend_session_id] = items
            self._write_all(data)
        return checkpoint

    async def mark_unavailable(self, frontend_session_id: str, checkpoint_id: str, *, reason: str) -> None:
        session_id = str(frontend_session_id or "").strip()
        wanted = str(checkpoint_id or "").strip()
        unavailable_reason = str(reason or "").strip()
        if not session_id or not wanted or not unavailable_reason:
            return
        async with self._lock:
            data = self._read_all()
            items = data.get(session_id)
            if not isinstance(items, list):
                return
            changed = False
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("checkpoint_id") or "").strip() != wanted:
                    continue
                item["unavailable_reason"] = unavailable_reason
                changed = True
            if changed:
                self._write_all(data)

    async def update_metadata(
        self,
        frontend_session_id: str,
        checkpoint_id: str,
        *,
        claude_session_id: str = "",
        affected_files: list[str] | None = None,
        rewound_checkpoint_id: str = "",
        rewound_at: float = 0.0,
    ) -> None:
        session_id = str(frontend_session_id or "").strip()
        wanted = str(checkpoint_id or "").strip()
        if not session_id or not wanted:
            return
        clean_claude_session_id = str(claude_session_id or "").strip()
        clean_files = _normalize_files(affected_files)
        clean_rewound_checkpoint_id = str(rewound_checkpoint_id or "").strip()
        clean_rewound_at = float(rewound_at or 0.0)
        async with self._lock:
            data = self._read_all()
            items = data.get(session_id)
            if not isinstance(items, list):
                return
            changed = False
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("checkpoint_id") or "").strip() != wanted:
                    continue
                if clean_claude_session_id and not str(item.get("claude_session_id") or "").strip():
                    item["claude_session_id"] = clean_claude_session_id
                    changed = True
                if clean_files:
                    existing = _normalize_files(item.get("affected_files"))
                    merged = _normalize_files([*existing, *clean_files])
                    if merged != existing:
                        item["affected_files"] = merged
                        changed = True
                if clean_rewound_checkpoint_id:
                    if str(item.get("rewound_checkpoint_id") or "").strip() != clean_rewound_checkpoint_id:
                        item["rewound_checkpoint_id"] = clean_rewound_checkpoint_id
                        changed = True
                    existing_rewound_at = float(item.get("rewound_at") or 0.0)
                    if clean_rewound_at > 0 and existing_rewound_at <= 0:
                        item["rewound_at"] = clean_rewound_at
                        changed = True
            if changed:
                self._write_all(data)

    def _read_all(self) -> dict[str, object]:
        if not self._path.exists():
            return {}
        text = self._path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}

    def _write_all(self, data: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self._path)


def _normalize_files(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    files: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        files.append(text)
    return files
