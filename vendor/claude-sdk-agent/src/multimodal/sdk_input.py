from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Mapping

from ..api.payload import build_initial_prompt, extract_latest_user_text
from .attachments import (
    AttachmentFetcher,
    attachment_ref_to_blocks,
    extract_latest_user_parts,
    extract_metadata_attachments,
    image_part_to_block,
    part_fingerprint,
)


@dataclass(slots=True)
class SdkPromptInput:
    content_blocks: list[dict[str, Any]]
    prompt_chars: int

    def as_stream(self) -> AsyncIterator[dict[str, Any]]:
        async def _stream() -> AsyncIterator[dict[str, Any]]:
            yield {
                "type": "user",
                "message": {"role": "user", "content": list(self.content_blocks)},
                "parent_tool_use_id": None,
            }

        return _stream()


def build_runtime_command_prompt_input(command_text: str) -> SdkPromptInput:
    text = str(command_text or "").strip()
    return SdkPromptInput(
        content_blocks=[{"type": "text", "text": text}],
        prompt_chars=len(text),
    )


async def build_sdk_prompt_input(
    payload: Mapping[str, Any],
    *,
    include_history: bool,
    request_headers: Mapping[str, str],
    timeout_sec: float,
    attachment_text_char_limit: int,
    fetcher: AttachmentFetcher | None = None,
) -> SdkPromptInput:
    prompt_text = build_initial_prompt(payload) if include_history else extract_latest_user_text(payload)
    content_blocks: list[dict[str, Any]] = []
    if prompt_text:
        content_blocks.append({"type": "text", "text": prompt_text})

    seen_fingerprints: set[str] = set()
    for part in extract_latest_user_parts(payload):
        if str(part.get("type") or "").strip().lower() != "image_url":
            continue
        block = await image_part_to_block(
            part,
            request_headers=request_headers,
            timeout_sec=timeout_sec,
            fetcher=fetcher,
        )
        if block is None:
            continue
        seen_fingerprints.add(part_fingerprint(part))
        content_blocks.append(block)

    for ref in extract_metadata_attachments(payload):
        if ref.fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(ref.fingerprint)
        blocks = await attachment_ref_to_blocks(
            ref,
            request_headers=request_headers,
            timeout_sec=timeout_sec,
            max_text_chars=attachment_text_char_limit,
            fetcher=fetcher,
        )
        content_blocks.extend(blocks)

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    return SdkPromptInput(content_blocks=content_blocks, prompt_chars=len(prompt_text))
