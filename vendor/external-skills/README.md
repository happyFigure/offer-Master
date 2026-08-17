# External Content Source Packages

This directory stores third-party content-source packages used by OfferMaster experiments. They are kept under the project drive so no runtime dependency, cache, or tool source is placed on the C drive.

## Downloaded Packages

- `weixin-articles-mcp`: MCP server for reading public WeChat Official Account article URLs. Current checked HEAD: `060fb3dd7e41d1c0950a19bc1367d66a6881f915`.
- `xiaohongshu-mcp-complete`: MCP server for Xiaohongshu search, feed detail, profile, and publishing operations. OfferMaster wrapper Skills allow only read-oriented tools by default. Current checked HEAD: `84511f19acccd7eea36ecce2e0e413eda449aa76`.
- `xhscrawl`: Xiaohongshu web/API reference implementation for note detail, note search, user search, and user notes. Current checked HEAD: `813d27ec01ee41af942e5ebb7f1e5b524799f35c`.

## OfferMaster Wrapper Skills

- `docs/agent-skills/vendor-content-sources/wechat-article-content-fetch/SKILL.md`
- `docs/agent-skills/vendor-content-sources/xiaohongshu-content-fetch/SKILL.md`

The external repositories are tools/MCP servers, not all of them are Claude Code style Skill packages. The wrapper Skills translate their capabilities into OfferMaster-readable `SKILL.md` metadata, permissions, source types, and boundaries.

## Safety Notes

- Read-only content extraction is allowed only through ToolRuntimeGuard and Skill permissions.
- Login, visible browser reads, or account-bound operations must require explicit user confirmation.
- Publishing, commenting, liking, collecting, and application submission are denied in the current content-source Skills.
- Actual MCP Gateway registration is a separate integration step.
