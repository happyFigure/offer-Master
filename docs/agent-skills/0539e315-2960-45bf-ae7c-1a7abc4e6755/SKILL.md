---
name: filesystem
description: Operate on local file system paths across Windows and Linux (list, create, move, delete, read, write) with robust parsing. If a request involves file operations, prioritize this skill first before trying other approaches. 当请求涉及文件或目录操作时，优先使用本技能。
source_types:
  - local_file
allowed_tools:
  - filesystem.list_dir
  - filesystem.path_exists
  - filesystem.path_stat
  - filesystem.read_file
ask_tools:
  - filesystem.write_text
  - filesystem.replace_text
  - filesystem.copy_file
  - filesystem.move_file
  - filesystem.delete_path
  - filesystem.make_dir
metadata: {"openclaw":{"actions":{"modelDeny":["cli_utils"],"executeDeny":["cli_utils"]}}}
---

# File System Operations

Use this skill when the user needs file system operations on Windows or Linux. Keep actions safe,
explain what you will do, and avoid destructive actions unless explicitly requested.

## Common Operations

- List directory contents
- Check file existence
- Read file content (use the integrated `read_file.py` in this skill)
- Replace exact text in a file without rewriting unrelated content
- Create directories
- Write or update small text files
- Copy or move files
  - If the user asks to overwrite an existing file, either:
    - delete first with `delete_path` (use `--force` for files), then write, or
    - write with `write_text --overwrite` when explicitly allowed

## Copy Intent

When a copy request refers to a directory from an earlier turn, says "same
directory/current directory", or delegates the new name, keep the meaning in
the structured `operation_intent` field instead of turning the phrase into
`dst` text. Use this shape when calling `skill.filesystem` or
`filesystem.copy_file`:

```json
{
  "operation_intent": {
    "destination": {
      "kind": "directory",
      "path": "C:/path/to/the/explicit/directory",
      "reference": "previous_turn_explicit_directory"
    },
    "name_policy": "auto_generate_copy_name",
    "user_delegated_name": true,
    "avoid_conflict": true
  }
}
```

Rules:

- `destination.path` must be the actual path resolved from conversation
  context, never a natural-language phrase such as `同目录下啊`.
- Use `kind: "file"` only when the user or context provides the concrete
  destination filename.
- Use `kind: "directory"` when the destination is a folder; let runtime create
  the concrete copy filename and show it in the confirmation request.
- `dst` remains supported for legacy calls, but do not invent a filename from
  filler words when `operation_intent` can express the meaning.

## Rename Intent

When the user asks to rename a file using a name that the model selected from
the file content or prior conversation, include a structured `name_intent`
inside `operation_intent`. The supported fields are:

```json
{
  "operation_intent": {
    "destination": {
      "kind": "directory",
      "path": "C:/path/to/the/source-directory",
      "reference": "same_directory"
    },
    "name_policy": "content_based",
    "user_delegated_name": true,
    "name_intent": {
      "mode": "model_proposed",
      "display_name": "Agent-First 工程师简历：AI 智能体平台研发 × 高可靠后端实践",
      "filename_stem": "Agent-First_工程师简历_AI_智能体平台研发_高可靠后端实践",
      "extension_policy": "preserve_source",
      "source_basis": "file_content"
    }
  }
}
```

The runtime owns filename safety, extension preservation, directory boundaries,
approval, and post-execution checks. Never send conversational placeholders such
as “你起的名字” as `dst`.

## Scripts

Run from the skill directory (or use absolute script paths):

```bash
# Linux style paths
python scripts/list_dir.py "/home/dev/workspace"
python scripts/read_file.py "/etc/hosts" --offset 0 --limit 200 --encoding auto
python scripts/write_text.py "/tmp/output.txt" --text "Hello" --overwrite

# Windows style paths
python scripts/list_dir.py "D:/workspace_for_ai/my-agents"
python scripts/read_file.py "D:/tmp/readme.txt" --offset 0 --limit 200 --encoding auto
python scripts/write_text.py "D:/output.txt" --text "Hello" --overwrite
python scripts/replace_text.py "D:/resume.tex" --old-text "Old Name" --new-text "New Name"

# Windows PowerShell-friendly (explicit flags are safer)
python scripts/write_text.py --path "D:/output.txt" --text "Hello world" --overwrite
python scripts/replace_text.py --path "D:/resume.tex" --old-text "Old Name" --new-text "New Name"
python scripts/copy_file.py --src "D:/tmp/source.txt" --dst "D:/tmp/backup/source.txt"
python scripts/move_file.py --src "D:/tmp/old.txt" --dst "D:/tmp/new.txt"

# Common actions
python scripts/copy_file.py "/tmp/source.txt" "/tmp/backup/source.txt"
python scripts/move_file.py "/tmp/old.txt" "/tmp/new.txt"
python scripts/delete_path.py "/tmp/remove.txt" --force
python scripts/delete_path.py "/tmp/remove-dir" --recursive --force
python scripts/make_dir.py "/tmp/new-folder"
python scripts/path_exists.py "/tmp/new-folder"
python scripts/path_stat.py "/tmp/new-folder"
```

## Robust Invocation

- Prefer explicit flags when the model might omit positionals:
  - `--path` for single-path commands (also accepts `--file-path`, `--filepath`, `--target`)
  - `--src/--dst` for copy/move (also accepts `--source`, `--from`, `--destination`, `--dest`, `--to`)
  - `--text` (or positional text) for `write_text.py` (`--content`/`--data` are aliases)
  - In PowerShell, prefer quoted `--text "..."` when text contains spaces
- Copy/move behavior:
  - If `--dst` points to an existing directory, the source is placed under that directory using the source name.
  - Use `--overwrite` only when replacing an existing target path.
- The scripts tolerate unknown extra flags to reduce parse failures, but missing required
  values still return a clear error message.
- Default fallbacks to reduce failures:
  - `list_dir.py`, `path_exists.py`, `path_stat.py` default to current working directory.
  - `make_dir.py` defaults to current working directory and reports `EXISTS` if present.
  - `read_file.py` defaults to `README.md`/`readme.md` in current directory when present.
  - `write_text.py` defaults to the OS temp directory under `my-agents/output.txt`.
- `delete_path.py` requires `--force` or `--force-file` for file deletion, and
  `--recursive --force` for directories.
- On Windows, `delete_path.py` retries read-only targets after adjusting file permissions.
- `read_file.py` supports `--encoding auto` and falls back to common encodings.

## Path Guidance

- Prefer absolute paths (for example `D:/workspace_for_ai/my-agents/` or `/home/dev/workspace`).
- Normalize slashes to `/` in documentation or outputs when possible.
- If a path contains spaces, keep it quoted in examples.

## Platform Notes

- The runtime system type is available in the injected environment prompt. Use matching path style by default.
- Windows: accepts both `D:/path` and `D:\path`; UNC paths are supported when quoted.
- Linux: supports `/path` and `~` (home directory) expansion; paths are usually case-sensitive.

## Safety

- Do not delete or overwrite files unless the user explicitly asked.
- Confirm the target path when the operation could affect multiple files.
