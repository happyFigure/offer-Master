from __future__ import annotations

import unittest
from pathlib import Path


class FilesystemSkillExecutorTest(unittest.TestCase):
    def test_executor_requires_structured_operation_instead_of_classifying_user_text(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            src = Path(directory) / "resume.tex"
            src.write_text("resume", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=_fake_script_root()).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="把文件名改成 resume-final.tex",
                    input_payload={"user_task": "把文件名改成 resume-final.tex", "src": str(src)},
                ),
                _context(),
            )

        self.assertEqual("failed", result.status)
        self.assertEqual(["operation"], result.missing_information)
        self.assertEqual("STRUCTURED_OPERATION_REQUIRED", result.raw_result["error_code"])
        self.assertFalse(result.requires_user_action)
        self.assertTrue(result.raw_result["recoverable"])
        self.assertEqual("continue_model_loop", result.raw_result["next_action"])

    def test_capabilities_declares_high_level_filesystem_skill(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, FILESYSTEM_SKILL_EXECUTOR_ID
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        capability = FilesystemSkillExecutor(script_root=_fake_script_root()).capabilities()[0]

        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, capability.capability_id)
        self.assertEqual("skill", capability.kind)
        self.assertEqual(FILESYSTEM_SKILL_EXECUTOR_ID, capability.executor_id)

    def test_structured_rename_returns_approval_request_before_mutation(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            src = Path(directory) / "source.tex"
            dst = Path(directory) / "resume-final.tex"
            src.write_text("resume", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=_fake_script_root()).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="根据模型命名结果重命名文件",
                    input_payload={
                        "user_task": "把文件名改成模型确定的名字",
                        "operation": "rename_file",
                        "src": str(src),
                        "operation_intent": {
                            "destination": {"kind": "file", "path": str(dst)},
                            "name_intent": {
                                "mode": "model_proposed",
                                "filename": dst.name,
                                "source_basis": "file_content",
                            },
                        },
                    },
                ),
                _context(),
            )

            self.assertEqual("waiting_user", result.status)
            self.assertTrue(result.requires_user_action)
            self.assertEqual("rename_file", result.raw_result["operation"])
            self.assertEqual(str(dst), result.raw_result["approval_payload"]["dst"])
            self.assertTrue(src.exists())
            self.assertFalse(dst.exists())

    def test_content_based_copy_requires_model_name_intent_after_file_was_read(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            src = Path(directory) / "resume.tex"
            src.write_text("Name: Liu Hanqing\nFocus: AI Agent", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=_fake_script_root()).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="读取后根据内容复制并命名",
                    input_payload={
                        "user_task": "根据文件内容自己起一个名字后复制",
                        "operation": "copy_file",
                        "src": str(src),
                        "operation_intent": {
                            "destination": {"kind": "directory", "path": directory},
                            "name_policy": "content_based",
                            "user_delegated_name": True,
                        },
                        "context_metadata": {
                            "active_file": {"path": str(src)},
                            "last_file_operation_result": {
                                "operation": "read_file",
                                "focus_path": str(src),
                                "completed": True,
                            },
                        },
                    },
                ),
                _context(),
            )

        self.assertEqual("failed", result.status)
        self.assertEqual(["name_intent"], result.missing_information)
        self.assertEqual("CONTENT_BASED_NAME_REQUIRED", result.raw_result["error_code"])
        self.assertTrue(result.raw_result["recoverable"])
        self.assertEqual("continue_model_loop", result.raw_result["next_action"])
        self.assertNotIn("approval_request", result.raw_result)
        self.assertFalse((Path(directory) / "resume-副本.tex").exists())

    def test_missing_destination_stays_in_native_model_loop(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            src = Path(directory) / "resume.tex"
            src.write_text("Name: Liu Hanqing", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=_fake_script_root()).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="复制文件并根据内容自行命名",
                    input_payload={
                        "user_task": "复制这个文件到同目录，并根据内容自己起一个合适的名字",
                        "operation": "copy_file",
                        "src": str(src),
                    },
                ),
                _context(),
            )

        self.assertEqual("failed", result.status)
        self.assertEqual(["dst"], result.missing_information)
        self.assertEqual("MISSING_REQUIRED_ARGUMENT", result.raw_result["error_code"])
        self.assertTrue(result.raw_result["recoverable"])
        self.assertEqual("continue_model_loop", result.raw_result["next_action"])
        self.assertFalse(result.requires_user_action)
        self.assertNotIn("approval_request", result.raw_result)

    def test_content_based_rename_requires_model_name_intent_after_file_was_read(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            src = Path(directory) / "resume.tex"
            src.write_text("Name: Liu Hanqing\nFocus: AI Agent", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=_fake_script_root()).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="读取后根据内容重命名",
                    input_payload={
                        "user_task": "根据文件内容自己起一个名字后重命名",
                        "operation": "rename_file",
                        "src": str(src),
                        "operation_intent": {
                            "destination": {"kind": "directory", "path": directory},
                            "name_policy": "content_based",
                            "user_delegated_name": True,
                        },
                        "context_metadata": {
                            "active_file": {"path": str(src)},
                            "last_file_operation_result": {
                                "operation": "read_file",
                                "focus_path": str(src),
                                "completed": True,
                            },
                        },
                    },
                ),
                _context(),
            )

        self.assertEqual("failed", result.status)
        self.assertEqual(["name_intent"], result.missing_information)
        self.assertEqual("CONTENT_BASED_NAME_REQUIRED", result.raw_result["error_code"])
        self.assertTrue(result.raw_result["recoverable"])
        self.assertEqual("continue_model_loop", result.raw_result["next_action"])
        self.assertFalse(src.exists())

    def test_placeholder_destination_is_rejected_without_runtime_repair(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            src = Path(directory) / "source.tex"
            bad_dst = Path(directory) / "你起的名字.tex"
            src.write_text("resume", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=_fake_script_root()).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="把文件名改成你起的名字",
                    input_payload={
                        "user_task": "把文件名改成你起的名字",
                        "operation": "rename_file",
                        "src": str(src),
                        "dst": str(bad_dst),
                    },
                ),
                _context(),
            )

            self.assertEqual("failed", result.status)
            self.assertEqual("MISSING_REQUIRED_ARGUMENT", result.raw_result["error_code"])
            self.assertEqual(["dst"], result.raw_result["missing_args"])
            self.assertTrue(src.exists())
            self.assertFalse(bad_dst.exists())

    def test_structured_copy_directory_intent_without_model_name_stays_recoverable(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            source_directory = Path(directory) / "source"
            destination_directory = Path(directory) / "target"
            source_directory.mkdir()
            destination_directory.mkdir()
            src = source_directory / "resume.tex"
            src.write_text("resume", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=_fake_script_root()).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="复制文件",
                    input_payload={
                        "user_task": "给我复制一下",
                        "operation": "copy_file",
                        "src": str(src),
                        "operation_intent": {
                            "destination": {"kind": "directory", "path": str(destination_directory)},
                            "name_policy": "copy_suffix",
                            "user_delegated_name": True,
                            "avoid_conflict": True,
                        },
                    },
                ),
                _context(),
            )

            self.assertEqual("failed", result.status)
            self.assertEqual(["dst"], result.missing_information)
            self.assertEqual("MISSING_REQUIRED_ARGUMENT", result.raw_result["error_code"])
            self.assertTrue(result.raw_result["recoverable"])
            self.assertEqual("continue_model_loop", result.raw_result["next_action"])
            self.assertNotIn("approval_request", result.raw_result)
            self.assertFalse((destination_directory / "resume-副本.tex").exists())

    def test_confirmed_rename_executes_script_and_postchecks_target(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            root = _write_move_script(Path(directory) / "skill-root")
            src = Path(directory) / "old.tex"
            dst = Path(directory) / "new.tex"
            src.write_text("resume", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=root).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="重命名文件",
                    input_payload={
                        "user_task": "重命名文件",
                        "operation": "rename_file",
                        "src": str(src),
                        "dst": str(dst),
                    },
                ),
                _context(confirmed=True),
            )

            self.assertEqual("succeeded", result.status)
            self.assertFalse(src.exists())
            self.assertTrue(dst.exists())
            self.assertEqual("move_file.py", result.raw_result["internal_script"])
            self.assertTrue(result.raw_result["filesystem_trace"]["postcheck"]["completed"])

    def test_confirmed_copy_executes_script_and_preserves_source(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            root = _write_copy_script(Path(directory) / "skill-root")
            src = Path(directory) / "resume.tex"
            dst = Path(directory) / "resume-copy.tex"
            src.write_text("resume", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=root).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="复制文件",
                    input_payload={
                        "user_task": "复制文件",
                        "operation": "copy_file",
                        "src": str(src),
                        "dst": str(dst),
                    },
                ),
                _context(confirmed=True),
            )

            self.assertEqual("succeeded", result.status)
            self.assertTrue(src.exists())
            self.assertTrue(dst.exists())
            self.assertEqual("resume", dst.read_text(encoding="utf-8"))
            self.assertEqual("copy_file.py", result.raw_result["internal_script"])

    def test_postcheck_failure_is_reported_as_runtime_failure(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            root = _write_noop_move_script(Path(directory) / "skill-root")
            src = Path(directory) / "old.tex"
            dst = Path(directory) / "new.tex"
            src.write_text("resume", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=root).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="重命名文件",
                    input_payload={
                        "user_task": "重命名文件",
                        "operation": "rename_file",
                        "src": str(src),
                        "dst": str(dst),
                    },
                ),
                _context(confirmed=True),
            )

            self.assertEqual("failed", result.status)
            self.assertEqual("FILESYSTEM_POSTCHECK_FAILED", result.raw_result["error_code"])
            self.assertFalse(result.raw_result["filesystem_trace"]["postcheck"]["completed"])
            self.assertTrue(src.exists())
            self.assertFalse(dst.exists())

    def test_structured_path_exists_uses_explicit_path_without_operation_inference(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            root = _write_path_exists_script(Path(directory) / "skill-root")
            target = Path(directory) / "resume.tex"
            target.write_text("resume", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=root).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="检查路径",
                    input_payload={
                        "user_task": "这个文件是否村",
                        "operation": "path_exists",
                        "path": str(target),
                    },
                ),
                _context(),
            )

            self.assertEqual("succeeded", result.status)
            self.assertEqual("path_exists", result.raw_result["operation"])
            self.assertTrue(result.raw_result["result"]["result"]["exists"])

    def test_structured_read_file_uses_explicit_path(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            root = _write_read_file_script(Path(directory) / "skill-root")
            target = Path(directory) / "resume.txt"
            target.write_text("姓名：刘汉卿", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=root).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="读取文件",
                    input_payload={"user_task": "文件内容", "operation": "read_file", "path": str(target)},
                ),
                _context(),
            )

            self.assertEqual("succeeded", result.status)
            self.assertEqual("read_file", result.raw_result["operation"])
            self.assertIn("姓名：刘汉卿", result.observation)

    def test_typo_path_question_without_structured_operation_is_recoverable(self) -> None:
        from tempfile import TemporaryDirectory

        from app.agent_runtime.agent_as_tool import AgentTask
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        with TemporaryDirectory() as directory:
            target = Path(directory) / "resume.tex"
            target.write_text("resume", encoding="utf-8")
            result = FilesystemSkillExecutor(script_root=_fake_script_root()).call(
                AgentTask(
                    capability_id="skill.filesystem",
                    goal="看下这个文件是否村",
                    input_payload={
                        "user_task": "看下这个文件是否村",
                        "context_metadata": {"active_file": {"path": str(target)}},
                    },
                ),
                _context(),
            )

        self.assertEqual("failed", result.status)
        self.assertEqual("STRUCTURED_OPERATION_REQUIRED", result.raw_result["error_code"])
        self.assertEqual(["operation"], result.missing_information)
        self.assertFalse(result.requires_user_action)
        self.assertTrue(result.raw_result["recoverable"])
        self.assertEqual("continue_model_loop", result.raw_result["next_action"])


def _context(*, confirmed: bool = False):
    from app.agent_runtime.agent_as_tool import AgentRuntimeContext

    return AgentRuntimeContext(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        permission_scope={"source_type": "agent_chat", "user_confirmed": confirmed},
    )


def _fake_script_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_move_script(root: Path) -> Path:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "move_file.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--src', required=True)\n"
        "parser.add_argument('--dst', required=True)\n"
        "parser.add_argument('--overwrite', action='store_true')\n"
        "args = parser.parse_args()\n"
        "src = Path(args.src)\n"
        "dst = Path(args.dst)\n"
        "if dst.exists() and not args.overwrite:\n"
        "    raise SystemExit(1)\n"
        "dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "src.replace(dst)\n"
        "print(f'MOVED:{src}->{dst}')\n",
        encoding="utf-8",
    )
    return root


def _write_noop_move_script(root: Path) -> Path:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "move_file.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--src', required=True)\n"
        "parser.add_argument('--dst', required=True)\n"
        "parser.add_argument('--overwrite', action='store_true')\n"
        "parser.parse_args()\n"
        "print('MOVED')\n",
        encoding="utf-8",
    )
    return root


def _write_copy_script(root: Path) -> Path:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "copy_file.py").write_text(
        "import argparse\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--src', required=True)\n"
        "parser.add_argument('--dst', required=True)\n"
        "parser.add_argument('--overwrite', action='store_true')\n"
        "args = parser.parse_args()\n"
        "src = Path(args.src)\n"
        "dst = Path(args.dst)\n"
        "if dst.exists() and not args.overwrite:\n"
        "    raise SystemExit(1)\n"
        "dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "shutil.copy2(src, dst)\n"
        "print(f'COPIED:{src}->{dst}')\n",
        encoding="utf-8",
    )
    return root


def _write_path_exists_script(root: Path) -> Path:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "path_exists.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--path', required=True)\n"
        "args = parser.parse_args()\n"
        "path = Path(args.path)\n"
        "print(f'EXISTS: file {path}' if path.exists() else f'MISSING: {path}')\n",
        encoding="utf-8",
    )
    return root


def _write_read_file_script(root: Path) -> Path:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "read_file.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--path', required=True)\n"
        "parser.add_argument('--encoding', default='utf-8')\n"
        "parser.add_argument('--offset', type=int, default=0)\n"
        "parser.add_argument('--limit', type=int, default=200)\n"
        "args = parser.parse_args()\n"
        "print(Path(args.path).read_text(encoding='utf-8'))\n",
        encoding="utf-8",
    )
    return root


if __name__ == "__main__":
    unittest.main()
