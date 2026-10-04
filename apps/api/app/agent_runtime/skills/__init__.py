__all__ = ["FilesystemSkillExecutor"]


def __getattr__(name: str):
    # Keep package import side-effect free. Lightweight modules such as the
    # operation catalog are imported by routing during startup, and eager-loading
    # the executor here would recreate an agent_as_tool -> routing -> skills cycle.
    if name == "FilesystemSkillExecutor":
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        return FilesystemSkillExecutor
    raise AttributeError(name)
