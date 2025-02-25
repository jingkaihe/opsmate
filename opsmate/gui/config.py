from opsmate.libs.config import Config as OpsmateConfig
from pydantic import Field, model_validator
from opsmate.contexts import k8s_ctx
from opsmate.plugins import PluginRegistry
from typing import List, Self


class Config(OpsmateConfig):
    session_name: str = Field(default="session", alias="OPSMATE_SESSION_NAME")
    token: str = Field(default="", alias="OPSMATE_TOKEN")

    tools: List[str] = Field(
        default=["ShellCommand", "KnowledgeRetrieval"], alias="OPSMATE_TOOLS"
    )
    system_prompt: str = Field(
        alias="OPSMATE_SYSTEM_PROMPT",
        default_factory=k8s_ctx,
    )

    model: str = Field(
        default="gpt-4o",
        alias="OPSMATE_MODEL",
        choices=["gpt-4o", "claude-3-5-sonnet-20241022", "grok-2-1212"],
    )

    review_command: bool = Field(
        default=False,
        alias="OPSMATE_REVIEW_COMMAND",
        description="If true, the command will be reviewed by the user before execution",
    )

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        PluginRegistry.discover(self.plugins_dir)
        return self

    def opsmate_tools(self):
        return PluginRegistry.get_tools_from_list(self.tools)
