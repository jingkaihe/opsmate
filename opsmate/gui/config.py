from opsmate.libs.config import Config as OpsmateConfig
from pydantic import Field, model_validator
from typing import List, Self
from pathlib import Path
from opsmate.contexts import k8s_ctx
from opsmate.plugins import PluginRegistry
import os
import structlog
import logging


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
    loglevel: str = Field(default="INFO", alias="OPSMATE_LOGLEVEL")
    model: str = Field(
        default="gpt-4o",
        alias="OPSMATE_MODEL",
        choices=["gpt-4o", "claude-3-5-sonnet-20241022", "grok-2-1212"],
    )

    @model_validator(mode="after")
    def validate_loglevel(self) -> Self:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.getLevelNamesMapping()[self.loglevel]
            ),
        )
        return self

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        PluginRegistry.discover(self.plugins_dir)
        return self

    def opsmate_tools(self):
        return PluginRegistry.get_tools_from_list(self.tools)
