from pydantic_settings import BaseSettings
from pydantic import Field, model_validator, field_validator
from pathlib import Path
from typing import Dict, Any, Self
import structlog
import logging
from opsmate.plugins import PluginRegistry
import importlib.util
import time
from functools import wraps

logger = structlog.get_logger(__name__)

default_embeddings_db_path = str(Path.home() / ".opsmate" / "embeddings")
default_db_url = f"sqlite:///{str(Path.home() / '.opsmate' / 'opsmate.db')}"
default_plugins_dir = str(Path.home() / ".opsmate" / "plugins")
default_contexts_dir = str(Path.home() / ".opsmate" / "contexts")
fs_embedding_desc = """
The configuration for the fs embeddings.

This is a dictionary with the following pattern of path=glob_pattern

Example:

your_repo_path=*.md
your_repo_path2=*.txt
"""

github_embedding_desc = """
The configuration for the github embeddings

This is a dictionary with the following pattern of owner/repo:branch=glob_pattern

If the branch is not specified, it will default to main.

Example:

opsmate/opsmate=main=*.md
opsmate/opsmate2=main=*.txt
"""

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_SENTENCE_TRANSFORMERS_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def timer():
    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            logger.info(
                "call completed",
                function=f"{f.__module__}.{f.__name__}",
                time=f"{end - start:.2f}s",
            )
            return result

        return wrapped

    return wrapper


class Config(BaseSettings):
    db_url: str = Field(default=default_db_url, alias="OPSMATE_DB_URL")

    plugins_dir: str = Field(
        default=default_plugins_dir,
        alias="OPSMATE_PLUGINS_DIR",
    )
    contexts_dir: str = Field(
        default=default_contexts_dir,
        alias="OPSMATE_CONTEXTS_DIR",
    )

    embeddings_db_path: str = Field(
        default=default_embeddings_db_path, description="The path to the lance db"
    )
    embedding_registry_name: str = Field(
        default="",
        choices=["openai", "sentence-transformers"],
        description="The name of the embedding registry",
        alias="OPSMATE_EMBEDDING_REGISTRY_NAME",
    )
    embedding_model_name: str = Field(
        default="",
        description="The name of the embedding model",
        alias="OPSMATE_EMBEDDING_MODEL_NAME",
    )
    reranker_model_name: str = Field(
        default="",
        description="The name of the reranker model",
        choices=["answerdotai", "openai", "cohere", "rrf", ""],
        alias="OPSMATE_RERANKER_MODEL_NAME",
    )
    fs_embeddings_config: Dict[str, str] = Field(
        default={}, description=fs_embedding_desc
    )
    github_embeddings_config: Dict[str, str] = Field(
        default={}, description=github_embedding_desc
    )
    categorise: bool = Field(
        default=True, description="Whether to categorise the embeddings"
    )
    splitter_config: Dict[str, Any] = Field(
        default={
            "splitter": "markdown_header",
            "headers_to_split_on": (
                ("##", "h2"),
                ("###", "h3"),
            ),
        },
        description="The splitter to use for the ingestion",
    )

    loglevel: str = Field(default="INFO", alias="OPSMATE_LOGLEVEL")

    @field_validator("embedding_registry_name")
    def validate_embedding_registry_name(cls, v):
        if v == "":
            if cls.transformers_available():
                return "sentence-transformers"
            else:
                return "openai"
        return v

    @field_validator("embedding_model_name")
    def validate_embedding_model_name(cls, v):
        if v == "":
            if cls.transformers_available():
                return DEFAULT_SENTENCE_TRANSFORMERS_EMBEDDING_MODEL
            else:
                return DEFAULT_OPENAI_EMBEDDING_MODEL
        return v

    @classmethod
    def transformers_available(cls):
        return importlib.util.find_spec("transformers") is not None

    # @computed_field
    # @property
    # def embedding_registry(self) -> str:
    #     if self.embedding_registry_name == "":
    #         if self.transformers_available():
    #             return "sentence-transformers"
    #         else:
    #             return "openai"
    #     return self.embedding_registry_name

    # @computed_field
    # @property
    # def embedding_model(self) -> str:
    #     if self.embedding_model_name == "":
    #         if self.transformers_available():
    #             return DEFAULT_SENTENCE_TRANSFORMERS_EMBEDDING_MODEL
    #         else:
    #             return DEFAULT_OPENAI_EMBEDDING_MODEL
    #     return self.embedding_model_name

    @model_validator(mode="after")
    def validate_loglevel(self) -> Self:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.getLevelNamesMapping()[self.loglevel]
            ),
        )
        return self

    @model_validator(mode="after")
    def mkdir(self):
        opsmate_dir = str(Path.home() / ".opsmate")
        Path(opsmate_dir).mkdir(parents=True, exist_ok=True)
        Path(self.plugins_dir).mkdir(parents=True, exist_ok=True)
        Path(self.embeddings_db_path).mkdir(parents=True, exist_ok=True)
        Path(self.contexts_dir).mkdir(parents=True, exist_ok=True)
        return self


config = Config()
