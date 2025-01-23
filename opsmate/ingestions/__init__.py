from .base import BaseIngestion
from .fs import FsIngestion
from .github import GithubIngestion
from typing import List
from opsmate.libs.config import Config
import structlog
from opsmate.dino import dino
from opsmate.knowledgestore.models import init_table, aconn, Category
from opsmate.ingestions.base import Chunk
import uuid
import json

logger = structlog.get_logger(__name__)

__all__ = ["BaseIngestion", "FsIngestion", "GithubIngestion"]


@dino(
    model="gpt-4o-mini",
    response_model=List[Category],
)
async def categorize(text: str) -> str:
    f"""
    You are a world class expert in categorizing text.
    Please categorise the text into one or more unique categories:
    """
    return text


async def categorize_chunk(chunk: Chunk):
    categories = await categorize(chunk.content)
    chunk.metadata["categories"] = categories
    return chunk


def ingestions_from_config(cfg: Config) -> List[BaseIngestion]:
    ingestions = []
    github_ingestions = GithubIngestion.from_config(cfg.github_embeddings_config)
    fs_ingestions = FsIngestion.from_config(cfg.fs_embeddings_config)
    ingestions.extend(github_ingestions)
    ingestions.extend(fs_ingestions)

    if cfg.categorise:
        for ingestion in ingestions:
            ingestion.post_chunk_hooks = [categorize_chunk]
    else:
        logger.info("categorise is disabled")

    return ingestions


async def ingest_from_config(cfg: Config) -> List[BaseIngestion]:
    """
    Ingest the data based on the env var config.
    """
    ingestions = ingestions_from_config(cfg)

    await init_table()
    db_conn = await aconn()
    table = await db_conn.open_table("knowledge_store")

    logger.info("start ingesting")
    for ingestion in ingestions:
        logger.info(
            "Ingesting",
            provider_name=ingestion.data_source_provider(),
            data_providersource=ingestion.data_source(),
        )

        async for chunk in ingestion.ingest():
            if "categories" in chunk.metadata:
                categories = [cat.value for cat in chunk.metadata["categories"]]
            else:
                categories = []

            del chunk.metadata["categories"]

            kb = {
                "uuid": str(uuid.uuid4()),
                # "summary": chunk.metadata["summary"],
                "data_source_provider": chunk.metadata["data_source_provider"],
                "data_source": chunk.metadata["data_source"],
                "path": chunk.metadata["path"],
                "categories": categories,
                "metadata": json.dumps(chunk.metadata),
                "content": chunk.content,
            }
            await table.add([kb])

    logger.info("ingestion completed")
