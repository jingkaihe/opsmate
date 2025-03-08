from opsmate.knowledgestore.models import aconn, Category
from opsmate.ingestions.base import Document
from opsmate.ingestions.chunk import chunk_document
from opsmate.ingestions.fs import FsIngestion
from opsmate.ingestions.github import GithubIngestion
from opsmate.ingestions.models import IngestionRecord, DocumentRecord
from opsmate.libs.config import config
from opsmate.dbq.dbq import enqueue_task, dbq_task
from opsmate.dino import dino
from opsmate.textsplitters import splitter_from_config
from typing import Dict, Any, List
from datetime import datetime, UTC, timedelta
import asyncio
import uuid
import json
import random
import structlog

logger = structlog.get_logger()


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


async def categorize_kb(kb: Dict[str, Any]):
    categories = await categorize(kb["content"])
    kb["categories"] = [cat.value for cat in categories]
    return kb


def backoff_func(retry_count: int):
    return datetime.now(UTC) + timedelta(
        milliseconds=2 ** (retry_count - 1) + random.uniform(0, 10)
    )


@dbq_task(
    retry_on=(Exception,),
    max_retries=10,
    back_off_func=backoff_func,
)
async def chunk_and_store(
    ingestion_record_id: int,
    splitter_config: Dict[str, Any] = {},
    doc: Dict[str, Any] = {},
    ctx: Dict[str, Any] = {},
):
    session = ctx["session"]

    ingestion_record = await IngestionRecord.find_by_id(session, ingestion_record_id)
    if ingestion_record is None:
        logger.error(
            "ingestion record not found",
            ingestion_record_id=ingestion_record_id,
        )
        return

    doc = Document(**doc)
    path = doc.metadata["path"]

    doc_record = await DocumentRecord.find_by_ingestion_id_and_path(
        session, ingestion_record.id, path
    )
    if doc_record is not None:
        if (
            doc_record.sha == doc.metadata.get("sha", "")
            and doc_record.chunk_config == splitter_config
        ):
            logger.info(
                "document already exists",
                ingestion_record_id=ingestion_record.id,
                path=path,
            )
            return
    else:
        doc_record = await DocumentRecord.find_or_create(
            session,
            ingestion_record.id,
            path,
            doc.metadata.get("sha", ""),
            splitter_config,
        )

    splitter = splitter_from_config(splitter_config)
    db_conn = await aconn()
    table = await db_conn.open_table("knowledge_store")

    kbs = []
    async for chunk in chunk_document(splitter=splitter, document=doc):
        kbs.append(
            {
                "uuid": str(uuid.uuid4()),
                "id": chunk.id,
                # "summary": chunk.metadata["summary"],
                "categories": [],
                "data_source_provider": doc.data_provider,
                "data_source": doc.data_source,
                "metadata": json.dumps(chunk.metadata),
                "path": path,
                "content": chunk.content,
                "created_at": datetime.now(),
            }
        )

    if config.categorise:
        tasks = [categorize_kb(kb) for kb in kbs]
        await asyncio.gather(*tasks)

    logger.info(
        "deleting chunks from data source",
        data_source_provider=doc.data_provider,
        data_source=doc.data_source,
        path=path,
    )
    await table.delete(
        f"data_source_provider = '{doc.data_provider}'"
        f"AND data_source = '{doc.data_source}'"
        f"AND path = '{path}'"
    )

    await table.add(kbs)
    await table.optimize()

    doc_record.update_chunk_count(session, len(kbs))

    logger.info(
        "chunks stored",
        data_provider=doc.data_provider,
        data_source=doc.data_source,
        path=path,
        num_kbs=len(kbs),
    )


@dbq_task(
    retry_on=(Exception,),
    max_retries=10,
    back_off_func=backoff_func,
)
async def ingest(
    ingestor_type: str,
    ingestor_config: Dict[str, Any],
    splitter_config: Dict[str, Any] = {},
    ctx: Dict[str, Any] = {},
):
    session = ctx["session"]

    ingestion = ingestor_from_config(ingestor_type, ingestor_config)

    ingestion_record = await IngestionRecord.find_or_create(
        session, ingestor_type, ingestor_config
    )

    async for doc in ingestion.load():
        logger.info(
            "ingesting document",
            ingestor_type=ingestor_type,
            ingestor_config=ingestor_config,
            splitter_config=splitter_config,
            doc_path=doc.metadata["path"],
        )
        enqueue_task(
            session,
            chunk_and_store,
            ingestion_record.id,
            splitter_config=splitter_config,
            doc=doc.model_dump(),
        )


@dbq_task(
    retry_on=(Exception,),
    max_retries=10,
    back_off_func=backoff_func,
)
async def delete_ingestion(ingestion_record_id: int, ctx: Dict[str, Any] = {}):
    session = ctx["session"]

    ingestion_record = await IngestionRecord.find_by_id(session, ingestion_record_id)
    if ingestion_record is None:
        logger.error(
            "ingestion record not found",
            ingestion_record_id=ingestion_record_id,
        )
        return

    session.delete(ingestion_record)
    session.commit()

    # remove all documents from lancedb
    db_conn = await aconn()
    table = await db_conn.open_table("knowledge_store")
    await table.delete(
        f"data_source_provider = '{ingestion_record.data_source_provider}'"
        f"AND data_source = '{ingestion_record.data_source}'"
    )


def ingestor_from_config(name: str, config: Dict[str, Any]):
    if name == "github":
        return GithubIngestion(**config)
    elif name == "fs":
        return FsIngestion(**config)
    else:
        raise ValueError(f"Unknown ingestor type: {name}")
