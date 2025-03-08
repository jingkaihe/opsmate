from opsmate.dbq.dbq import Worker
from opsmate.libs.config import config
import asyncio
from sqlmodel import create_engine, Session, text
import structlog
import signal

logger = structlog.get_logger()


async def main(worker_count: int = 10, worker_queue: str = "default"):
    engine = create_engine(
        config.db_url,
        connect_args={"check_same_thread": False, "timeout": 20},
        # echo=True,
    )
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.close()

    session = Session(engine)
    worker = Worker(session, worker_count, queue_name=worker_queue)

    def handle_signal(signal_number, frame):
        logger.info("Received signal", signal_number=signal_number)
        asyncio.create_task(worker.stop())

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
