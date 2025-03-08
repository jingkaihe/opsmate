from typing import List, Any, Dict, Callable, Awaitable, Optional, Protocol, Type
from sqlmodel import Column, JSON
from enum import Enum
from datetime import datetime, UTC
from sqlmodel import (
    SQLModel as _SQLModel,
    Session,
    MetaData,
    Field,
    update,
    select,
    func,
    col,
)
from sqlalchemy.orm import registry
import importlib
import asyncio
import structlog
import time
import traceback
import inspect

logger = structlog.get_logger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


DEFAULT_PRIORITY = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_QUEUE_NAME = "default"


class SQLModel(_SQLModel, registry=registry()):
    metadata = MetaData()


class TaskItem(SQLModel, table=True):
    id: int = Field(primary_key=True)
    args: List[Any] = Field(sa_column=Column(JSON))
    kwargs: Dict[str, Any] = Field(sa_column=Column(JSON))
    func: str

    result: Any = Field(sa_column=Column(JSON))
    error: Optional[str] = Field(default=None, nullable=True)
    status: TaskStatus = Field(default=TaskStatus.PENDING, index=True)

    generation_id: int = Field(default=1)
    created_at: datetime = Field(default=datetime.now(UTC))
    updated_at: datetime = Field(default=datetime.now(UTC))

    priority: int = Field(default=DEFAULT_PRIORITY)
    queue_name: str = Field(default=DEFAULT_QUEUE_NAME)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    wait_until: datetime = Field(default=datetime.now(UTC))


class BackOffFunc(Protocol):
    """
    A function that returns a datetime object for the next retry.
    """

    def __call__(self, retry_count: int) -> datetime:
        """
        Calculate the wait time for the next retry.

        Parameters:
            retry_count (int): The number of retries so far.

        Returns:
            datetime: The datetime when the task should be retried.
        """
        ...


class RetryException(Exception):
    """
    A base class for exceptions that should be retried.
    """

    pass


class Task:
    """
    Task represents the executable to run runnable on the dbq worker.

    In most cases you don't need to use this class directly but enqueue a function directly.
    That being said some time you want granular control over the task like standardising the
    max retries or backoff function and how error are handled.
    """

    def __init__(
        self,
        max_retries: int = 3,
        back_off_func: BackOffFunc = None,
        executable: Callable[..., Awaitable[Any]] = None,
        retry_on: List[Type[Exception]] = [],
        priority: int = DEFAULT_PRIORITY,
    ):
        self.max_retries = max_retries
        self.back_off_func = back_off_func
        if not asyncio.iscoroutinefunction(executable):
            raise TypeError("Executable must be an async function")
        self.executable = executable
        self.priority = priority
        self.retry_on = retry_on

    async def run(
        self,
        *args: List[Any],
        ctx: Dict[str, Any] = {},
        **kwargs: Dict[str, Any],
    ):
        try:
            if "ctx" in inspect.signature(self.executable).parameters:
                return await self.executable(*args, ctx=ctx, **kwargs)
            else:
                return await self.executable(*args, **kwargs)
        except Exception as e:
            # when no retry on is provided, we treat it as a retryable error
            if len(self.retry_on) == 0:
                raise RetryException(e)

            # if the error is in the retry on list, we raise a RetryException
            if isinstance(e, tuple(self.retry_on)):
                raise RetryException(e)

            # otherwise we raise the original error
            raise

    async def before_run(self, task_item: TaskItem, ctx: Dict[str, Any] = {}):
        """
        This method is called before the task is run.
        """
        pass

    async def on_failure(
        self, task_item: TaskItem, error: Exception, ctx: Dict[str, Any] = {}
    ):
        """
        This method is called when the task fails.
        """
        pass

    async def on_success(self, task_item: TaskItem, ctx: Dict[str, Any] = {}):
        """
        This method is called when the task succeeds.
        """
        pass


def dbq_task(
    max_retries: int = 3,
    back_off_func: BackOffFunc = lambda _retry_cnt: datetime.now(UTC),
    priority: int = DEFAULT_PRIORITY,
    retry_on: List[Type[Exception]] = [],
    task_type: Type[Task] = Task,
):
    """
    A decorator for retrying a function call with exponential backoff.

    Parameters:
        max_retries (int): The maximum number of retries before giving up.
        back_off_func (BackOffFunc): A function that returns a datetime object for the next retry.
        priority (int): The priority of the task.
        retry_on (List[Type[Exception]]): A list of exceptions that should be retried.
        task_type (Type[Task]): The type of task to use. Default to Task if not provided.
    """

    def decorator(func):
        task = task_type(
            max_retries=max_retries,
            back_off_func=back_off_func,
            executable=func,
            priority=priority,
            retry_on=retry_on,
        )
        task.__name__ = func.__name__
        task.__module__ = func.__module__
        return task

    return decorator


def enqueue_task(
    session: Session,
    fn: Callable[..., Awaitable[Any]] | Task,
    *args: List[Any],
    queue_name: str = DEFAULT_QUEUE_NAME,
    priority: int | None = None,
    max_retries: int | None = None,
    **kwargs: Dict[str, Any],
):
    """
    Enqueue a task to be executed by the worker.

    Parameters:
        session (Session): The database session to use.
        fn (Callable[..., Awaitable[Any]] | Task): The function to execute - can be a function or a Task object.
        *args (List[Any]): The arguments to pass to the function.
        queue_name (str): The name of the queue to enqueue the task to. Defaults to DEFAULT_QUEUE_NAME.
        priority (int | None): The priority of the task. Default to DEFAULT_PRIORITY if not provided.
        max_retries (int | None): The maximum number of retries for the task. Default to DEFAULT_MAX_RETRIES if not provided.
        **kwargs (Dict[str, Any]): The keyword arguments to pass to the function.
    """
    fn_module = fn.__module__
    fn_name = fn.__name__

    task = TaskItem(
        func=f"{fn_module}.{fn_name}",
        args=args,
        kwargs=kwargs,
        queue_name=queue_name,
    )

    if max_retries is None:
        if isinstance(fn, Task):
            task.max_retries = fn.max_retries
        else:
            task.max_retries = DEFAULT_MAX_RETRIES
    else:
        task.max_retries = max_retries

    if priority is None:
        if isinstance(fn, Task):
            task.priority = fn.priority
        else:
            task.priority = DEFAULT_PRIORITY
    else:
        task.priority = priority

    session.add(task)
    session.commit()

    return task.id


def dequeue_task(session: Session, queue_name: str = DEFAULT_QUEUE_NAME):
    task = session.exec(
        select(TaskItem)
        .where(TaskItem.status == TaskStatus.PENDING)
        .where(TaskItem.wait_until <= datetime.now(UTC))
        .where(TaskItem.queue_name == queue_name)
        .order_by(TaskItem.priority.desc())
    ).first()

    if not task:
        return None

    # an optimistic lock to prevent race condition
    result = session.exec(
        update(TaskItem)
        .where(TaskItem.id == task.id)
        .where(TaskItem.generation_id == task.generation_id)
        .values(
            status=TaskStatus.RUNNING,
            generation_id=task.generation_id + 1,
            updated_at=datetime.now(UTC),
        )
    )
    if result.rowcount == 0:
        return dequeue_task(session, queue_name=queue_name)

    session.refresh(task)
    return task


async def await_task_completion(
    session: Session,
    task_id: int,
    timeout: float = 5,
    interval: float = 0.1,
) -> TaskItem:
    start = time.time()
    while True:
        task = session.exec(select(TaskItem).where(TaskItem.id == task_id)).first()
        if task.status == TaskStatus.COMPLETED or task.status == TaskStatus.FAILED:
            return task
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Task {task_id} did not complete within {timeout} seconds"
            )
        await asyncio.sleep(interval)


class Worker:
    def __init__(
        self,
        session: Session,
        concurrency: int = 1,
        context: Dict[str, Any] = {},
        queue_name: str = DEFAULT_QUEUE_NAME,
    ):
        self.session = session
        self.running = True
        self.lock = asyncio.Lock()
        self.concurrency = concurrency
        self.context = context
        self.queue_name = queue_name
        if self.context.get("session") is None:
            self.context["session"] = self.session

    async def start(self):
        logger.info(
            "starting dbq (database queue) worker",
            concurrency=self.concurrency,
            queue_name=self.queue_name,
        )
        tasks = [self._start(coroutine_id) for coroutine_id in range(self.concurrency)]
        logger.info("dbq coroutines started", concurrency=self.concurrency)
        await asyncio.gather(*tasks)
        logger.info("dbq stopped")

    async def _start(self, coroutine_id: int):
        while True:
            async with self.lock:
                if not self.running:
                    break
            await self._run(coroutine_id)
        logger.info("dbq coroutine stopped", coroutine_id=coroutine_id)

    async def _on_success(
        self, task: TaskItem, fn: Callable[..., Awaitable[Any]] | Task
    ):
        if not isinstance(fn, Task):
            return

        try:
            await fn.on_success(task, self.context)
        except Exception as e:
            logger.error("error on task success", task_id=task.id, error=str(e))

    async def _on_failure(
        self,
        task: TaskItem,
        fn: Callable[..., Awaitable[Any]] | Task,
        error: Exception,
    ):
        if not isinstance(fn, Task):
            return

        try:
            await fn.on_failure(task, error, self.context)
        except Exception as e:
            logger.error("error on task failure", task_id=task.id, error=str(e))

    async def _before_run(
        self, task: TaskItem, fn: Callable[..., Awaitable[Any]] | Task
    ):
        if not isinstance(fn, Task):
            return

        try:
            await fn.before_run(task, self.context)
        except Exception as e:
            logger.error("error on task before run", task_id=task.id, error=str(e))

    async def _run(self, coroutine_id: int):
        task = dequeue_task(self.session, queue_name=self.queue_name)

        if not task:
            await asyncio.sleep(0.1)
            return
        logger.info("dequeue task", task_id=task.id, coroutine_id=coroutine_id)
        try:
            logger.info("importing function", func=task.func)
            fn_module, fn_name = task.func.rsplit(".", 1)
            fn = getattr(importlib.import_module(fn_module), fn_name)
            logger.info("imported function", func=task.func)

            await self._before_run(task, fn)
            result = await self.maybe_context_fn(fn, task.args, task.kwargs)

            logger.info(
                "task completed",
                task_id=task.id,
                result=result,
                coroutine_id=coroutine_id,
            )
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.updated_at = datetime.now(UTC)
            task.generation_id = task.generation_id + 1
            task.wait_until = datetime.now(UTC)  # Reset wait_until on success
            self.session.commit()
            await self._on_success(task, fn)
        except RetryException as e:
            if task.retry_count >= task.max_retries:
                logger.error(
                    "error running task, max retries exceeded",
                    task_id=task.id,
                    error=str(e),
                    stack_trace=traceback.format_exc(),
                    coroutine_id=coroutine_id,
                )
                task.error = str(e)
                task.status = TaskStatus.FAILED
            else:
                task.retry_count += 1
                logger.warning(
                    "error running task, will retry",
                    task_id=task.id,
                    error=str(e),
                    stack_trace=traceback.format_exc(),
                    coroutine_id=coroutine_id,
                )
                task.status = TaskStatus.PENDING

                if isinstance(fn, Task):
                    task.wait_until = fn.back_off_func(task.retry_count)
                else:
                    task.wait_until = datetime.now(UTC)
                logger.info(
                    "retrying task after backoff",
                    task_id=task.id,
                    wait_until=task.wait_until,
                    coroutine_id=coroutine_id,
                )

            task.updated_at = datetime.now(UTC)
            task.generation_id = task.generation_id + 1
            self.session.commit()
            await self._on_failure(task, fn, e)
            return
        except Exception as e:
            logger.error(
                "error running task",
                task_id=task.id,
                error=str(e),
                coroutine_id=coroutine_id,
            )
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.updated_at = datetime.now(UTC)
            task.generation_id = task.generation_id + 1
            self.session.commit()

    async def maybe_context_fn(
        self,
        fn: Callable[..., Awaitable[Any]] | Task,
        args: List[Any],
        kwargs: Dict[str, Any],
    ):
        """Check if the fn has a ctx argument, if so, add the session to it"""

        if isinstance(fn, Task):
            run = fn.run
        else:
            run = fn

        if "ctx" in inspect.signature(run).parameters:
            return await run(*args, ctx=self.context, **kwargs)
        return await run(*args, **kwargs)

    def queue_size(self):
        return self.session.exec(
            select(func.count(col(TaskItem.id)))
            .select_from(TaskItem)
            .where(TaskItem.status == TaskStatus.PENDING)
        ).one()

    def inflight_size(self):
        return self.session.exec(
            select(func.count(col(TaskItem.id)))
            .select_from(TaskItem)
            .where(TaskItem.status == TaskStatus.RUNNING)
        ).one()

    def idle(self):
        return self.queue_size() == 0 and self.inflight_size() == 0

    async def stop(self):
        async with self.lock:
            self.running = False
