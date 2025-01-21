from typing import (
    List,
    Union,
    Callable,
    Coroutine,
    Any,
    ParamSpec,
    TypeVar,
    Awaitable,
)
from pydantic import BaseModel
from .dino import dino
from .types import Message, React, ReactAnswer, Observation, ToolCall, Context
from functools import wraps
import inspect
import structlog
import yaml

logger = structlog.get_logger(__name__)


async def _react_prompt(
    question: str, message_history: List[Message] = [], tool_names: List[BaseModel] = []
):
    """
    <assistant>
    You run in a loop of question, thought, action.
    At the end of the loop you output an answer.
    Use "Question" to describe the question you have been asked.
    Use "Thought" to describe your thought
    Use "Action" to describe the action you are going to take based on the thought.
    Use "Answer" as the final answer to the question.
    </assistant>

    <response format 1>
    During the thought phase you response with the following format:
    thought: ...
    action: ...
    </response_format 1>

    <response format 2>
    When you have an answer, you response with the following format:
    answer: ...
    </response_format 2>

    <important 1>
    When you know how to perform a task, provide the steps as an action rather than giving them as an answer.

    BAD EXAMPLE:
    <react>
    answer: to kill process with pid 1234, use `kill -TERM 1234`
    </react>

    GOOD EXAMPLE:

    <react>
    thought: I need to kill process using the kill command
    action: run `kill -TERM 1234`
    </react>
    </important 1 >

    <important 2>
    If you know the answer straight away, feel free to give the answer without going through the thought process.
    </important 2>

    <important 3>
    Action must be atomic.

    Good: find all the processes that use 100%+ CPU
    Bad: find all the processes that use 100%+ CPU and kill them
    </important 3>
    """

    return [
        Message.user(
            f"""
Here is a list of tools you can use:
{"\n".join(f"<tool>{t.__name__}: {t.__doc__}</tool>" for t in tool_names)}
""",
        ),
        Message.user(question),
        *message_history,
    ]


async def run_react(
    question: str,
    contexts: List[str | Message] = [],
    model: str = "gpt-4o",
    tools: List[ToolCall] = [],
    chat_history: List[Message] = [],
    max_iter: int = 10,
    react_prompt: Callable[
        [str, List[Message], List[ToolCall]], Coroutine[Any, Any, List[Message]]
    ] = _react_prompt,
    **kwargs: Any,
):
    ctxs = []
    for ctx in contexts:
        if isinstance(ctx, str):
            ctxs.append(Message.system(ctx))
        elif isinstance(ctx, Message):
            ctxs.append(ctx)
        else:
            raise ValueError(f"Invalid context type: {type(ctx)}")

    @dino(model, response_model=Observation, tools=tools, **kwargs)
    async def run_action(react: React):
        """
        You are a world class expert to carry out action using the tools you are given.
        Please stictly only carry out the action within the <action>...</action> tag.
        The tools you use must be relevant to the action.
        """
        return [
            *ctxs,
            *message_history,
            Message.assistant(
                f"""
<context>
thought: {react.thoughts}
</context>
            """,
            ),
            Message.user(
                f"""<action>
{react.action}
</action>"""
            ),
        ]

    react = dino(model, response_model=Union[React, ReactAnswer], **kwargs)(
        react_prompt
    )

    message_history = Message.normalise(chat_history)
    for ctx in ctxs:
        message_history.append(ctx)
    for _ in range(max_iter):
        react_result = await react(
            question, message_history=message_history, tool_names=tools
        )
        if isinstance(react_result, React):
            message_history.append(Message.user(react_result.model_dump_json()))
            yield react_result
            observation = await run_action(react_result)

            observation_out = observation.model_dump()
            for idx, tool_output in enumerate(observation.tool_outputs):
                if isinstance(tool_output, ToolCall):
                    observation_out["tool_outputs"][idx] = tool_output.model_dump()
                elif isinstance(tool_output, str):
                    observation_out["tool_outputs"][idx] = tool_output

            message_history.append(Message.user(yaml.dump(observation_out)))
            yield observation
        elif isinstance(react_result, ReactAnswer):
            yield react_result
            break


def react(
    model: str,
    tools: List[ToolCall] = [],
    contexts: List[str | Context] = [],
    max_iter: int = 10,
    iterable: bool = False,
    callback: Callable[[React | ReactAnswer | Observation], None] = None,
    react_kwargs: Any = {},
):
    """
    Decorator to run a function in a loop of question, thought, action.

    Example:

    ```
    @react(model="gpt-4o", tools=[knowledge_query], context="you are a domain knowledge expert")
    async def knowledge_agent(query: str):
        return f"answer the query: {query}"
    ```
    """

    P = ParamSpec("P")
    T = TypeVar("T")

    def wrapper(fn: Callable[P, Awaitable[T]]):
        ctxs = []
        for ctx in contexts:
            if isinstance(ctx, str):
                ctxs.append(Message.system(ctx))
            elif isinstance(ctx, Context):
                ctxs.extend(ctx.all_contexts())
            else:
                raise ValueError(f"Invalid context type: {type(ctx)}")

        _tools = set(tools)
        for ctx in contexts:
            if isinstance(ctx, Context):
                for tool in ctx.all_tools():
                    _tools.add(tool)

        @wraps(fn)
        async def wrapper(
            *args: P.args, **kwargs: P.kwargs
        ) -> Awaitable[React | Observation | ReactAnswer]:
            if inspect.iscoroutinefunction(fn):
                prompt = await fn(*args, **kwargs)
            else:
                prompt = fn(*args, **kwargs)
            chat_history = kwargs.get("chat_history", [])
            if iterable:

                def gen():
                    return run_react(
                        prompt,
                        model=model,
                        contexts=ctxs,
                        tools=list(_tools),
                        max_iter=max_iter,
                        **react_kwargs,
                        chat_history=chat_history,
                    )

                return gen()
            else:
                async for result in run_react(
                    prompt,
                    contexts=ctxs,
                    tools=list(_tools),
                    max_iter=max_iter,
                    chat_history=chat_history,
                ):
                    if callback:
                        if inspect.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                    if isinstance(result, ReactAnswer):
                        return result

        return wrapper

    return wrapper
