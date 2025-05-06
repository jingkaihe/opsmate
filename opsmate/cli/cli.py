from opsmate import __version__
from opsmate.libs.core.trace import traceit, start_trace
from opsmate.libs.config.base_settings import CommaSeparatedList
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich.markdown import Markdown
from opsmate.dino import dino, run_react
from opsmate.dino.types import (
    Observation,
    ReactAnswer,
    React,
    Message,
    ToolCall,
)
from opsmate.dino.provider import Provider
from opsmate.dino.context import ContextRegistry
from functools import wraps
from opsmate.config import config
from opsmate.gui.config import config as gui_config
from opsmate.plugins import PluginRegistry
from opsmate.runtime import Runtime
from opsmate.cli.formatting import OpsmateGroup, OpsmateCommand, get_console

from functools import cache
from typing import Dict
from runpy import run_module
from contextlib import aclosing
import asyncio
import os
import click
import structlog
import sys
import inspect
import opsmate.tools  # noqa: F401
import traceback
import json

console = get_console()


@cache
def addon_discovery():
    PluginRegistry.discover(config.plugins_dir)
    ContextRegistry.discover(config.contexts_dir)


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


logger = structlog.get_logger(__name__)


class StdinArgument(click.ParamType):
    name = "stdin"

    def convert(self, value, param, ctx):
        if value == "-":
            return sys.stdin.read().strip()
        return value


@click.group(cls=OpsmateGroup)
@click.option(
    "--no-color",
    is_flag=True,
    default=True,
    help="Disable colored output, alternatively use OPSMATE_COLOR environment variable [1/true/on/Y/y/yes], or [0/false/no/N/n/off]",
)
@click.pass_context
def opsmate_cli(ctx, no_color):
    """
    Opsmate is an SRE AI assistant that helps you manage production environment.
    This is the cli tool to interact with Opsmate.
    """
    ctx.ensure_object(dict)
    ctx.obj["no_color"] = no_color
    start_trace(spans_to_discard=["dbq.dequeue_task"])


def opsmate_command(**kwargs):
    """Decorator to create an opsmate command with custom formatting."""
    def decorator(f):
        # Set cls=OpsmateCommand for enhanced help formatting
        kwargs.setdefault('cls', OpsmateCommand)
        return opsmate_cli.command(**kwargs)(f)
    return decorator


def config_params(cli_config=config):
    """Decorator to inject pydantic settings config as the click options."""

    def decorator(func):
        # do not use __annotations__ as it does not include the field metadata from the parent class
        config_fields = cli_config.model_fields

        # For each field, create a Click option
        for field_name, field_info in config_fields.items():
            # get the metadata
            field_info = cli_config.model_fields.get(field_name)
            if not field_info:
                continue

            field_type = field_info.annotation
            if field_type in (Dict, list, tuple) or "Dict[" in str(field_type):
                continue

            default_value = getattr(cli_config, field_name)
            is_type_iterable = isinstance(default_value, list) or isinstance(
                default_value, tuple
            )
            if is_type_iterable:
                default_value = ",".join(default_value)
            description = field_info.description or f"Set {field_name}"
            env_var = field_info.alias

            option_names = [f"--{field_name.replace('_', '-')}"]
            if field_info.json_schema_extra and field_info.json_schema_extra.get(
                "abbr"
            ):
                option_names.append(
                    f"-{field_info.json_schema_extra['abbr']}",
                )
            func = click.option(
                *option_names,
                default=default_value,
                help=f"{description} (env: {env_var})",
                show_default=True,
                cls=CommaSeparatedList if is_type_iterable else None,
            )(func)

        def config_from_kwargs(kwargs):
            for field_name in config_fields:
                if field_name in kwargs:
                    setattr(cli_config, field_name, kwargs.pop(field_name))

            cli_config.set_loglevel()
            return cli_config

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs["config"] = config_from_kwargs(kwargs)

            addon_discovery()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def tool_config_params(func):
    for tool_name, tool_config in ToolCall._tool_configs.items():
        func = tool_config.config_params(tool_name)(func)
    return func


def runtime_params(func):
    for runtime_name, runtime_config in Runtime.configs.items():
        func = runtime_config.config_params(runtime_name)(func)
    return func


def runtimes_from_kwargs(kwargs):
    config = kwargs.get("config")
    if config is None:
        raise ValueError("config is required")
    selected_tools = config.opsmate_tools()
    selected_tool_names = [t.__name__ for t in selected_tools]

    tool_configs = kwargs.pop("ToolCallConfig")
    runtime_configs = kwargs.pop("RuntimeConfig")

    runtimes = {}
    configs = {}
    for tool_name, tool_config in tool_configs.items():
        if tool_name not in selected_tool_names:
            continue

        runtime_name = tool_config.runtime
        runtime_class: Runtime = Runtime.runtimes[runtime_name]

        runtime_config = runtime_configs[runtime_name]
        runtimes[tool_name] = runtime_class(runtime_config)
        configs[runtime_name] = runtime_config

    return runtimes, configs, tool_configs


def with_runtime(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        runtimes, _, _ = runtimes_from_kwargs(kwargs)
        tool_call_context = kwargs.get("tool_call_context")
        tool_call_context["runtimes"] = runtimes
        kwargs["runtimes"] = runtimes

        try:
            for runtime in runtimes.values():
                await runtime.connect()
            return await func(*args, **kwargs)
        finally:
            for runtime in runtimes.values():
                await runtime.disconnect()

    return wrapper


def common_params(func):
    @click.option(
        "-r",
        "--review",
        is_flag=True,
        default=False,
        show_default=True,
        help="Review and edit commands before execution",
    )
    @click.option(
        "-s",
        "--system-prompt",
        default=None,
        show_default=True,
        help="System prompt to use",
    )
    @click.option(
        "-l",
        "--max-output-length",
        default=10000,
        show_default=True,
        help="Max length of the output, if the output is truncated, the tmp file will be printed in the output",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        addon_discovery()

        config = kwargs.get("config")
        if config is None:
            raise ValueError("config is required")

        try:
            config.opsmate_context()
        except ValueError as e:
            console.print(
                f"Error: {e}. Run the list-contexts command to see all the contexts available."
            )
            exit(1)

        try:
            config.opsmate_tools()
        except ValueError as e:
            console.print(
                f"Error: {e}. Run the list-tools command to see all the tools available."
            )
            exit(1)

        review = kwargs.pop("review", False)
        kwargs["tool_call_context"] = {
            "max_output_length": kwargs.pop("max_output_length"),
            "in_terminal": True,
        }
        if review:
            kwargs["tool_call_context"]["confirmation"] = confirmation_prompt

        return func(*args, **kwargs)

    return wrapper


def auto_migrate(func):
    @click.option(
        "--auto-migrate",
        default=True,
        show_default=True,
        help="Automatically migrate the database to the latest version",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.pop("auto_migrate", True):
            ctx = click.Context(db_migrate)
            ctx.invoke(db_migrate)
        return func(*args, **kwargs)

    return wrapper


async def confirmation_prompt(tool_call: ToolCall):
    console.print(
        Markdown(
            f"""
## Execution Confirmation

Edit the execution if needed, then press Enter to execute:
!cancel - Cancel the execution
"""
        )
    )
    try:
        for field in tool_call.confirmation_fields():
            prompt = Prompt.ask(
                f"Enter the value for {field}",
                default=tool_call.model_dump()[field],
            )
            setattr(tool_call, field, prompt)
        return True
    except (KeyboardInterrupt, EOFError):
        console.print("\nCommand cancelled")
        return False


@opsmate_command()
@click.argument("instruction", type=StdinArgument())
@click.option(
    "-nt",
    "--no-tool-output",
    is_flag=True,
    help="Do not print tool outputs",
)
@click.option(
    "-no",
    "--no-observation",
    is_flag=True,
    help="Do not print observation",
)
@config_params()
@tool_config_params
@runtime_params
@common_params
@coro
@with_runtime
@traceit(exclude=["system_prompt", "config", "tool_call_context", "tools", "runtimes"])
async def run(
    instruction,
    tool_call_context,
    system_prompt,
    no_tool_output,
    no_observation,
    runtimes,
    config,
    span,
):
    """
    Run a task with the Opsmate.
    """

    ctx = config.opsmate_context()
    tools = config.opsmate_tools()
    run_kwargs = config.models_config.get(config.model, {})

    span.set_attributes(
        {
            "cli.run.instruction": instruction,
            "cli.run.model": config.model,
            "cli.run.model_config": json.dumps(run_kwargs),
            "cli.run.context": config.context,
            "cli.run.tools": [t.__name__ for t in tools],
            "cli.run.system_prompt": system_prompt if system_prompt else "",
            "cli.run.no_tool_output": no_tool_output,
            "cli.run.no_observation": no_observation,
            "cli.run.max_output_length": tool_call_context["max_output_length"],
        }
    )

    logger.info("Running on", instruction=instruction, model=config.model)

    @dino(config.model, response_model=Observation, tools=tools, **run_kwargs)
    async def run_command(instruction: str, context={}):
        sys_prompts = await ctx.resolve_contexts(runtimes=runtimes)
        if system_prompt:
            sys_prompts = [
                Message.system(f"<system_prompt>{system_prompt}</system_prompt>")
            ]
        return [
            *sys_prompts,
            Message.user(instruction),
        ]

    try:
        observation = await run_command(
            instruction,
            context=tool_call_context,
            tool_calls_only=no_observation,
        )
        if no_observation:
            for tool_call in observation.tool_outputs:
                console.print(
                    Markdown(tool_call.display(context={"in_terminal": True}))
                )
            return

        if no_tool_output:
            print(observation.observation)
        else:
            for tool_call in observation.tool_outputs:
                console.print(
                    Markdown(tool_call.display(context={"in_terminal": True}))
                )
                console.print(Markdown(observation.observation))
    except Exception as e:
        console.print(f"Error: {e}")
        console.print(f"Traceback:\n{traceback.format_exc()}")
        exit(1)


@opsmate_command()
@click.argument("instruction", type=StdinArgument())
@click.option(
    "-i",
    "--max-iter",
    default=10,
    show_default=True,
    help="Max number of iterations the AI assistant can reason about",
)
@click.option(
    "-nt",
    "--no-tool-output",
    is_flag=True,
    help="Do not print tool outputs",
)
@click.option(
    "-a",
    "--answer-only",
    is_flag=True,
    help="Print only the answer",
)
@click.option(
    "--tool-calls-per-action",
    default=1,
    show_default=True,
    help="Number of tool calls per action",
)
@config_params()
@tool_config_params
@runtime_params
@common_params
@coro
@with_runtime
@traceit(exclude=["system_prompt", "config", "tool_call_context", "runtimes", "tools"])
async def solve(
    instruction,
    max_iter,
    tool_call_context,
    system_prompt,
    no_tool_output,
    answer_only,
    tool_calls_per_action,
    config,
    runtimes,
    span,
):
    """
    Solve a problem with the Opsmate.
    
    Examples:
      opsmate solve "Why is my CPU usage spiking?"
      opsmate solve --context terraform "What is the status of my EC2 instances?"
    """
    ctx = config.opsmate_context()

    tools = config.opsmate_tools()
    run_react_kwargs = config.models_config.get(config.model, {})

    span.set_attributes(
        {
            "cli.solve.instruction": instruction,
            "cli.solve.model": config.model,
            "cli.solve.model_config": json.dumps(run_react_kwargs),
            "cli.solve.context": config.context,
            "cli.solve.tools": [t.__name__ for t in tools],
            "cli.solve.max_iter": max_iter,
            "cli.solve.no_tool_output": no_tool_output,
            "cli.solve.answer_only": answer_only,
            "cli.solve.tool_calls_per_action": tool_calls_per_action,
        }
    )
    contexts = await ctx.resolve_contexts(runtimes=runtimes)
    if system_prompt:
        contexts = [Message.system(f"<system_prompt>{system_prompt}</system_prompt>")]

    # to deal with https://github.com/open-telemetry/opentelemetry-python/issues/2606
    # src of fix: https://logfire.pydantic.dev/docs/reference/advanced/generators/#use-a-generator-as-a-context-manager
    async with aclosing(
        run_react(
            instruction,
            contexts=contexts,
            model=config.model,
            max_iter=max_iter,
            tools=tools,
            tool_call_context=tool_call_context,
            tool_calls_per_action=tool_calls_per_action,
            **run_react_kwargs,
        )
    ) as run:
        async for output in run:
            match output:
                case React():
                    if answer_only:
                        continue
                    console.print(
                        Markdown(
                            f"""
## Thought process
### Thought

{output.thoughts}

### Action

{output.action}
"""
                        )
                    )
                case Observation():
                    if answer_only:
                        continue
                    console.print(Markdown("## Observation"))
                    if not no_tool_output:
                        for tool_call in output.tool_outputs:
                            console.print(
                                Markdown(
                                    tool_call.display(context={"in_terminal": True})
                                )
                            )
                    console.print(Markdown(output.observation))
                case ReactAnswer():
                    if answer_only:
                        print(output.answer)
                        break
                    console.print(
                        Markdown(
                            f"""
## Answer

{output.answer}
"""
                        )
                    )


help_msg = """
Commands:

!clear - Clear the chat history
!exit - Exit the chat
!help - Show this message
"""


@opsmate_command()
@click.option(
    "-i",
    "--max-iter",
    default=10,
    show_default=True,
    help="Max number of iterations the AI assistant can reason about",
)
@click.option(
    "--tool-calls-per-action",
    default=1,
    show_default=True,
    help="Number of tool calls per action",
)
@config_params()
@tool_config_params
@runtime_params
@common_params
@coro
@with_runtime
@traceit(exclude=["system_prompt", "config", "tool_call_context", "tools", "runtimes"])
async def chat(
    max_iter,
    tool_call_context,
    system_prompt,
    tool_calls_per_action,
    runtimes,
    config,
    span,
):
    """
    Chat with the Opsmate.
    """

    ctx = config.opsmate_context()
    tools = config.opsmate_tools()
    run_react_kwargs = config.models_config.get(config.model, {})
    span.set_attributes(
        {
            "cli.chat.model": config.model,
            "cli.chat.model_config": json.dumps(run_react_kwargs),
            "cli.chat.max_iter": max_iter,
            "cli.chat.context": config.context,
            "cli.chat.tools": [t.__name__ for t in tools],
            "cli.chat.system_prompt": system_prompt if system_prompt else "",
            "cli.chat.tool_calls_per_action": tool_calls_per_action,
        }
    )
    opsmate_says("Howdy! How can I help you?\n" + help_msg)

    try:
        chat_history = []
        while True:
            user_input = console.input("[bold cyan]You> [/bold cyan]")
            if user_input == "!clear":
                chat_history = []
                opsmate_says("Chat history cleared")
                continue
            elif user_input == "!exit":
                break
            elif user_input == "!help":
                console.print(help_msg)
                continue

            contexts = await ctx.resolve_contexts(runtimes=runtimes)
            if system_prompt:
                contexts = [
                    Message.system(f"<system_prompt>{system_prompt}</system_prompt>")
                ]

            run = run_react(
                user_input,
                contexts=contexts,
                model=config.model,
                max_iter=max_iter,
                tools=tools,
                chat_history=chat_history,
                tool_call_context=tool_call_context,
                tool_calls_per_action=tool_calls_per_action,
                **run_react_kwargs,
            )
            chat_history.append(Message.user(user_input))

            async for output in run:
                if isinstance(output, React):
                    tp = f"""
## Thought process
### Thought

{output.thoughts}

### Action

{output.action}
"""
                    console.print(Markdown(tp))
                    chat_history.append(Message.assistant(tp))
                elif isinstance(output, ReactAnswer):
                    tp = f"""
## Answer

{output.answer}
"""
                    console.print(Markdown(tp))
                    chat_history.append(Message.assistant(tp))
                elif isinstance(output, Observation):
                    tp = f"""##Observation
### Tool outputs
"""
                    for tool_call in output.tool_outputs:
                        tp += f"""
{tool_call.display(context={"in_terminal": True})}
"""
                    tp += f"""
### Observation

{output.observation}
"""
                    console.print(Markdown(tp))
                    chat_history.append(Message.assistant(tp))
    except (KeyboardInterrupt, EOFError):
        opsmate_says("Goodbye!")


@opsmate_command()
@config_params()
def list_contexts(config):
    """
    List all the contexts available.
    """
    table = Table(title="Contexts", show_header=True)
    table.add_column("Context")
    table.add_column("Description")

    for ctx in ContextRegistry.get_contexts().values():
        table.add_row(ctx.name, ctx.description)

    console.print(table)


@opsmate_command()
@config_params()
@click.option("--skip-confirm", is_flag=True, help="Skip confirmation")
@coro
async def reset(skip_confirm, config):
    """
    Reset the Opsmate database and embeddings db.
    Note that if the database is using litestream it will not be reset.
    Same applies to the embeddings db, if the embedding db is using GCS, S3 or Azure Blob Storage, it will not be reset.
    """
    import glob
    import shutil

    def remove_db_url(db_url):
        if db_url == ":memory:":
            return

        # Remove the main db and all related files (journal, wal, shm, etc)
        for f in glob.glob(f"{db_url}*"):
            if os.path.exists(f):
                if os.path.isdir(f):
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    os.remove(f)

    def remove_embeddings_db_path(embeddings_db_path):
        if (
            embeddings_db_path.startswith("gs://")
            or embeddings_db_path.startswith("az://")
            or embeddings_db_path.startswith("s3://")
        ):
            logger.info(
                "Skipping deletion of embeddings db path",
                embeddings_db_path=embeddings_db_path,
            )
            return
        shutil.rmtree(embeddings_db_path, ignore_errors=True)

    db_url = config.db_url
    db_url = db_url.replace("sqlite:///", "")

    if skip_confirm:
        console.print("Resetting Opsmate")
        remove_db_url(db_url)
        remove_embeddings_db_path(config.embeddings_db_path)
        return

    if (
        Prompt.ask(
            f"""Are you sure you want to reset Opsmate? This will delete:
- {db_url}
- {config.embeddings_db_path}
""",
            default="no",
            choices=["yes", "no"],
        )
        == "no"
    ):
        console.print("Reset cancelled")
        return

    remove_db_url(db_url)
    remove_embeddings_db_path(config.embeddings_db_path)


@opsmate_command()
@click.option(
    "-h", "--host", default="0.0.0.0", show_default=True, help="Host to serve on"
)
@click.option("-p", "--port", default=8080, show_default=True, help="Port to serve on")
@click.option(
    "-w",
    "--workers",
    default=2,
    show_default=True,
    help="Number of uvicorn workers to serve on",
)
@click.option(
    "--dev",
    is_flag=True,
    help="Run in development mode",
)
@config_params(gui_config)
@tool_config_params
@runtime_params
@auto_migrate
@coro
async def serve(host, port, workers, dev, **kwargs):
    """
    Start the Opsmate server.
    """

    # serialize config to environment variables
    config = kwargs.get("config")
    config.serialize_to_env()

    kwargs["context"] = config.context
    kwargs["tools"] = config.opsmate_tools()
    _, runtime_configs, tool_configs = runtimes_from_kwargs(kwargs)
    for _, runtime_config in runtime_configs.items():
        runtime_config.serialize_to_env()

    for _, tool_config in tool_configs.items():
        tool_config.serialize_to_env()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import uvicorn
    from sqlmodel import Session
    from opsmate.gui.app import kb_ingest
    from opsmate.gui.seed import seed_blueprints

    await kb_ingest()
    engine = config.db_engine()

    with Session(engine) as session:
        seed_blueprints(session)

    if dev:
        uvicorn.run(
            "opsmate.apiserver.apiserver:app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=["opsmate"],
        )
        return
    if workers > 1:
        uvicorn.run(
            "opsmate.apiserver.apiserver:app",
            host=host,
            port=port,
            workers=workers,
        )
    else:
        uvicorn_config = uvicorn.Config(
            "opsmate.apiserver.apiserver:app",
            host=host,
            port=port,
        )
        server = uvicorn.Server(uvicorn_config)
        await server.serve()


@opsmate_command()
@click.option(
    "-w",
    "--workers",
    default=10,
    show_default=True,
    help="Number of concurrent background workers",
)
@click.option(
    "-q",
    "--queue",
    default="default",
    show_default=True,
    help="Queue to use for the worker",
)
@config_params()
@auto_migrate
@coro
async def worker(workers, queue, config):
    """
    Start the Opsmate worker.
    """
    from opsmate.dbqapp import app as dbqapp
    from opsmate.knowledgestore.models import init_table

    try:
        await init_table()
        task = asyncio.create_task(dbqapp.main(workers, queue))
        await task
    except KeyboardInterrupt:
        task.cancel()
        await task


@opsmate_command()
@click.option(
    "-i",
    "--interval-seconds",
    default=30,
    show_default=True,
    help="Interval seconds to run the reindex task",
)
@click.option(
    "-nw",
    "--no-wait-for-completion",
    is_flag=True,
    help="Do not wait for the reindex task to complete before scheduling the next one",
)
@config_params()
@auto_migrate
@coro
async def schedule_embeddings_reindex(config, interval_seconds, no_wait_for_completion):
    """
    Schedule the reindex embeddings table task.
    It will purge all the reindex tasks before scheduling the new one.
    After schedule the reindex task will be run periodically every 30 seconds.
    """
    from opsmate.knowledgestore.models import schedule_reindex_table
    from opsmate.dbq.dbq import purge_tasks
    from sqlmodel import Session

    reindex_task_name = "opsmate.knowledgestore.models.reindex_table"
    engine = config.db_engine()
    with Session(engine) as session:
        while True:
            purged, running = purge_tasks(
                session, task_name=reindex_task_name, non_running=True
            )
            if running and not no_wait_for_completion:
                console.print(
                    f"""{purged} reindex tasks purged, {running} reindex tasks running.
Waiting for them to complete before scheduling the next one...
"""
                )
                await asyncio.sleep(5)
            else:
                break

        await schedule_reindex_table(session, interval_seconds)


@opsmate_command()
@click.option(
    "--prometheus-endpoint",
    default=lambda: os.getenv("PROMETHEUS_ENDPOINT", "") or "http://localhost:9090",
    show_default=True,
    help="Prometheus endpoint. If not provided it uses $PROMETHEUS_ENDPOINT environment variable, or defaults to http://localhost:9090",
)
@click.option(
    "--prometheus-user-id",
    # prompt=True,
    default=lambda: os.getenv("PROMETHEUS_USER_ID", ""),
    show_default=True,
    help="Prometheus user id. If not provided it uses $PROMETHEUS_USER_ID environment variable, or defaults to empty string",
)
@click.option(
    "--prometheus-api-key",
    # prompt=True,
    default=lambda: os.getenv("PROMETHEUS_API_KEY", ""),
    show_default=True,
    hide_input=True,
    help="Prometheus api key. If not provided it uses $PROMETHEUS_API_KEY environment variable, or defaults to empty string",
)
@config_params()
@auto_migrate
@coro
async def ingest_prometheus_metrics_metadata(
    prometheus_endpoint, prometheus_user_id, prometheus_api_key, config
):
    """
    Ingest prometheus metrics metadata into the knowledge base.
    The ingestion is done via fetching the metrics metadata from the prometheus server, and then storing it into the knowledge base.
    The ingested metrics metadata will be used for providing context to the LLM when querying prometheus based metrics
    Note this only enqueues the tasks to ingest metrics. To execute the actual ingestion in the background, run `opsmate worker`.
    Please run: `opsmate worker -w 1 -q lancedb-batch-ingest`
    """
    from opsmate.tools.prom import PromQL
    from opsmate.knowledgestore.models import init_table
    from sqlmodel import Session

    await init_table()

    prom = PromQL(
        endpoint=prometheus_endpoint,
        user_id=prometheus_user_id,
        api_key=prometheus_api_key,
    )

    engine = config.db_engine()
    with Session(engine) as session:
        await prom.ingest_metrics(session)


@opsmate_command()
@config_params()
@traceit(exclude=["config"])
def list_tools(config):
    """
    List all the tools available.
    """
    table = Table(title="Tools", show_header=True, show_lines=True)
    table.add_column("Tool")
    table.add_column("Description")

    for tool_name, tool in PluginRegistry.get_tools().items():
        table.add_row(tool_name, inspect.getdoc(tool))

    console.print(table)


@opsmate_command()
@click.option(
    "--provider",
    help="Provider to list the models for",
)
def list_models(provider):
    """
    List all the models available.
    """
    table = Table(title="Models", show_header=True, show_lines=True)
    table.add_column("Provider")
    table.add_column("Model")

    if provider:
        if provider not in Provider.providers:
            console.print(f"Provider {provider} not found")
            exit(1)
        provider_models = Provider.providers[provider]
        for model in provider_models.models:
            table.add_row(provider, model)
    else:
        for provider_name, provider in Provider.providers.items():
            for model in provider.models:
                table.add_row(provider_name, model)

    console.print(table)


@opsmate_command()
@click.option(
    "--source",
    help="Source of the knowledge base fs:////path/to/kb or github:///owner/repo[:branch]",
)
@click.option(
    "--path",
    default="",
    show_default=True,
    help="Path to the knowledge base",
)
@click.option(
    "--glob",
    default="**/*.md",
    show_default=True,
    help="Glob to use to find the knowledge base",
)
@config_params()
@auto_migrate
@coro
async def ingest(source, path, glob, config):
    """
    Ingest a knowledge base.
    Notes the ingestion worker needs to be started separately with `opsmate worker`.
    """

    from sqlmodel import Session
    from opsmate.dbq.dbq import enqueue_task
    from opsmate.ingestions.jobs import ingest
    from opsmate.knowledgestore.models import init_table

    await init_table()

    engine = config.db_engine()

    splitted = source.split(":///")
    if len(splitted) != 2:
        console.print(
            "Invalid source. Use the format fs:///path/to/kb or github:///owner/repo[:branch]"
        )
        exit(1)

    provider, source = splitted

    if ":" in source:
        source, branch = source.split(":")
    else:
        branch = "main"

    splitter_config = config.splitter_config

    with Session(engine) as session:
        match provider:
            case "fs":
                enqueue_task(
                    session,
                    ingest,
                    ingestor_type="fs",
                    ingestor_config={"local_path": source, "glob_pattern": glob},
                    splitter_config=splitter_config,
                )
            case "github":
                enqueue_task(
                    session,
                    ingest,
                    ingestor_type="github",
                    ingestor_config={
                        "repo": source,
                        "branch": branch,
                        "path": path,
                        "glob": glob,
                    },
                    splitter_config=splitter_config,
                )
    console.print("Ingesting knowledges in the background...")


alembic_cfg_path = os.path.join(
    os.path.dirname(__file__), "..", "migrations", "alembic.ini"
)


@opsmate_command()
@coro
async def list_runtimes():
    """
    List all the runtimes available.
    """
    table = Table(title="Runtimes", show_header=True, show_lines=True)
    table.add_column("Name")
    table.add_column("Description")

    for name, runtime in Runtime.runtimes.items():
        table.add_row(name, inspect.getdoc(runtime))

    console.print(table)


@opsmate_command()
@click.option(
    "-r",
    "--revision",
    default="head",
    show_default=True,
    help="Revision to upgrade to",
)
def db_migrate(revision):
    """Apply migrations."""
    from alembic import command
    from alembic.config import Config as AlembicConfig

    alembic_cfg = AlembicConfig(alembic_cfg_path)
    command.upgrade(alembic_cfg, revision)
    click.echo(f"Database upgraded to: {revision}")


@opsmate_command()
@click.option(
    "-r",
    "--revision",
    default="-1",
    show_default=True,
    help="Revision to downgrade to",
)
def db_rollback(revision):
    """Rollback migrations."""
    from alembic import command
    from alembic.config import Config as AlembicConfig

    alembic_cfg = AlembicConfig(alembic_cfg_path)
    command.downgrade(alembic_cfg, revision)
    click.echo(f"Database downgraded to: {revision}")


@opsmate_command()
def db_revisions():
    """
    List all the revisions available.
    """
    from alembic import command
    from alembic.config import Config as AlembicConfig

    alembic_cfg = AlembicConfig(alembic_cfg_path)
    command.history(alembic_cfg)


@opsmate_command()
@click.argument("packages", nargs=-1, required=False)
@click.option(
    "-U",
    "--upgrade",
    is_flag=True,
    help="Upgrade the given packages to the latest version",
)
@click.option(
    "--force-reinstall",
    is_flag=True,
    help="Reinstall all packages even if they are already up-to-date",
)
@click.option(
    "-e",
    "--editable",
    help="""Install a project in editable mode (i.e. setuptools "develop mode") from a local project path or a VCS url""",
)
@click.option(
    "--no-cache-dir",
    is_flag=True,
    help="Disable the cache",
)
def install(packages, upgrade, force_reinstall, editable, no_cache_dir):
    """
    Install the opsmate plugins.
    """
    args = ["pip", "install"]
    if upgrade:
        args.append("--upgrade")
    if force_reinstall:
        args.append("--force-reinstall")
    if editable:
        args.extend(["--editable", editable])
    if no_cache_dir:
        args.append("--no-cache-dir")
    args.extend(packages)

    sys.argv = args
    run_module("pip", run_name="__main__")


@opsmate_command()
@click.argument("packages", nargs=-1, required=True)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Do not prompt for confirmation",
)
def uninstall(packages, yes):
    """
    Uninstall the given packages.
    """
    args = ["pip", "uninstall"]
    if yes:
        args.append("--yes")
    args.extend(packages)

    sys.argv = args
    run_module("pip", run_name="__main__")


@opsmate_command()
def version():
    """
    Show the version of the Opsmate.
    """
    console.print(__version__)


def opsmate_says(message: str):
    text = Text()
    text.append("Opsmate> ", style="bold green")
    text.append(message)
    console.print(text)


if __name__ == "__main__":
    opsmate_cli()
