from opsmate import __version__
from opsmate.libs.core.trace import traceit
from openai_otel import OpenAIAutoInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich.markdown import Markdown
from opsmate.dino import dino, run_react
from opsmate.dino.types import Observation, ReactAnswer, React, Message
from opsmate.tools.command_line import ShellCommand
from opsmate.dino.provider import Provider
from opsmate.dino.context import ContextRegistry
from functools import wraps
from opsmate.libs.config import config
from opsmate.plugins import PluginRegistry
import asyncio
import os
import click
import structlog
import sys


def addon_discovery():
    PluginRegistry.discover(config.plugins_dir)
    ContextRegistry.discover(config.contexts_dir)


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


console = Console()
resource = Resource(attributes={SERVICE_NAME: os.getenv("SERVICE_NAME", "opamate")})

otel_enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"

if otel_enabled:
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        insecure=True,
    )
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    OpenAIAutoInstrumentor().instrument()

logger = structlog.get_logger(__name__)


class StdinArgument(click.ParamType):
    name = "stdin"

    def convert(self, value, param, ctx):
        if value == "-":
            return sys.stdin.read().strip()
        return value


@click.group()
def opsmate_cli():
    """
    OpsMate is an SRE AI assistant that helps you manage production environment.
    This is the cli tool to interact with OpsMate.
    """
    pass


def common_params(func):
    @click.option(
        "--tools",
        default="",
        help="Comma separated list of tools to use",
    )
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
        _tool_names = kwargs.pop("tools")
        _tool_names = _tool_names.split(",")
        _tool_names = [t for t in _tool_names if t != ""]
        try:
            tools = PluginRegistry.get_tools_from_list(_tool_names)
        except ValueError as e:
            console.print(
                f"Tool {e} not found. Run the list-tools command to see all the tools available."
            )
            exit(1)

        kwargs["tools"] = tools

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


async def confirmation_prompt(tool_call: ShellCommand):
    console.print(
        Markdown(
            f"""
## Command Confirmation

Edit the command if needed, then press Enter to execute:
!cancel - Cancel the command
"""
        )
    )
    try:
        prompt = Prompt.ask(
            "Press Enter or edit the command",
            default=tool_call.command,
        )
        tool_call.command = prompt
        if prompt == "!cancel":
            return False
        return True
    except (KeyboardInterrupt, EOFError):
        console.print("\nCommand cancelled")
        return False


@opsmate_cli.command()
@click.argument("instruction", type=StdinArgument())
@click.option(
    "-m",
    "--model",
    show_default=True,
    default="gpt-4o",
    help="OpenAI model to use. To list models available please run the list-models command.",
)
@click.option(
    "-c",
    "--context",
    show_default=True,
    default="cli",
    help="Context to be added to the prompt. Run the list-contexts command to see all the contexts available.",
)
@click.option(
    "-n",
    "--no-tool-output",
    is_flag=True,
    help="Do not print tool outputs",
)
@common_params
@traceit
@coro
async def run(
    instruction, model, context, tools, tool_call_context, system_prompt, no_tool_output
):
    """
    Run a task with the OpsMate.
    """

    ctx = get_context(context)

    if len(tools) == 0:
        tools = ctx.resolve_tools()

    logger.info("Running on", instruction=instruction, model=model)

    @dino(model, response_model=Observation, tools=tools)
    async def run_command(instruction: str, context={}):
        sys_prompts = await ctx.resolve_contexts()
        if system_prompt:
            sys_prompts = [
                Message.system(f"<system_prompt>{system_prompt}</system_prompt>")
            ]
        return [
            *sys_prompts,
            Message.user(instruction),
        ]

    observation = await run_command(instruction, context=tool_call_context)

    if not no_tool_output:
        for tool_call in observation.tool_outputs:
            console.print(Markdown(tool_call.markdown(context={"in_terminal": True})))
        console.print(Markdown(observation.observation))
    else:
        print(observation.observation)


@opsmate_cli.command()
@click.argument("instruction", type=StdinArgument())
@click.option(
    "-m",
    "--model",
    default="gpt-4o",
    show_default=True,
    help="OpenAI model to use. To list models available please run the list-models command.",
)
@click.option(
    "-i",
    "--max-iter",
    default=10,
    show_default=True,
    help="Max number of iterations the AI assistant can reason about",
)
@click.option(
    "-c",
    "--context",
    default="cli",
    show_default=True,
    help="Context to be added to the prompt. Run the list-contexts command to see all the contexts available.",
)
@click.option(
    "-n",
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
@common_params
@traceit
@coro
async def solve(
    instruction,
    model,
    max_iter,
    context,
    tools,
    tool_call_context,
    system_prompt,
    no_tool_output,
    answer_only,
    tool_calls_per_action,
):
    """
    Solve a problem with the OpsMate.
    """
    ctx = get_context(context)

    if len(tools) == 0:
        tools = ctx.resolve_tools()

    contexts = await ctx.resolve_contexts()

    if system_prompt:
        contexts = [Message.system(f"<system_prompt>{system_prompt}</system_prompt>")]

    async for output in run_react(
        instruction,
        contexts=contexts,
        model=model,
        max_iter=max_iter,
        tools=tools,
        tool_call_context=tool_call_context,
        tool_calls_per_action=tool_calls_per_action,
    ):
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
                            Markdown(tool_call.markdown(context={"in_terminal": True}))
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


@opsmate_cli.command()
@click.option(
    "-m",
    "--model",
    default="gpt-4o",
    show_default=True,
    help="OpenAI model to use. To list models available please run the list-models command.",
)
@click.option(
    "-i",
    "--max-iter",
    default=10,
    show_default=True,
    help="Max number of iterations the AI assistant can reason about",
)
@click.option(
    "-c",
    "--context",
    default="cli",
    show_default=True,
    help="Context to be added to the prompt. Run the list-contexts command to see all the contexts available.",
)
@click.option(
    "--tool-calls-per-action",
    default=1,
    show_default=True,
    help="Number of tool calls per action",
)
@common_params
@traceit
@coro
async def chat(
    model,
    max_iter,
    context,
    tools,
    tool_call_context,
    system_prompt,
    tool_calls_per_action,
):
    """
    Chat with the OpsMate.
    """

    ctx = get_context(context)

    if len(tools) == 0:
        tools = ctx.tools

    opsmate_says("Howdy! How can I help you?\n" + help_msg)

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

        contexts = await ctx.resolve_contexts()
        if system_prompt:
            contexts = [
                Message.system(f"<system_prompt>{system_prompt}</system_prompt>")
            ]

        run = run_react(
            user_input,
            contexts=contexts,
            model=model,
            max_iter=max_iter,
            tools=tools,
            chat_history=chat_history,
            tool_call_context=tool_call_context,
            tool_calls_per_action=tool_calls_per_action,
        )
        chat_history.append(Message.user(user_input))

        try:
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
    {tool_call.markdown(context={"in_terminal": True})}
    """
                    tp += f"""
### Observation

{output.observation}
"""
                    console.print(Markdown(tp))
                    chat_history.append(Message.assistant(tp))
        except (KeyboardInterrupt, EOFError):
            opsmate_says("Goodbye!")


@opsmate_cli.command()
def list_contexts():
    """
    List all the contexts available.
    """
    addon_discovery()
    table = Table(title="Contexts", show_header=True)
    table.add_column("Context")
    table.add_column("Description")

    for ctx in ContextRegistry.get_contexts().values():
        table.add_row(ctx.name, ctx.description)

    console.print(table)


@opsmate_cli.command()
@click.option("--skip-confirm", is_flag=True, help="Skip confirmation")
@coro
async def reset(skip_confirm):
    """
    Reset the OpsMate.
    """
    from opsmate.libs.config import config
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
        shutil.rmtree(embeddings_db_path, ignore_errors=True)

    db_url = config.db_url
    db_url = db_url.replace("sqlite:///", "")

    if skip_confirm:
        console.print("Resetting OpsMate")
        remove_db_url(db_url)
        remove_embeddings_db_path(config.embeddings_db_path)
        return

    if (
        Prompt.ask(
            f"""Are you sure you want to reset OpsMate? This will delete:
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


@opsmate_cli.command()
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
@auto_migrate
@coro
async def serve(host, port, workers, dev):
    """
    Start the OpsMate server.
    """
    import uvicorn
    from sqlmodel import create_engine, text, Session
    from opsmate.gui.app import kb_ingest
    from opsmate.gui.seed import seed_blueprints
    from opsmate.libs.config import config

    await kb_ingest()
    engine = create_engine(
        config.db_url,
        connect_args={"check_same_thread": False, "timeout": 20},
        # echo=True,
    )
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.close()

    with Session(engine) as session:
        seed_blueprints(session)

    if dev:
        await uvicorn.run(
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
        config = uvicorn.Config(
            "opsmate.apiserver.apiserver:app",
            host=host,
            port=port,
        )
        server = uvicorn.Server(config)
        await server.serve()


@opsmate_cli.command()
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
@auto_migrate
@coro
async def worker(workers, queue):
    """
    Start the OpsMate worker.
    """
    from opsmate.dbqapp import app as dbqapp
    from opsmate.knowledgestore.models import init_table

    addon_discovery()

    try:
        await init_table()
        task = asyncio.create_task(dbqapp.main(workers, queue))
        await task
    except KeyboardInterrupt:
        task.cancel()
        await task


@opsmate_cli.command()
@click.option(
    "--prometheus-endpoint",
    default=lambda: os.getenv("prometheus_endpoint", "http://localhost:9090"),
    show_default=True,
    help="Prometheus endpoint",
)
@click.option(
    "--prometheus-user-id",
    # prompt=True,
    default=lambda: os.getenv("prometheus_user_id", ""),
    show_default=True,
    help="Prometheus user id",
)
@click.option(
    "--prometheus-api-key",
    # prompt=True,
    default=lambda: os.getenv("prometheus_api_key", ""),
    show_default=True,
    hide_input=True,
    help="Prometheus api key",
)
@coro
async def ingest_metrics(prometheus_endpoint, prometheus_user_id, prometheus_api_key):
    """
    Ingest metrics metadata into the knowledge base.
    Note this only enqueues the taks to ingest metrics. To execute the actual ingestion in the background, run `opsmate worker`.
    Please run: `opsmate worker -w 1 -q lancedb-batch-ingest`
    """
    from opsmate.tools.prom import PromQL
    from opsmate.knowledgestore.models import init_table, aconn
    from sqlmodel import create_engine, text, Session
    from opsmate.libs.config import config

    await init_table()
    dbconn = await aconn()
    table = await dbconn.open_table("knowledge_store")

    prom = PromQL(
        endpoint=prometheus_endpoint,
        user_id=prometheus_user_id,
        api_key=prometheus_api_key,
    )

    engine = create_engine(
        config.db_url,
        connect_args={"check_same_thread": False, "timeout": 20},
        # echo=True,
    )
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.close()
    with Session(engine) as session:
        await prom.ingest_metrics(session)


@opsmate_cli.command()
def list_tools():
    """
    List all the tools available.
    """
    addon_discovery()
    table = Table(title="Tools", show_header=True, show_lines=True)
    table.add_column("Tool")
    table.add_column("Description")

    for tool_name, tool in PluginRegistry.get_tools().items():
        table.add_row(tool_name, tool.__doc__)

    console.print(table)


@opsmate_cli.command()
def list_models():
    """
    List all the models available.
    """
    table = Table(title="Models", show_header=True, show_lines=True)
    table.add_column("Provider")
    table.add_column("Model")

    for provider_name, provider in Provider.providers.items():
        for model in provider.models:
            table.add_row(provider_name, model)

    console.print(table)


@opsmate_cli.command()
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
@auto_migrate
@coro
async def ingest(source, path, glob):
    """
    Ingest a knowledge base.
    Notes the ingestion worker needs to be started separately with `opsmate worker`.
    """

    from opsmate.libs.config import config
    from sqlmodel import create_engine, text, Session
    from opsmate.dbq.dbq import enqueue_task
    from opsmate.ingestions.jobs import ingest
    from opsmate.knowledgestore.models import init_table

    addon_discovery()
    await init_table()

    engine = create_engine(
        config.db_url,
        connect_args={"check_same_thread": False, "timeout": 20},
        # echo=True,
    )
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.close()

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


@opsmate_cli.command()
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

    alembic_cfg = AlembicConfig("opsmate/migrations/alembic.ini")
    command.upgrade(alembic_cfg, revision)
    click.echo(f"Database upgraded to: {revision}")


@opsmate_cli.command()
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

    alembic_cfg = AlembicConfig("opsmate/migrations/alembic.ini")
    command.downgrade(alembic_cfg, revision)
    click.echo(f"Database downgraded to: {revision}")


@opsmate_cli.command()
def db_revisions():
    """
    List all the revisions available.
    """
    from alembic import command
    from alembic.config import Config as AlembicConfig

    alembic_cfg = AlembicConfig("opsmate/migrations/alembic.ini")
    command.history(alembic_cfg)


@opsmate_cli.command()
def version():
    """
    Show the version of the OpsMate.
    """
    console.print(__version__)


def get_context(ctx_name: str):
    ctx = ContextRegistry.get_context(ctx_name)
    if not ctx:
        console.print(
            f"Context {ctx_name} not found. Run the list-contexts command to see all the contexts available."
        )
        exit(1)
    return ctx


def opsmate_says(message: str):
    text = Text()
    text.append("OpsMate> ", style="bold green")
    text.append(message)
    console.print(text)


if __name__ == "__main__":
    opsmate_cli()
