from fasthtml.common import *
from opsmate.gui.assets import *
from opsmate.gui.models import (
    Cell,
    CellLangEnum,
    get_active_stage,
    executor,
    k8s_agent,
)
from opsmate.libs.core.types import (
    ExecResults,
    Observation,
    ReactProcess,
    ReactAnswer,
)
from opsmate.libs.core.engine.agent_executor import AgentCommand
import pickle
import sqlmodel
import structlog
import asyncio
import subprocess

logger = structlog.get_logger()

# Set up the app, including daisyui and tailwind for the chat component
tlink = (Script(src="https://cdn.tailwindcss.com"),)
nav = (
    Nav(
        Div(A("Opsmate Workspace", cls="btn btn-ghost text-xl"), cls="flex-1"),
        Div(
            Label(
                Input(
                    type="checkbox",
                    value="synthwave",
                    cls="theme-controller",
                    hidden=true,
                ),
                sun_icon_svg,
                moon_icon_svg,
                cls="swap swap-rotate",
            ),
        ),
        cls="navbar bg-base-100 shadow-lg mb-4 fixed top-0 left-0 right-0 z-50",
    ),
)

dlink = Link(
    rel="stylesheet",
    href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
)


def output_cell(cell: Cell):
    if cell.output:
        outputs = pickle.loads(cell.output)
        outputs = [
            cell_render_funcs[output["type"]](k8s_agent.metadata.name, output["output"])
            for output in outputs
        ]
    else:
        outputs = []
    return Div(
        Span(f"Out [{cell.execution_sequence}]:", cls="text-gray-500 text-sm"),
        Div(
            *outputs,
            id=f"cell-output-{cell.id}",
        ),
        cls="px-4 py-2 bg-gray-50 border-t rounded-b-lg overflow-hidden",
    )


def cell_component(cell: Cell, cell_size: int):
    """Renders a single cell component"""
    # Determine if the cell is active
    active_class = "border-green-500 bg-white" if cell.active else "border-gray-300"

    return Div(
        # Add Cell Button Menu
        Div(
            Div(
                Button(
                    plus_icon_svg,
                    tabindex="0",
                    cls="btn btn-ghost btn-xs",
                ),
                Ul(
                    Li(
                        Button(
                            "Insert Above",
                            hx_post=f"/cell/{cell.id}?above=true",
                        )
                    ),
                    Li(
                        Button(
                            "Insert Below",
                            hx_post=f"/cell/{cell.id}?above=false",
                        )
                    ),
                    tabindex="0",
                    cls="dropdown-content z-10 menu p-2 shadow bg-base-100 rounded-box",
                ),
                cls="dropdown dropdown-right",
            ),
            cls="absolute -left-8 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity",
        ),
        # Main Cell Content
        Div(
            # Cell Header
            Div(
                Div(
                    Span(
                        f"In [{cell.execution_sequence}]:", cls="text-gray-500 text-sm"
                    ),
                    # Add cell type selector
                    cls="flex items-center gap-2",
                ),
                Div(
                    Select(
                        Option(
                            "Text Instruction",
                            value=CellLangEnum.TEXT_INSTRUCTION.value,
                            selected=cell.lang == CellLangEnum.TEXT_INSTRUCTION,
                        ),
                        Option(
                            "Bash",
                            value=CellLangEnum.BASH.value,
                            selected=cell.lang == CellLangEnum.BASH,
                        ),
                        name="lang",
                        hx_put=f"/cell/{cell.id}",
                        hx_trigger="change",
                        cls="select select-sm ml-2",
                    ),
                    Button(
                        trash_icon_svg,
                        hx_delete=f"/cell/{cell.id}",
                        cls="btn btn-ghost btn-sm opacity-0 group-hover:opacity-100 hover:text-red-500",
                        disabled=cell_size == 1,
                    ),
                    Form(
                        Input(type="hidden", value=cell.id, name="cell_id"),
                        Button(
                            run_icon_svg,
                            cls="btn btn-ghost btn-sm",
                        ),
                        ws_connect=f"/cell/run/ws/",
                        ws_send=True,
                        hx_ext="ws",
                    ),
                    cls="ml-auto flex items-center gap-2",
                ),
                id=f"cell-header-{cell.id}",
                cls="flex items-center px-4 py-2 bg-gray-100 border-b justify-between rounded-t-lg overflow-hidden",
            ),
            # Cell Input - Updated with conditional styling
            Div(
                Form(
                    Textarea(
                        cell.input,
                        name="input",
                        cls=f"w-full h-24 p-2 font-mono text-sm border rounded focus:outline-none focus:border-blue-500",
                        placeholder="Enter your instruction here...",
                        id=f"cell-input-{cell.id}",
                    ),
                    Div(
                        hx_put=f"/cell/input/{cell.id}",
                        hx_trigger=f"keyup[!(shiftKey&&keyCode===13)] changed delay:500ms from:#cell-input-{cell.id}",
                        hx_swap=f"#cell-input-form-{cell.id}",
                    ),
                    # xxx: shift+enter is being registered as a newline
                    Div(
                        Input(type="hidden", value=cell.id, name="cell_id"),
                        ws_connect=f"/cell/run/ws/",
                        ws_send=True,
                        hx_ext="ws",
                        hx_trigger=f"keydown[shiftKey&&keyCode===13] from:#cell-input-{cell.id}",
                        hx_swap=f"#cell-input-form-{cell.id}",
                    ),
                    id=f"cell-input-form-{cell.id}",
                ),
                hx_include="input",
                cls="p-4",
            ),
            # Cell Output (if any)
            output_cell(cell),
            cls=f"rounded-lg shadow-sm border {active_class}",  # Apply the active class here
        ),
        cls="group relative",
        key=cell.id,
        id=f"cell-component-{cell.id}",
    )


add_cell_button = (
    Div(
        Button(
            add_cell_svg,
            "Add Cell",
            hx_post="/cell/bottom",
            cls="btn btn-primary btn-sm flex items-center gap-2",
        ),
        id="add-cell-button",
        hx_swap_oob="true",
        cls="flex justify-end",
    ),
)

reset_button = (
    Div(
        Button(
            "Reset",
            cls="btn btn-secondary btn-sm flex items-center gap-1",
        ),
        hx_post="/cells/reset",
        hx_swap_oob="true",
        id="reset-button",
        cls="flex",
    ),
)


def stage_button(stage: dict):
    cls = "px-6 py-3 text-sm font-medium border-0"
    if stage["active"]:
        cls += " bg-white border-b-2 border-b-blue-500 text-blue-600"
    else:
        cls += " bg-gray-50 text-gray-600 hover:bg-gray-100"
    return Button(
        stage["title"],
        hx_put=f"/stage/{stage['id']}/switch",
        cls=cls,
    )


async def execute_llm_instruction(
    cell: Cell, swap: str, send, session: sqlmodel.Session
):
    logger.info("executing llm instruction", cell_id=cell.id)

    outputs = []
    await send(
        Div(
            *outputs,
            hx_swap_oob="true",
            id=f"cell-output-{cell.id}",
        )
    )
    msg = cell.input.rstrip()
    # execution = executor.supervise(supervisor, msg)
    execution = executor.execute(k8s_agent, msg)

    async for stage in async_wrapper(execution):
        actor = k8s_agent.metadata.name
        output = stage
        partial = None
        if isinstance(output, ExecResults):
            partial = render_exec_results_markdown(actor, output)
        elif isinstance(output, AgentCommand):
            partial = render_agent_command_markdown(actor, output)
        elif isinstance(output, ReactProcess):
            partial = render_react_markdown(actor, output)
        elif isinstance(output, ReactAnswer):
            partial = render_react_answer_markdown(actor, output)
        # elif isinstance(output, Observation):
        #     partial = render_observation_markdown(actor, output)
        if partial:
            outputs.append(
                {
                    "type": type(output).__name__,
                    "output": output,
                }
            )
            await send(
                Div(
                    partial,
                    hx_swap_oob=swap,
                    id=f"cell-output-{cell.id}",
                )
            )

    cell.output = pickle.dumps(outputs)
    session.add(cell)
    session.commit()


async def execute_bash_instruction(
    cell: Cell, swap: str, send, session: sqlmodel.Session
):
    logger.info("executing bash instruction", cell_id=cell.id)
    outputs = []
    await send(
        Div(
            *outputs,
            hx_swap_oob="true",
            id=f"cell-output-{cell.id}",
        )
    )

    script = cell.input.rstrip()
    # execute the script using subprocess with combined output
    process = subprocess.Popen(
        script,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
    )

    combined_output = ""
    while True:
        output = process.stdout.readline()
        error = process.stderr.readline()

        if output == "" and error == "" and process.poll() is not None:
            break

        if output:
            combined_output += output
        if error:
            combined_output += error

    partial = render_bash_output_markdown(k8s_agent.metadata.name, combined_output)
    outputs.append(
        {
            "type": "BashOutput",
            "output": combined_output,
        }
    )
    cell.output = pickle.dumps(outputs)
    session.add(cell)
    session.commit()
    await send(
        Div(
            partial,
            hx_swap_oob=swap,
            id=f"cell-output-{cell.id}",
        )
    )


async def async_wrapper(generator: Generator):
    for stage in generator:
        await asyncio.sleep(0)
        yield stage


def render_cell_container(cells: list[Cell], hx_swap_oob: str = None):
    div = Div(
        *[cell_component(cell, len(cells)) for cell in cells],
        cls="space-y-4 mt-4",
        id="cells-container",
    )
    if hx_swap_oob:
        div.hx_swap_oob = hx_swap_oob
    return div


def render_stage_panel(stages: list[dict]):
    active_stage = get_active_stage()
    return Div(
        Div(
            *[stage_button(stage) for stage in stages],
            cls="flex border-t",
        ),
        # stage Panels
        Div(
            Div(
                Div(
                    Span(
                        f"Current Phase: {active_stage['title']}",
                        cls="font-medium",
                    ),
                    cls="flex items-center gap-2 text-sm text-gray-500",
                ),
                cls="space-y-6",
            ),
            cls="block p-4",
        ),
        # stage description
        Div(
            Div(
                Div(
                    active_stage["description"],
                    cls="text-sm text-gray-700 marked",
                ),
                cls="flex items-center gap-2",
            ),
            cls="bg-blue-50 p-4 rounded-lg border border-blue-100",
        ),
        hx_swap_oob="true",
        id="stage-panel",
    )


def render_react_markdown(agent: str, output: ReactProcess):
    return Div(
        f"""
**{agent} thought process**

| Thought | Action |
| --- | --- |
| {output.thought} | {output.action} |
""",
        cls="marked",
    )


def render_react_answer_markdown(agent: str, output: ReactAnswer):
    return Div(
        f"""
**{agent} answer**

{output.answer}
""",
        cls="marked",
    )


def render_agent_command_markdown(agent: str, output: AgentCommand):
    return Div(
        f"""
**{agent} task delegation**

{output.instruction}

<br>
""",
        cls="marked",
    )


def render_observation_markdown(agent: str, output: Observation):
    return Div(
        f"""
**{agent} observation**

{output.observation}
""",
        cls="marked",
    )


def render_exec_results_markdown(agent: str, output: ExecResults):
    markdown_outputs = []
    markdown_outputs.append(
        Div(
            f"""
**{agent} results**
""",
            cls="marked",
        )
    )
    for result in output.results:
        output = ""
        column_names = result.table_column_names()
        columns = result.table_columns()

        for idx, column in enumerate(columns):
            output += f"""
**{column_names[idx][0]}**

```
{column}
```
---

"""

        markdown_outputs.append(Div(output, cls="marked"))
    return Div(*markdown_outputs)


def render_bash_output_markdown(agent: str, output: str):
    return Div(
        f"""
**{agent} results**

```bash
{output}
```
""",
        cls="marked",
    )


cell_render_funcs = {
    "ReactProcess": render_react_markdown,
    "AgentCommand": render_agent_command_markdown,
    "ReactAnswer": render_react_answer_markdown,
    "Observation": render_observation_markdown,
    "ExecResults": render_exec_results_markdown,
    "BashOutput": render_bash_output_markdown,
}
