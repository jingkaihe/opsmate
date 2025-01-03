from fasthtml.common import *
from sqlmodel import Session
from pydantic import BaseModel
from opsmate.gui.assets import *
from opsmate.gui.models import (
    Cell,
    CellLangEnum,
    WorkflowEnum,
    BluePrint,
    Workflow,
    ThinkingSystemEnum,
    CellType,
    CreatedByType,
    k8s_react,
)
from opsmate.gui.components import CellComponent, CellOutputRenderer
from opsmate.dino.types import Message, Observation
from typing import Coroutine, Any

from opsmate.polya.models import (
    InitialUnderstandingResponse,
    InfoGathered,
    NonTechnicalQuery,
    TaskPlan,
    Solution,
    ReportExtracted,
)

from sqlmodel import select

from opsmate.polya.understanding import (
    initial_understanding,
    load_inital_understanding,
    info_gathering,
    generate_report,
    report_breakdown,
)
from opsmate.polya.planning import planning
import pickle
import sqlmodel
import structlog
import asyncio
import subprocess
import json
from opsmate.workflow.workflow import (
    WorkflowContext,
    step,
    WorkflowExecutor,
    build_workflow,
    cond,
    Step,
    step_factory,
)
from opsmate.workflow.workflow import (
    Workflow as OpsmateWorkflow,
    WorkflowStep as OpsmateWorkflowStep,
)

logger = structlog.get_logger()

# Set up the app, including daisyui and tailwind for the chat component
tlink = Script(src="https://cdn.tailwindcss.com?plugins=typography")
dlink = Link(
    rel="stylesheet",
    href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
)

nav = (
    Nav(
        Div(
            A("Opsmate Workspace", cls="btn btn-ghost text-xl", href="/"),
            A("Freestyle", href="/blueprint/freestyle", cls="btn btn-ghost text-sm"),
            cls="flex-1",
        ),
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


def add_cell_button(blueprint: BluePrint):
    return (
        Div(
            Button(
                add_cell_svg,
                "Add Cell",
                hx_post=f"/blueprint/{blueprint.id}/cell/bottom",
                cls="btn btn-primary btn-sm flex items-center gap-2",
            ),
            id="add-cell-button",
            hx_swap_oob="true",
            cls="flex justify-end",
        ),
    )


def reset_button(blueprint: BluePrint):
    return (
        Div(
            Button(
                "Reset",
                cls="btn btn-secondary btn-sm flex items-center gap-1",
            ),
            hx_post=f"/blueprint/{blueprint.id}/cells/reset",
            hx_swap_oob="true",
            id="reset-button",
            cls="flex",
        ),
    )


def workflow_button(workflow: Workflow):
    cls = "px-6 py-3 text-sm font-medium border-0"
    if workflow.active:
        cls += " bg-white border-b-2 border-b-blue-500 text-blue-600"
    else:
        cls += " bg-gray-50 text-gray-600 hover:bg-gray-100"
    return Button(
        workflow.title,
        hx_put=f"/workflow/{workflow.id}/switch",
        cls=cls,
    )


def marshal_output(output: dict | BaseModel | str):
    if isinstance(output, BaseModel):
        return output.model_dump()
    elif isinstance(output, str):
        return output
    elif isinstance(output, dict):
        for k, v in output.items():
            output[k] = marshal_output(v)
        return output


async def prefill_conversation(cell: Cell, session: sqlmodel.Session):
    chat_history = []
    for conversation in conversation_context(cell, session):
        chat_history.append(Message.user(conversation))
    return chat_history


def conversation_context(cell: Cell, session: sqlmodel.Session):
    workflow = cell.workflow
    previous_cells = workflow.find_previous_cells(session, cell)

    for idx, previous_cell in enumerate(previous_cells):
        assistant_response = ""
        if previous_cell.output is None:
            continue
        for output in pickle.loads(previous_cell.output):
            o = output["output"]
            marshalled_output = marshal_output(o)
            try:
                if isinstance(marshalled_output, dict) or isinstance(
                    marshalled_output, list
                ):
                    assistant_response += json.dumps(marshalled_output, indent=2) + "\n"
                else:
                    assistant_response += marshalled_output + "\n"
            except Exception as e:
                logger.error("Error marshalling output", error=e)

        conversation = f"""
Conversation {idx + 1}:

<user instruction>
{previous_cell.input}
</user instruction>

<assistant response>
{assistant_response}
</assistant response>
"""
        yield conversation


async def execute_llm_react_instruction(
    cell: Cell, swap: str, send, session: sqlmodel.Session
):

    logger.info("executing llm react instruction", cell_id=cell.id)

    chat_history = await prefill_conversation(cell, session)

    outputs = []
    await send(
        Div(
            *outputs,
            hx_swap_oob="true",
            id=f"cell-output-{cell.id}",
        )
    )
    msg = cell.input.rstrip()

    logger.info("chat_history", chat_history=chat_history)
    async for stage in k8s_react(msg, chat_history=chat_history):
        output = stage

        logger.info("output", output=output)

        partial = CellOutputRenderer.render_model(output)
        if partial:
            if isinstance(output, Observation):
                outputs.append(
                    {
                        "type": "Observation",
                        "output": Observation(
                            tool_outputs=[
                                output.__class__(**output.model_dump())
                                for output in output.tool_outputs
                            ],
                            observation=output.observation,
                        ),
                    }
                )
            else:
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


############################## Start of steps ##############################


async def initial_understanding_success_hook(ctx: WorkflowContext, result):
    session = ctx.input["session"]
    cell, _ = result
    cell = Cell.find_by_id(session, cell.id)

    cell.internal_workflow_id = ctx.workflow_id
    cell.internal_workflow_step_id = ctx.step_id

    session.add(cell)
    session.commit()


@step(post_success_hooks=[initial_understanding_success_hook])
async def manage_initial_understanding_cell(
    ctx: WorkflowContext,
):
    parent_cell = ctx.input["question_cell"]
    session = ctx.input["session"]
    send = ctx.input["send"]
    workflow = parent_cell.workflow

    cells = workflow.cells
    # get the highest sequence number
    max_sequence = max(cell.sequence for cell in cells) if cells else 0
    # get the higest execution sequence number
    max_execution_sequence = (
        max(cell.execution_sequence for cell in cells) if cells else 0
    )
    cell = ctx.input.get("current_cell")
    iu = ctx.input.get("iu")
    is_new_cell = cell is None
    if is_new_cell:
        logger.info("creating new initial understanding cell")
        cell = Cell(
            input="",
            output=b"",
            lang=CellLangEnum.TEXT_INSTRUCTION,
            thinking_system=ThinkingSystemEnum.TYPE2,
            sequence=max_sequence + 1,
            execution_sequence=max_execution_sequence + 1,
            active=True,
            workflow_id=parent_cell.workflow_id,
            cell_type=CellType.UNDERSTANDING_ASK_QUESTIONS,
            created_by=CreatedByType.ASSISTANT,
            parent_cell_ids=[parent_cell.id],
            hidden=True,
        )
        session.add(cell)
        session.commit()

        workflow.activate_cell(session, cell.id)
        session.commit()

        context = [
            conversation for conversation in conversation_context(parent_cell, session)
        ]

        iu = await initial_understanding(
            parent_cell.input.rstrip(), chat_history=context
        )
    else:
        iu = InitialUnderstandingResponse(**iu.model_dump())
        cell.hidden = True
        logger.info("updating existing initial understanding cell")

    outputs = []
    await send(
        Div(
            *outputs,
            hx_swap_oob="true",
            id=f"cell-output-{cell.id}",
        )
    )
    if isinstance(iu, NonTechnicalQuery):
        outputs.append(
            {
                "type": "NonTechnicalQuery",
                "output": NonTechnicalQuery(**iu.model_dump()),
            }
        )

        cell.output = pickle.dumps(outputs)
        session.add(cell)
        session.commit()

        # can only be a new cell
        await send(
            Div(
                CellComponent(cell),
                hx_swap_oob="beforeend",
                id="cells-container",
            )
        )

        return cell, None

    outputs.append(
        {
            "type": "InitialUnderstanding",
            "output": InitialUnderstandingResponse(**iu.model_dump()),
        }
    )

    cell.output = pickle.dumps(outputs)

    cell.input = CellOutputRenderer(outputs[0]).render()[0]
    session.add(cell)
    session.commit()

    if is_new_cell:
        await send(
            Div(
                CellComponent(cell),
                hx_swap_oob="beforeend",
                id="cells-container",
            )
        )
    else:
        await send(CellComponent(cell, hx_swap_oob="true"))

    return cell, iu


@step
async def cond_is_technical_query(ctx: WorkflowContext):
    _, iu = ctx.step_results
    return iu is not None


@step
async def manage_potential_solution_cells(
    ctx: WorkflowContext,
):
    session = ctx.input["session"]
    send = ctx.input["send"]
    cells, info_gathered = zip(*ctx.step_results)
    cells, info_gathered = list(cells), list(info_gathered)
    initial_understanding_result = ctx.find_result(
        "manage_initial_understanding_cell", session
    )
    _, iu = initial_understanding_result
    summary = iu.summary

    report = await generate_report(summary, info_gathered=info_gathered)
    report_extracted = await report_breakdown(report)
    for solution in report_extracted.potential_solutions:
        await __manage_potential_solution_cell(
            cells, report_extracted.summary, solution, session, send
        )
    # xxx: pickle issue occur need resolving
    return ReportExtracted(**report_extracted.model_dump())


@step
async def generate_report_with_breakdown(ctx: WorkflowContext):
    session = ctx.input["session"]
    send = ctx.input["send"]
    cells, info_gathered = zip(*ctx.step_results)
    cells, info_gathered = list(cells), list(info_gathered)
    initial_understanding_result = ctx.find_result(
        "manage_initial_understanding_cell", session
    )
    _, iu = initial_understanding_result
    summary = iu.summary

    report = await generate_report(summary, info_gathered=info_gathered)
    report_extracted = await report_breakdown(report)

    return cells, ReportExtracted(**report_extracted.model_dump())


def make_manage_potential_solution_cell(solution_id: int):
    async def success_hook(ctx: WorkflowContext, result):
        session = ctx.input["session"]
        cell = Cell.find_by_id(session, result.id)
        cell.internal_workflow_id = ctx.workflow_id
        cell.internal_workflow_step_id = ctx.step_id
        session.add(cell)
        session.commit()

    @step_factory
    @step(post_success_hooks=[success_hook])
    async def manage_potential_solution_cell(ctx: WorkflowContext):
        cells, report_extracted = ctx.step_results
        session = ctx.input["session"]
        send = ctx.input["send"]
        solution_id = ctx.metadata["solution_id"]
        if len(report_extracted.potential_solutions) <= solution_id:
            return None
        return await __manage_potential_solution_cell(
            cells,
            report_extracted.summary,
            report_extracted.potential_solutions[solution_id],
            session,
            send,
        )

    return manage_potential_solution_cell(metadata={"solution_id": solution_id})


manage_potential_solution_cells = [
    make_manage_potential_solution_cell(0),
    make_manage_potential_solution_cell(1),
    make_manage_potential_solution_cell(2),
]


@step
async def store_report_extracted(ctx: WorkflowContext):
    cell = ctx.input["question_cell"]
    session = ctx.input["session"]

    _, report_extracted = ctx.find_result("generate_report_with_breakdown", session)

    workflow = Workflow.find_by_id(session, cell.workflow_id)
    workflow.result = report_extracted.model_dump_json()
    session.add(workflow)
    session.commit()


def make_manage_info_gathering_cell(question_id: int):
    async def success_hook(ctx: WorkflowContext, result):
        session = ctx.input["session"]
        cell, _ = result
        cell = Cell.find_by_id(session, cell.id)
        cell.internal_workflow_id = ctx.workflow_id
        cell.internal_workflow_step_id = ctx.step_id
        session.add(cell)
        session.commit()

    @step_factory
    @step(
        post_success_hooks=[success_hook],
    )
    async def manage_info_gathering_cell(
        ctx: WorkflowContext,
    ):
        session = ctx.input["session"]
        send = ctx.input["send"]
        parent_cell, iu = ctx.find_result("manage_initial_understanding_cell", session)
        info_gather_task = info_gathering(
            iu.summary, iu.questions[ctx.metadata["question_id"]]
        )
        return await __manage_info_gathering_cell(
            parent_cell, info_gather_task, session, send
        )

    return manage_info_gathering_cell(metadata={"question_id": question_id})


manage_info_gather_cells = [
    make_manage_info_gathering_cell(0),
    make_manage_info_gathering_cell(1),
    make_manage_info_gathering_cell(2),
]


async def __manage_info_gathering_cell(
    parent_cell: Cell,
    info_gather_task: Coroutine[Any, Any, InfoGathered],
    session: sqlmodel.Session,
    send,
):
    workflow = parent_cell.workflow

    cells = workflow.cells
    # get the highest sequence number
    max_sequence = max(cell.sequence for cell in cells) if cells else 0
    # get the higest execution sequence number
    max_execution_sequence = (
        max(cell.execution_sequence for cell in cells) if cells else 0
    )

    outputs = []
    info_gathered = await info_gather_task
    info_gathered = InfoGathered(**info_gathered.model_dump())
    outputs.append(
        {
            "type": "InfoGathered",
            "output": info_gathered,
        }
    )

    cell = Cell(
        input=info_gathered.question,
        output=pickle.dumps(outputs),
        lang=CellLangEnum.TEXT_INSTRUCTION,
        thinking_system=ThinkingSystemEnum.TYPE2,
        sequence=max_sequence + 1,
        execution_sequence=max_execution_sequence + 1,
        active=True,
        workflow_id=parent_cell.workflow_id,
        parent_cell_ids=[parent_cell.id],
        cell_type=CellType.UNDERSTANDING_GATHER_INFO,
        created_by=CreatedByType.ASSISTANT,
        hidden=True,
    )
    session.add(cell)
    session.commit()

    workflow.activate_cell(session, cell.id)
    session.commit()

    await send(
        Div(
            CellComponent(cell),
            hx_swap_oob="beforeend",
            id="cells-container",
        )
    )

    return cell, info_gathered


async def __manage_potential_solution_cell(
    parent_cells: list[Cell],
    summary: str,
    solution: Solution,
    session: sqlmodel.Session,
    send,
):
    # workflow = parent_cells[0].workflow
    workflow = Workflow.find_by_id(session, parent_cells[0].workflow_id)

    cells = workflow.cells
    # get the highest sequence number
    max_sequence = max(cell.sequence for cell in cells) if cells else 0
    # get the higest execution sequence number
    max_execution_sequence = (
        max(cell.execution_sequence for cell in cells) if cells else 0
    )

    outputs = []
    output = {
        "summary": summary,
        "solution": Solution(**solution.model_dump()),
    }
    outputs.append(
        {
            "type": "PotentialSolution",
            "output": output,
        }
    )

    logger.info(
        "create potential solution cell",
        parent_cells=[parent_cell.id for parent_cell in parent_cells],
    )

    cell = Cell(
        input=CellOutputRenderer(outputs[0]).render()[0],
        output=pickle.dumps(outputs),
        lang=CellLangEnum.NOTES,
        thinking_system=ThinkingSystemEnum.TYPE1,
        sequence=max_sequence + 1,
        execution_sequence=max_execution_sequence + 1,
        active=True,
        workflow_id=parent_cells[0].workflow_id,
        parent_cell_ids=[parent_cell.id for parent_cell in parent_cells],
        cell_type=CellType.UNDERSTANDING_SOLUTION,
        created_by=CreatedByType.ASSISTANT,
        hidden=True,
    )
    session.add(cell)
    session.commit()

    workflow.activate_cell(session, cell.id)
    session.commit()

    await send(
        Div(
            CellComponent(cell),
            hx_swap_oob="beforeend",
            id="cells-container",
        )
    )

    return cell


@step
async def empty_cell(ctx: WorkflowContext):
    session = ctx.input["session"]
    cell = ctx.input["question_cell"]
    send = ctx.input["send"]
    await send(
        Div(
            hx_swap_oob="true",
            id=f"cell-output-{cell.id}",
        )
    )


############################## End of steps ##############################


async def execute_llm_type2_instruction(
    cell: Cell, swap: str, send, session: sqlmodel.Session
):
    workflow = cell.workflow
    if workflow.name == WorkflowEnum.UNDERSTANDING:
        if cell.cell_type == CellType.UNDERSTANDING_ASK_QUESTIONS:
            return await update_initial_understanding(cell, send, session)
        elif cell.cell_type == CellType.UNDERSTANDING_GATHER_INFO:
            return await execute_info_gathering(cell, send, session)
        else:
            return await execute_polya_understanding_instruction(cell, send, session)
    elif workflow.name == WorkflowEnum.PLANNING:
        return await execute_polya_planning_instruction(cell, swap, send, session)
    elif workflow.name == WorkflowEnum.EXECUTION:
        return await execute_polya_execution_instruction(cell, swap, send, session)


async def execute_polya_understanding_instruction(
    cell: Cell, send, session: sqlmodel.Session
):
    msg = cell.input.rstrip()
    logger.info("executing polya understanding instruction", cell_id=cell.id, input=msg)

    blueprint = (
        empty_cell
        >> manage_initial_understanding_cell
        >> cond(
            cond_is_technical_query,
            left=(
                reduce(lambda x, y: x | y, manage_info_gather_cells)
                >> generate_report_with_breakdown
                >> reduce(lambda x, y: x | y, manage_potential_solution_cells)
                >> store_report_extracted
            ),
        )
    )

    opsmate_workflow = build_workflow(
        "understanding",
        "Understand the problem",
        blueprint,
        session,
    )
    executor = WorkflowExecutor(opsmate_workflow, session)
    ctx = WorkflowContext(
        input={"session": session, "question_cell": cell, "send": send}
    )

    await executor.run(ctx)


async def update_initial_understanding(
    cell: Cell,
    send,
    session: sqlmodel.Session,
):
    iu = await load_inital_understanding(cell.input)

    opsmate_workflow_step = session.exec(
        select(OpsmateWorkflowStep)
        .where(OpsmateWorkflowStep.id == cell.internal_workflow_step_id)
        .where(OpsmateWorkflowStep.workflow_id == cell.internal_workflow_id)
    ).first()
    if not opsmate_workflow_step:
        logger.error("Opsmate workflow step not found", cell_id=cell.id)
        return
    opsmate_workflow = opsmate_workflow_step.workflow

    executor = WorkflowExecutor(opsmate_workflow, session)
    await executor.mark_rerun(opsmate_workflow_step)

    await executor.run(
        WorkflowContext(
            input={
                "session": session,
                "question_cell": cell.parent_cells(session)[0],
                "current_cell": cell,
                "iu": iu,
                "send": send,
            }
        )
    )


async def execute_info_gathering(
    cell: Cell,
    send,
    session: sqlmodel.Session,
):
    outputs = []
    await send(
        Div(
            *outputs,
            hx_swap_oob="true",
            id=f"cell-output-{cell.id}",
        )
    )
    parent_cells = cell.parent_cells(session)
    if len(parent_cells) == 0:
        logger.error("No parent cells found", cell_id=cell.id)
        return

    parent_cell = parent_cells[0]

    parent_outputs = pickle.loads(parent_cell.output)
    iu = parent_outputs[0]["output"]
    summary = iu.summary
    info_gather_task = info_gathering(summary, cell.input)

    info_gathered = await info_gather_task
    info_gathered = InfoGathered(**info_gathered.model_dump())

    outputs.append(
        {
            "type": "InfoGathered",
            "output": info_gathered,
        }
    )

    cell.output = pickle.dumps(outputs)
    cell.hidden = True
    session.add(cell)
    session.commit()

    await send(
        Div(
            *[CellOutputRenderer(output).render() for output in outputs],
            hx_swap_oob="true",
            id=f"cell-output-{cell.id}",
        )
    )

    info_gather_cells = session.exec(
        select(Cell).where(
            Cell.workflow_id == cell.workflow_id,
            Cell.cell_type == CellType.UNDERSTANDING_GATHER_INFO,
        )
    ).all()

    infos_gathered = []
    for info_gather_cell in info_gather_cells:
        info_gathered = pickle.loads(info_gather_cell.output)[0]["output"]
        infos_gathered.append(info_gathered)

    logger.info(
        "info_gather_cells", info_gather_cells=[cell.id for cell in info_gather_cells]
    )

    report_extracted = await manage_potential_solution_cells(
        iu.summary,
        info_gather_cells,
        infos_gathered,
        session,
        send,
    )

    workflow = cell.workflow
    workflow.result = report_extracted.model_dump_json()
    session.add(workflow)
    session.commit()


async def execute_polya_planning_instruction(
    cell: Cell, swap: str, send, session: sqlmodel.Session
):
    msg = cell.input.rstrip()
    logger.info("executing polya understanding instruction", cell_id=cell.id, input=msg)

    blueprint = cell.workflow.blueprint
    understanding_workflow: Workflow = blueprint.find_workflow_by_name(
        session, WorkflowEnum.UNDERSTANDING
    )
    report_extracted_json = understanding_workflow.result

    # pick the most optimal solution
    report_extracted = ReportExtracted.model_validate_json(report_extracted_json)
    solution = report_extracted.potential_solutions[0]
    summary = report_extracted.summary

    outputs = []
    await send(
        Div(
            *outputs,
            hx_swap_oob="true",
            id=f"cell-output-{cell.id}",
        )
    )

    task_plan = await planning(msg, solution.summarize(summary))
    outputs.append(
        {
            "type": "TaskPlan",
            "output": TaskPlan(**task_plan.model_dump()),
        }
    )

    await send(
        Div(
            *[CellOutputRenderer(output).render() for output in outputs],
            hx_swap_oob=swap,
            id=f"cell-output-{cell.id}",
        )
    )

    cell.output = pickle.dumps(outputs)
    session.add(cell)
    session.commit()

    workflow = cell.workflow
    workflow.result = task_plan.model_dump_json()
    session.add(workflow)
    session.commit()


async def execute_polya_execution_instruction(
    cell: Cell, swap: str, send, session: sqlmodel.Session
):
    msg = cell.input.rstrip()
    logger.info("executing polya execution instruction", cell_id=cell.id, input=msg)

    blueprint = cell.workflow.blueprint
    understanding_workflow: Workflow = blueprint.find_workflow_by_name(
        session, WorkflowEnum.UNDERSTANDING
    )
    report_extracted_json = understanding_workflow.result
    report_extracted = ReportExtracted.model_validate_json(report_extracted_json)

    solution_summary = report_extracted.potential_solutions[0].summarize(
        report_extracted.summary, show_probability=False
    )

    planning_workflow: Workflow = blueprint.find_workflow_by_name(
        session, WorkflowEnum.PLANNING
    )
    task_plan_json = planning_workflow.result

    task_plan = TaskPlan.model_validate_json(task_plan_json)

    print(task_plan.model_dump_json(indent=2))


async def execute_notes_instruction(
    cell: Cell, swap: str, send, session: sqlmodel.Session
):
    logger.info("executing notes instruction", cell_id=cell.id)

    output = {
        "type": "NotesOutput",
        "output": cell.input,
    }
    outputs = [output]
    await send(
        Div(
            CellOutputRenderer(output).render(),
            hx_swap_oob="true",
            id=f"cell-output-{cell.id}",
        )
    )

    cell.output = pickle.dumps(outputs)
    cell.hidden = True
    session.add(cell)
    session.commit()

    textarea = CellComponent(cell).cell_text_area()
    textarea.hx_swap_oob = "true"
    await send(textarea)


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
            *[CellOutputRenderer(output).render() for output in outputs],
            hx_swap_oob=swap,
            id=f"cell-output-{cell.id}",
        )
    )


def home_body(db_session: Session, session_name: str, blueprint: BluePrint):
    active_workflow = blueprint.active_workflow(db_session)
    workflows = blueprint.workflows
    cells = active_workflow.cells

    logger.info(
        "home body",
        cells=[cell.id for cell in cells],
        sequence=[cell.sequence for cell in cells],
    )
    return Body(
        Div(
            Card(
                # Header
                Div(
                    Div(
                        H1(session_name, cls="text-2xl font-bold"),
                        Span(
                            "Press Shift+Enter to run cell",
                            cls="text-sm text-gray-500",
                        ),
                        cls="flex flex-col",
                    ),
                    Div(
                        reset_button(blueprint),
                        add_cell_button(blueprint),
                        cls="flex gap-2 justify-start",
                    ),
                    cls="mb-4 flex justify-between items-start pt-16",
                ),
                render_workflow_panel(workflows, active_workflow),
                # Cells Container
                render_cell_container(cells),
                # cls="overflow-hidden",
            ),
            cls="max-w-6xl mx-auto p-4 bg-gray-50 min-h-screen",
        )
    )


def render_workflow_panel(workflows: list[Workflow], active_workflow: Workflow):
    return Div(
        Div(
            *[workflow_button(workflow) for workflow in workflows],
            cls="flex border-t",
        ),
        # workflow Panels
        Div(
            Div(
                Div(
                    Span(
                        f"Current Phase: {active_workflow.title}",
                        cls="font-medium",
                    ),
                    cls="flex items-center gap-2 text-sm text-gray-500",
                ),
                cls="space-y-6",
            ),
            cls="block p-4",
        ),
        # workflow description
        Div(
            Div(
                Div(
                    active_workflow.description,
                    cls="text-sm text-gray-700 marked prose max-w-none",
                ),
                cls="flex items-center gap-2",
            ),
            cls="bg-blue-50 p-4 rounded-lg border border-blue-100",
        ),
        hx_swap_oob="true",
        id="workflow-panel",
    )


def render_cell_container(cells: list[Cell], hx_swap_oob: str = None):
    div = Div(
        *[CellComponent(cell) for cell in cells],
        cls="space-y-4 mt-4",
        id="cells-container",
        ws_connect="/cell/run/ws/",
        hx_ext="ws",
    )
    if hx_swap_oob:
        div.hx_swap_oob = hx_swap_oob
    return div


async def async_wrapper(generator: Generator):
    for stage in generator:
        await asyncio.sleep(0)
        yield stage
