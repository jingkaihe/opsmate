import structlog
import sqlmodel
from fasthtml.common import *
from opsmate.gui.models import (
    Cell,
    CellLangEnum,
    ThinkingSystemEnum,
    BluePrint,
    Workflow,
    default_new_cell,
    CellStateEnum,
    EnvVar,
    ExecutionConfirmation,
    default_new_cell,
)
from opsmate.gui.config import Config
from opsmate.gui.views import (
    tlink,
    dlink,
    picolink,
    nav,
    reset_button,
    add_cell_button,
    render_cells_container,
    render_workflow_panel,
    execute_llm_simple_instruction,
    execute_llm_react_instruction,
    execute_llm_type2_instruction,
    execute_bash_instruction,
    execute_notes_instruction,
    home_body,
    knowledges_body,
    new_knowledge_form,
    add_knowledge_button,
    render_knowledge_row,
    settings_body,
    new_envvar_form,
    add_envvar_button,
)
from opsmate.gui.components import CellComponent
from opsmate.ingestions import ingest_from_config
from opsmate.ingestions.models import IngestionRecord
from opsmate.ingestions.jobs import ingest, delete_ingestion
from opsmate.dbq.dbq import enqueue_task
from sqlmodel import text
from uuid import uuid4

config = Config()


logger = structlog.get_logger()


# start a sqlite database
engine = sqlmodel.create_engine(
    config.db_url,
    connect_args={"check_same_thread": False, "timeout": 20},
    # echo=True
)

with engine.connect() as conn:
    conn.execute(text("PRAGMA journal_mode=WAL"))
    conn.close()


def before(req, session):
    if config.token == "":
        session["token"] = ""
        return
    if req.query_params.get("token") is not None:
        session["token"] = req.query_params.get("token", "")

    if session.get("token", "") != config.token:
        return Response("unauthorized", status_code=401)


bware = Beforeware(before)

app = FastHTML(
    hdrs=(
        tlink,
        dlink,
        picolink,
        MarkdownJS(),
        HighlightJS(langs=("python", "bash")),
        nav,
    ),
    exts="ws",
    before=bware,
)


async def kb_ingest():
    await ingest_from_config(config, engine)


@app.on_event("startup")
async def startup():
    dev = os.environ.get("DEV", "false").lower() == "true"
    if dev:
        await kb_ingest()


@app.route("/")
async def get():
    with sqlmodel.Session(engine) as session:
        blueprint = BluePrint.find_by_name(session, "polya")
        page = home_body(session, config.session_name, blueprint)
        return Title(f"{config.session_name}"), page


@app.route("/knowledges")
async def get():
    with sqlmodel.Session(engine) as session:
        return Title("Knowledges"), knowledges_body(session, config.session_name)


@app.route("/settings")
async def get():
    with sqlmodel.Session(engine) as session:
        return Title("Settings"), settings_body(session, config.session_name)


@app.route("/settings/envvars/new")
async def post():
    with sqlmodel.Session(engine) as session:
        uuid = str(uuid4())
        return (
            Div(
                new_envvar_form(uuid),
                hx_swap_oob="beforeend",
                id="envvars",
            ),
            add_envvar_button(),
        )


@app.route("/settings/envvars/virtual/{uuid}")
async def delete(uuid: str):
    with sqlmodel.Session(engine) as session:
        return (
            Div(
                id=f"new-envvar-form-{uuid}",
                hx_swap_oob="delete",
            ),
        )


@app.route("/settings/envvars/")
async def post(key: str, value: str):
    with sqlmodel.Session(engine) as session:
        envvar = EnvVar(key=key.strip(), value=value.strip())
        session.add(envvar)
        session.commit()

        return Title("Settings"), settings_body(session, config.session_name)


@app.route("/settings/envvars/{id}")
async def delete(id: str):
    with sqlmodel.Session(engine) as session:
        envvar = EnvVar.find_by_id(session, id)
        session.delete(envvar)
        session.commit()
        return Title("Settings"), settings_body(session, config.session_name)


@app.route("/settings/envvars/{id}")
async def put(id: str, value: str):
    with sqlmodel.Session(engine) as session:
        envvar = EnvVar.find_by_id(session, id)
        envvar.value = value.strip()
        session.add(envvar)
        session.commit()
        return Title("Settings"), settings_body(session, config.session_name)


@app.route("/knowledges/{id}")
async def put(id: str, branch: str, glob: str):
    with sqlmodel.Session(engine) as session:
        ingestion_record = await IngestionRecord.find_by_id(session, id)
        ingestion_record.branch = branch
        ingestion_record.glob = glob
        session.add(ingestion_record)
        session.commit()
        logger.info("ingesting knowledge", ingestion_record_id=ingestion_record.id)
        enqueue_task(
            session,
            ingest,
            ingestor_type=ingestion_record.data_source_provider,
            ingestor_config=await ingestion_record.ingest_config(),
            splitter_config=ingestion_record.splitter_config,
        )
        return Title("Knowledges"), knowledges_body(session, config.session_name)


@app.route("/knowledges/new")
async def post():
    with sqlmodel.Session(engine) as session:
        uuid = str(uuid4())
        return (
            Div(
                new_knowledge_form(uuid),
                hx_swap_oob="beforeend",
                id="ingestions",
            ),
            add_knowledge_button(),
        )


@app.route("/knowledges/virtual/{uuid}")
async def delete(uuid: str):
    with sqlmodel.Session(engine) as session:
        return (
            Div(
                id=f"new-ingestion-form-{uuid}",
                hx_swap_oob="delete",
            ),
        )


@app.route("/knowledges/")
async def post(
    uuid: str,
    data_source: str,
    data_source_provider: str,
    branch: str,
    glob: str,
):
    with sqlmodel.Session(engine) as session:
        ingestion_record = IngestionRecord(
            data_source_provider=data_source_provider,
            data_source=data_source,
            glob=glob,
            branch=branch,
        )
        session.add(ingestion_record)
        session.commit()

        logger.info("enqueueing ingestion", ingestion_record_id=ingestion_record.id)
        enqueue_task(
            session,
            ingest,
            ingestor_type=ingestion_record.data_source_provider,
            ingestor_config=await ingestion_record.ingest_config(),
            splitter_config=config.splitter_config,
        )

        return (
            Div(
                render_knowledge_row(ingestion_record),
                hx_swap_oob="beforeend",
                id="ingestions",
            ),
            Div(
                id=f"new-ingestion-form-{uuid}",
                hx_swap_oob="delete",
            ),
        )


@app.route("/knowledges/{id}")
async def delete(id: str):
    with sqlmodel.Session(engine) as session:
        logger.info("deleting knowledge", id=id)
        enqueue_task(
            session,
            delete_ingestion,
            id,
        )
        return (
            Div(
                id=f"ingestion-record-{id}",
                hx_swap_oob="delete",
            ),
            add_knowledge_button(),
        )


@app.route("/blueprint/freestyle")
async def get():
    with sqlmodel.Session(engine) as session:
        blueprint = BluePrint.find_by_name(session, "freestyle")
        page = home_body(session, config.session_name, blueprint)
        return Title(f"{config.session_name}"), page


@app.route("/blueprint/{blueprint_id}/cell/bottom")
async def post(blueprint_id: int):
    with sqlmodel.Session(engine) as session:
        blueprint = BluePrint.find_by_id(session, blueprint_id)
        active_workflow = blueprint.active_workflow(session)
        cells = active_workflow.cells

        # get the highest sequence number
        max_sequence = max(cell.sequence for cell in cells) if cells else 0
        # get the higest execution sequence number
        max_execution_sequence = (
            max(cell.execution_sequence for cell in cells) if cells else 0
        )
        new_cell = default_new_cell(active_workflow)
        new_cell.sequence = max_sequence + 1
        new_cell.execution_sequence = max_execution_sequence + 1

        session.add(new_cell)
        session.commit()

        active_workflow.activate_cell(session, new_cell.id)

        session.refresh(active_workflow)
        cells = active_workflow.cells
        return (
            # Return the new cell to be added
            render_cells_container(cells, hx_swap_oob="true"),
            # Return the button to preserve it
            add_cell_button(blueprint),
        )


# Add cell manipulation routes
@app.route("/blueprint/{blueprint_id}/cell/{cell_id}")
async def post(
    blueprint_id: int,
    cell_id: int,
    above: bool = False,
    session: sqlmodel.Session = None,
):
    with sqlmodel.Session(engine) as session:
        blueprint = BluePrint.find_by_id(session, blueprint_id)
        active_workflow = blueprint.active_workflow(session)
        selected_cell = active_workflow.find_cell_by_id(session, cell_id)
        cells = active_workflow.cells

        new_cell = default_new_cell(active_workflow)

        # get the highest execution sequence number
        max_execution_sequence = (
            max(cell.execution_sequence for cell in cells) if cells else 0
        )
        new_cell.execution_sequence = max_execution_sequence + 1

        if above:
            new_cell.sequence = selected_cell.sequence
        else:
            new_cell.sequence = selected_cell.sequence + 1

        session.add(new_cell)
        # find all cells with a sequence greater than the current cell
        cells_to_shift = [cell for cell in cells if cell.sequence >= new_cell.sequence]
        for cell in cells_to_shift:
            cell.sequence += 1
            session.add(cell)
        session.commit()

        # reload the cells
        active_workflow.activate_cell(session, new_cell.id)
        session.refresh(active_workflow)
        cells = active_workflow.cells
        return render_cells_container(cells, hx_swap_oob="true")


@app.route("/blueprint/{blueprint_id}/cell/{cell_id}")
async def delete(blueprint_id: int, cell_id: int):
    with sqlmodel.Session(engine) as session:
        blueprint = BluePrint.find_by_id(session, blueprint_id)
        active_workflow = blueprint.active_workflow(session)
        selected_cell = active_workflow.find_cell_by_id(session, cell_id)

        if selected_cell is None:
            return ""

        deleted_cell_count = Cell.delete_cell(session, cell_id)
        session.commit()

        logger.info(
            "deleted cell", cell_id=cell_id, deleted_cell_count=deleted_cell_count
        )

        # find all cells with a sequence greater than the current cell
        cells_to_shift = session.exec(
            sqlmodel.select(Cell)
            .where(Cell.workflow_id == active_workflow.id)
            .where(Cell.sequence > selected_cell.sequence)
        ).all()

        logger.info(
            "cells to shift", cells_to_shift=[cell.id for cell in cells_to_shift]
        )
        for idx, cell in enumerate(cells_to_shift):
            cell.sequence = cell.sequence + idx
            session.add(cell)
        session.commit()

        blueprint = BluePrint.find_by_id(session, blueprint_id)
        active_workflow = blueprint.active_workflow(session)
        cells = active_workflow.cells

        return render_cells_container(cells, hx_swap_oob="true")


@app.route("/blueprint/{blueprint_id}/cell/{cell_id}")
async def put(
    blueprint_id: int,
    cell_id: int,
    input: str = None,
    lang: str = None,
    thinking_system: str = None,
    hidden: bool = False,
):
    logger.info(
        "updating cell",
        cell_id=cell_id,
        input=input,
        lang=lang,
        thinking_system=thinking_system,
        hidden=hidden,
    )

    with sqlmodel.Session(engine) as session:
        blueprint = BluePrint.find_by_id(session, blueprint_id)
        active_workflow = blueprint.active_workflow(session)
        selected_cell = active_workflow.find_cell_by_id(session, cell_id)
        if selected_cell is None:
            return ""

        selected_cell.hidden = hidden
        selected_cell.active = True
        if input is not None:
            selected_cell.input = input
        if lang is not None:
            if lang == CellLangEnum.TEXT_INSTRUCTION.value:
                selected_cell.lang = CellLangEnum.TEXT_INSTRUCTION
            elif lang == CellLangEnum.BASH.value:
                selected_cell.lang = CellLangEnum.BASH
            elif lang == CellLangEnum.NOTES.value:
                selected_cell.lang = CellLangEnum.NOTES

        if thinking_system is not None:
            if thinking_system == ThinkingSystemEnum.REASONING.value:
                logger.info("setting thinking system to type 1", cell_id=cell_id)
                selected_cell.thinking_system = ThinkingSystemEnum.REASONING
            elif thinking_system == ThinkingSystemEnum.SIMPLE.value:
                logger.info("setting thinking system to simple", cell_id=cell_id)
                selected_cell.thinking_system = ThinkingSystemEnum.SIMPLE
            elif thinking_system == ThinkingSystemEnum.TYPE2.value:
                logger.info("setting thinking system to type 2", cell_id=cell_id)
                selected_cell.thinking_system = ThinkingSystemEnum.TYPE2
            else:
                logger.error(
                    "unknown thinking system",
                    cell_id=cell_id,
                    thinking_system=thinking_system,
                )

        session.add(selected_cell)
        session.commit()

        active_workflow.activate_cell(session, selected_cell.id)

        session.refresh(active_workflow)
        cells = active_workflow.cells

        return render_cells_container(cells, hx_swap_oob="true")


@app.route("/blueprint/{blueprint_id}/cell/input/{cell_id}")
async def put(blueprint_id: int, cell_id: int, input: str):
    with sqlmodel.Session(engine) as session:
        blueprint = BluePrint.find_by_id(session, blueprint_id)
        active_workflow = blueprint.active_workflow(session)
        selected_cell = active_workflow.find_cell_by_id(session, cell_id)

        if selected_cell is None:
            return ""

        selected_cell.input = input
        selected_cell.active = True
        session.add(selected_cell)
        session.commit()

        active_workflow.activate_cell(session, selected_cell.id)
        return ""


@app.route("/blueprint/{blueprint_id}/cell/{cell_id}/stop")
async def put(blueprint_id: int, cell_id: int):
    """
    Stop a cell

    This does not actually stop the cell but instead mark the cell
    """
    with sqlmodel.Session(engine) as session:
        blueprint = BluePrint.find_by_id(session, blueprint_id)
        active_workflow = blueprint.active_workflow(session)
        selected_cell = active_workflow.find_cell_by_id(session, cell_id)
        if selected_cell is None:
            return ""

        selected_cell.state = CellStateEnum.STOPPING
        session.add(selected_cell)
        session.commit()

        active_workflow.activate_cell(session, selected_cell.id)

        return Div(
            CellComponent(selected_cell),
            hx_swap_oob="true",
            id=f"cell-component-{selected_cell.id}",
        )


@app.route("/workflow/{workflow_id}/switch")
async def put(workflow_id: str):
    logger.info("switching workflow", workflow_id=workflow_id)

    with sqlmodel.Session(engine) as session:
        workflow = Workflow.find_by_id(session, workflow_id)
        blueprint = workflow.blueprint
        blueprint.activate_workflow(session, workflow_id)

        session.refresh(blueprint)
        active_workflow = blueprint.active_workflow(session)

        return (
            render_workflow_panel(blueprint.workflows, active_workflow),
            render_cells_container(active_workflow.cells, hx_swap_oob="true"),
        )


@app.route("/blueprint/{blueprint_id}/cells/reset")
async def post(blueprint_id: int):
    with sqlmodel.Session(engine) as session:
        blueprint = BluePrint.find_by_id(session, blueprint_id)
        active_workflow = blueprint.active_workflow(session)
        session.exec(
            sqlmodel.delete(Cell).where(Cell.workflow_id == active_workflow.id)
        )
        session.commit()

        # create new cells
        new_cell = default_new_cell(active_workflow)
        session.add(new_cell)
        session.commit()

        session.refresh(active_workflow)
        session.refresh(new_cell)
        return (
            render_cells_container(active_workflow.cells, hx_swap_oob="true"),
            reset_button(blueprint),
        )


@app.ws("/cell/run/ws/")
async def ws(cell_id: int, input: str, send, session):
    logger.info("running cell", cell_id=cell_id, input=input)
    # Check authentication token
    if session.get("token", "") != config.token:
        logger.error("unauthorized", token=session.get("token"))
        return  # Exit if unauthorized

    with sqlmodel.Session(engine) as session:
        cell = Cell.find_by_id(session, cell_id)
        active_workflow = cell.workflow
        active_workflow.activate_cell(session, cell_id)

        cell = session.exec(sqlmodel.select(Cell).where(Cell.id == cell_id)).first()
        logger.info(
            "selected cell",
            cell_id=cell_id,
            input=cell.input,
            cell_lang=cell.lang,
        )
        cell.active = True

        if cell.lang == CellLangEnum.NOTES:
            logger.info("hiding notescell", cell_id=cell_id)
            cell.hidden = True

        session.add(cell)
        session.commit()

        if cell is None:
            logger.error("cell not found", cell_id=cell_id)
            return

        deleted_cell_ids = Cell.delete_cell(session, cell_id, children_only=True)
        session.commit()

        logger.info("deleted cells", deleted_cell_ids=deleted_cell_ids)

        # find all cells with a sequence greater than the current cell
        cells_to_shift = session.exec(
            sqlmodel.select(Cell)
            .where(Cell.workflow_id == active_workflow.id)
            .where(Cell.sequence > cell.sequence)
        ).all()

        logger.info(
            "cells to shift",
            cells_to_shift=[cell.id for cell in cells_to_shift],
            sequences=[cell.sequence for cell in cells_to_shift],
        )
        for idx, cell_to_shift in enumerate(cells_to_shift):
            cell_to_shift.sequence = cell.sequence + idx + 1
            session.add(cell_to_shift)
        session.commit()

        logger.info(
            "cells shifted",
            cells_to_shift=[cell.id for cell in cells_to_shift],
            sequences=[cell.sequence for cell in cells_to_shift],
        )

        for deleted_cell_id in deleted_cell_ids:
            await send(
                Div(
                    id=f"cell-component-{deleted_cell_id}",
                    hx_swap_oob="delete",
                )
            )

        logger.info(
            "executing cell",
            cell_id=cell_id,
            cell_lang=cell.lang.value,
            input=cell.input,
            thinking_system=cell.thinking_system.value,
        )
        swap = "beforeend"
        if cell.lang == CellLangEnum.TEXT_INSTRUCTION:
            if cell.thinking_system == ThinkingSystemEnum.SIMPLE:
                await execute_llm_simple_instruction(cell, swap, send, session)
            elif cell.thinking_system == ThinkingSystemEnum.REASONING:
                await execute_llm_react_instruction(cell, swap, send, session)
            elif cell.thinking_system == ThinkingSystemEnum.TYPE2:
                await execute_llm_type2_instruction(cell, swap, send, session)
        elif cell.lang == CellLangEnum.BASH:
            await execute_bash_instruction(cell, swap, send, session)
        elif cell.lang == CellLangEnum.NOTES:
            await execute_notes_instruction(cell, swap, send, session)
        else:
            logger.error("unknown cell type", cell_id=cell.id, cell_lang=cell.lang)


@app.route("/execution_confirmation/{id}")
async def post(id: int, command: str):
    with sqlmodel.Session(engine) as session:
        confirmation = ExecutionConfirmation.find_by_id(session, id)
        confirmation.confirmed = True
        confirmation.command = command
        session.add(confirmation)
        session.commit()

        return Div(
            id=f"confirmation-form-{confirmation.id}",
            hx_swap="delete",
        )


if __name__ == "__main__":
    serve()
