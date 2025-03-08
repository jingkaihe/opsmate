import enum
from sqlmodel import (
    SQLModel as _SQLModel,
    Field,
    Column,
    Enum,
    LargeBinary,
    update,
    select,
    Session,
    JSON,
    Text,
    MetaData,
)
from datetime import datetime
from typing import List, Dict
from sqlmodel import Relationship
import structlog
from opsmate.dino.types import Message, Observation, ToolCall
from opsmate.dino.react import react
from opsmate.dino import dino
from sqlalchemy.orm import registry
from opsmate.gui.config import Config
import yaml
import pickle
from pydantic import BaseModel
from opsmate.contexts import k8s_ctx

logger = structlog.get_logger(__name__)


class SQLModel(_SQLModel, registry=registry()):
    metadata = MetaData()


class CellLangEnum(enum.Enum):
    TEXT_INSTRUCTION = "text instruction"
    NOTES = "notes"
    BASH = "bash"


class WorkflowEnum(str, enum.Enum):
    FREESTYLE = "freestyle"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"


class ThinkingSystemEnum(str, enum.Enum):
    SIMPLE = "simple"
    REASONING = "reasoning"
    TYPE2 = "type-2"


class BluePrint(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: int = Field(primary_key=True, sa_column_kwargs={"autoincrement": True})
    name: str = Field(unique=True, index=True)
    description: str = Field(default="")

    workflows: List["Workflow"] = Relationship(
        back_populates="blueprint", sa_relationship_kwargs={"order_by": "Workflow.id"}
    )

    @classmethod
    def find_by_id(cls, session: Session, id: int):
        return session.exec(select(cls).where(cls.id == id)).first()

    @classmethod
    def find_by_name(cls, session: Session, name: str):
        return session.exec(select(cls).where(cls.name == name)).first()

    @classmethod
    def find_workflow_by_name(cls, session: Session, name: str):
        return session.exec(
            select(Workflow)
            .where(Workflow.blueprint_id == cls.id)
            .where(Workflow.name == name)
        ).first()

    def active_workflow(self, session: Session):
        return session.exec(
            select(Workflow)
            .where(Workflow.blueprint_id == self.id)
            .where(Workflow.active == True)
        ).first()

    def activate_workflow(self, session: Session, id: int):
        # update all workflows to inactive
        session.exec(
            update(Workflow)
            .where(Workflow.blueprint_id == self.id)
            .values(active=False)
        )
        # update the workflow to active
        session.exec(
            update(Workflow)
            .where(Workflow.blueprint_id == self.id)
            .where(Workflow.id == id)
            .values(active=True)
        )
        session.commit()


class Workflow(SQLModel, table=True):
    __tablename__ = "workflow"
    __table_args__ = {
        "extend_existing": True,
        # "UniqueConstraint": UniqueConstraint(
        #     "name", "blueprint_id", name="unique_workflow_name_per_blueprint"
        # ),
    }

    id: int = Field(primary_key=True, sa_column_kwargs={"autoincrement": True})
    name: str = Field(index=True)
    title: str = Field(nullable=False)
    description: str = Field(nullable=False)
    active: bool = Field(default=False)
    blueprint_id: int = Field(foreign_key="blueprint.id")
    blueprint: BluePrint = Relationship(back_populates="workflows")

    depending_workflow_ids: List[int] = Field(sa_column=Column(JSON), default=[])

    result: str = Field(
        default="",
        description="The result of the workflow execution",
        sa_column=Column(Text),
    )

    cells: List["Cell"] = Relationship(
        back_populates="workflow", sa_relationship_kwargs={"order_by": "Cell.sequence"}
    )

    @classmethod
    def find_by_id(cls, session: Session, id: int):
        return session.exec(select(cls).where(cls.id == id)).first()

    def depending_workflows(self, session: Session):
        if not self.depending_workflow_ids:
            return []
        return session.exec(
            select(Workflow).where(Workflow.id.in_(self.depending_workflow_ids))
        ).all()

    def activate_cell(self, session: Session, cell_id: int):
        # update all cells to inactive
        session.exec(
            update(Cell).where(Cell.workflow_id == self.id).values(active=False)
        )
        # update the cell to active
        session.exec(
            update(Cell)
            .where(Cell.workflow_id == self.id)
            .where(Cell.id == cell_id)
            .values(active=True)
        )
        session.commit()

    def active_cell(self, session: Session):
        return session.exec(
            select(Cell).where(Cell.workflow_id == self.id).where(Cell.active == True)
        ).first()

    def find_cell_by_name(self, session: Session, cell_name: str):
        return session.exec(
            select(Cell)
            .where(Cell.workflow_id == self.id)
            .where(Cell.name == cell_name)
        ).first()

    def find_cell_by_id(self, session: Session, cell_id: int):
        return session.exec(
            select(Cell).where(Cell.workflow_id == self.id).where(Cell.id == cell_id)
        ).first()

    def find_previous_cells(self, session: Session, cell: "Cell"):
        return session.exec(
            select(Cell)
            .where(Cell.workflow_id == self.id)
            .where(Cell.sequence < cell.sequence)
            .order_by(Cell.sequence)
        ).all()


class CellStateEnum(str, enum.Enum):
    INITIAL = "initial"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class CellType(str, enum.Enum):
    UNDERSTANDING_ASK_QUESTIONS = "understanding_ask_questions"
    UNDERSTANDING_GATHER_INFO = "understanding_gather_info"
    UNDERSTANDING_GENERATE_REPORT = "understanding_generate_report"
    UNDERSTANDING_REPORT_BREAKDOWN = "understanding_report_breakdown"
    UNDERSTANDING_SOLUTION = "understanding_solution"

    PLANNING_OPTIMAL_SOLUTION = "planning_optimal_solution"
    PLANNING_KNOWLEDGE_RETRIEVAL = "planning_knowledge_retrieval"
    PLANNING_TASK_PLAN = "planning_task_plan"

    REASONING_THOUGHTS = "reasoning_thoughts"
    REASONING_OBSERVATION = "reasoning_observation"
    REASONING_ANSWER = "reasoning_answer"

    SIMPLE_RESULT = "simple_result"


class CreatedByType(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Cell(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: int = Field(primary_key=True, sa_column_kwargs={"autoincrement": True})
    input: str = Field(default="")
    output: bytes = Field(sa_column=Column(LargeBinary))
    lang: CellLangEnum = Field(
        sa_column=Column(
            Enum(CellLangEnum),
            default=CellLangEnum.TEXT_INSTRUCTION,
            nullable=True,
            index=False,
        )
    )
    thinking_system: ThinkingSystemEnum = Field(default=ThinkingSystemEnum.REASONING)
    sequence: int = Field(default=0)
    execution_sequence: int = Field(default=0)
    active: bool = Field(default=False)
    state: CellStateEnum = Field(default=CellStateEnum.INITIAL)
    workflow_id: int = Field(foreign_key="workflow.id")
    workflow: Workflow = Relationship(back_populates="cells")

    internal_workflow_id: int = Field(default=0)
    internal_workflow_step_id: int = Field(default=0)

    hidden: bool = Field(default=False)

    cell_type: CellType | None = Field(default=None)
    created_by: CreatedByType = Field(default=CreatedByType.USER)

    confirmations: List["ExecutionConfirmation"] = Relationship(
        back_populates="cell",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    parent_cell_ids: List[int] = Field(sa_column=Column(JSON), default=[])

    def parent_cells(self, session: Session):
        if not self.parent_cell_ids:
            return []
        return session.exec(select(Cell).where(Cell.id.in_(self.parent_cell_ids))).all()

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def find_by_id(cls, session: Session, id: int):
        return session.exec(select(cls).where(cls.id == id)).first()

    @classmethod
    def delete_cell(cls, session: Session, id: int, children_only: bool = False):
        """
        Delete cells recursively based on the parent cell ids
        """
        cell = cls.find_by_id(session, id)

        if cell is None:
            return []

        workflow_id = cell.workflow_id

        deleted_cell_ids = []
        # delete the cell
        if not children_only:
            session.delete(cell)
            deleted_cell_ids.append(cell.id)

        workflow = Workflow.find_by_id(session, workflow_id)
        for other_cell in workflow.cells:
            if cell.id in other_cell.parent_cell_ids:
                cell_ids = cls.delete_cell(session, other_cell.id)
                deleted_cell_ids.extend(cell_ids)

        return deleted_cell_ids


class EnvVar(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: int = Field(primary_key=True, sa_column_kwargs={"autoincrement": True})
    key: str = Field(unique=True, index=True)
    value: str = Field(sa_column=Column(Text))
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())

    @classmethod
    def all(cls, session: Session) -> Dict[str, str]:
        envvars = session.exec(select(cls)).all()
        return {envvar.key: envvar.value for envvar in envvars}

    @classmethod
    def get(cls, session: Session, key: str) -> str:
        envvar = session.exec(select(cls).where(cls.key == key)).first()
        return envvar.value if envvar else ""

    @classmethod
    def find_by_id(cls, session: Session, id: int):
        return session.exec(select(cls).where(cls.id == id)).first()


class ExecutionConfirmation(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: int = Field(primary_key=True, sa_column_kwargs={"autoincrement": True})
    cell_id: int = Field(foreign_key="cell.id")
    cell: Cell = Relationship(back_populates="confirmations")
    command: str = Field(default="")
    confirmed: bool = Field(default=False)
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())

    @classmethod
    def find_by_id(cls, session: Session, id: int):
        return session.exec(select(cls).where(cls.id == id)).first()


class KVStore(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: int = Field(primary_key=True, sa_column_kwargs={"autoincrement": True})
    key: str = Field(unique=True, index=True)
    value: JSON = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())

    class Config:
        arbitrary_types_allowed = True


def default_new_cell(workflow: Workflow):
    if workflow.blueprint.name == "polya":
        thinking_system = ThinkingSystemEnum.TYPE2
    else:
        thinking_system = ThinkingSystemEnum.REASONING

    return Cell(
        input="",
        active=True,
        workflow_id=workflow.id,
        thinking_system=thinking_system,
    )


def gen_k8s_react(config: Config):
    contexts = [config.system_prompt] if config.system_prompt != "" else [k8s_ctx]

    @react(
        model=config.model,
        contexts=contexts,
        tools=config.opsmate_tools(),
        iterable=True,
    )
    async def k8s_react(question: str, chat_history: List[Message] = []):
        return question

    return k8s_react


def gen_k8s_simple(config: Config):
    system_prompt = (
        config.system_prompt if config.system_prompt != "" else k8s_ctx.system_prompt
    )

    @dino(
        model=config.model,
        response_model=Observation,
        tools=config.opsmate_tools(),
    )
    def instruction(question: str, chat_history: List[Message] = []):
        f"""
        {system_prompt}
        """
        return [
            *chat_history,
            Message.user(
                f"Please answer the question:\n<question>{question}</question>"
            ),
        ]

    return instruction


def normalize_output_format(
    output: list | str | int | float | dict | BaseModel | ToolCall,
):
    match output:
        case ToolCall():
            return output.prompt_display()
        case BaseModel():
            return output.model_dump()
        case str() | int() | float():
            return output
        case dict():
            for k, v in output.items():
                output[k] = normalize_output_format(v)
            return output
        case list():
            return [normalize_output_format(item) for item in output]


def conversation_context(cell: Cell, session: Session):
    workflow = cell.workflow
    previous_cells = workflow.find_previous_cells(session, cell)

    for idx, previous_cell in enumerate(previous_cells):
        assistant_response = ""
        if previous_cell.output is None:
            continue
        assistant_resp = []

        for output in pickle.loads(previous_cell.output):
            o = output["output"]
            marshalled_output = normalize_output_format(o)

            try:
                if isinstance(marshalled_output, dict) or isinstance(
                    marshalled_output, list
                ):
                    assistant_resp.append(yaml.dump(marshalled_output, indent=2))
                else:
                    assistant_resp.append(marshalled_output)
            except Exception as e:
                logger.error("Error marshalling output", error=e)
        assistant_resp = "---\n".join(assistant_resp)

        conversation = f"""
Conversation {idx + 1}:

<user instruction>
{previous_cell.input}
</user instruction>

<assistant response>
{assistant_resp}
</assistant response>
"""
        yield conversation
