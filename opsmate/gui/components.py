from fasthtml.common import *
from opsmate.gui.models import (
    Cell,
    CellLangEnum,
    CreatedByType,
    ThinkingSystemEnum,
    CellStateEnum,
    CellType,
    WorkflowEnum,
)
from opsmate.gui.assets import *
import pickle
from opsmate.dino.types import React, ReactAnswer, Observation
from opsmate.polya.models import (
    InitialUnderstandingResponse,
    InfoGathered,
    NonTechnicalQuery,
    TaskPlan,
    Facts,
)
from jinja2 import Template
import plotly.express as px
import pandas as pd
from opsmate.tools.prom import PrometheusTool
from datetime import datetime
import yaml
import structlog

logger = structlog.get_logger()


class CellComponent:
    def __init__(self, cell: Cell, hx_swap_oob=None):
        self.cell = cell
        self.cell_size = len(self.cell.workflow.cells)
        self.blueprint = self.cell.workflow.blueprint
        self.hx_swap_oob = hx_swap_oob

    def __ft__(self):
        """Renders a single cell component"""
        # Determine if the cell is active
        active_class = "border-green-500" if self.cell.active else "border-gray-300"

        div = Div(
            # Add Cell Button Menu
            self.cell_insert_dropdown(),
            # Main Cell Content
            Div(
                # Cell Header
                self.cell_header(),
                # Cell Input - Updated with conditional styling
                self.cell_input_form(),
                # Cell Output (if any)
                self.cell_output(),
                cls=f"rounded-lg shadow-sm border {active_class}",  # Apply the active class here
            ),
            cls="group relative",
            key=self.cell.id,
            id=f"cell-component-{self.cell.id}",
        )
        if self.hx_swap_oob:
            div.hx_swap_oob = self.hx_swap_oob
        return div

    def cell_insert_dropdown(self):
        return (
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
                                hx_post=f"/blueprint/{self.blueprint.id}/cell/{self.cell.id}?above=true",
                            )
                        ),
                        Li(
                            Button(
                                "Insert Below",
                                hx_post=f"/blueprint/{self.blueprint.id}/cell/{self.cell.id}?above=false",
                            )
                        ),
                        tabindex="0",
                        cls="dropdown-content z-10 menu p-2 shadow bg-base-100 rounded-box",
                    ),
                    cls="dropdown dropdown-left",
                ),
                cls="absolute -left-8 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity",
            ),
        )

    def can_edit(self):
        cell_type = self.cell.cell_type
        if cell_type == CellType.REASONING_OBSERVATION:
            return False
        return True

    def can_run(self):
        cell_type = self.cell.cell_type
        if cell_type in [
            CellType.SIMPLE_RESULT,
            CellType.REASONING_OBSERVATION,
        ]:
            return False
        return True

    def can_delete(self):
        return self.cell_size > 1

    def can_edit_thinking_system(self):
        return self.cell.created_by != CreatedByType.ASSISTANT

    def can_edit_lang(self):
        return self.cell.created_by != CreatedByType.ASSISTANT

    def can_stop(self):
        return self.cell.state == CellStateEnum.RUNNING

    def cell_header(self):
        return (
            Div(
                Div(
                    Span(
                        f"In [{self.cell.execution_sequence}]:",
                        cls="text-gray-500 text-sm",
                    ),
                    # Add cell type selector
                    cls="flex items-center gap-2",
                ),
                Div(
                    Select(
                        Option(
                            "Text Instruction",
                            value=CellLangEnum.TEXT_INSTRUCTION.value,
                            selected=self.cell.lang == CellLangEnum.TEXT_INSTRUCTION,
                        ),
                        Option(
                            "Bash",
                            value=CellLangEnum.BASH.value,
                            selected=self.cell.lang == CellLangEnum.BASH,
                        ),
                        Option(
                            "Notes",
                            value=CellLangEnum.NOTES.value,
                            selected=self.cell.lang == CellLangEnum.NOTES,
                        ),
                        name="lang",
                        hx_put=f"/blueprint/{self.blueprint.id}/cell/{self.cell.id}",
                        hx_trigger="change",
                        disabled=not self.can_edit_lang(),
                        cls="select select-sm ml-2",
                    ),
                    Select(
                        Option(
                            "Simple",
                            value=ThinkingSystemEnum.SIMPLE.value,
                            selected=self.cell.thinking_system
                            == ThinkingSystemEnum.SIMPLE,
                        ),
                        Option(
                            "Reasoning",
                            value=ThinkingSystemEnum.REASONING.value,
                            selected=self.cell.thinking_system
                            == ThinkingSystemEnum.REASONING
                            or self.cell.lang == CellLangEnum.BASH,
                        ),
                        Option(
                            "Type 2 - Slow but thorough",
                            value=ThinkingSystemEnum.TYPE2.value,
                            selected=self.cell.thinking_system
                            == ThinkingSystemEnum.TYPE2,
                            disabled=self.cell.workflow.name == WorkflowEnum.FREESTYLE,
                        ),
                        name="thinking_system",
                        hx_put=f"/blueprint/{self.blueprint.id}/cell/{self.cell.id}",
                        hx_trigger="change",
                        cls="select select-sm ml-2 min-w-[240px]",
                        hidden=self.cell.lang != CellLangEnum.TEXT_INSTRUCTION,
                        disabled=not self.can_edit_thinking_system(),
                    ),
                    Button(
                        trash_icon_svg,
                        hx_delete=f"/blueprint/{self.blueprint.id}/cell/{self.cell.id}",
                        cls="btn btn-ghost btn-sm opacity-0 group-hover:opacity-100 hover:text-red-500",
                        disabled=not self.can_delete(),
                    ),
                    Button(
                        edit_icon_svg,
                        Input(type="hidden", value="false", name="hidden"),
                        hx_put=f"/blueprint/{self.blueprint.id}/cell/{self.cell.id}",
                        cls="btn btn-ghost btn-sm",
                        disabled=not self.can_edit(),
                    ),
                    Button(
                        stop_icon_svg,
                        hx_put=f"/blueprint/{self.blueprint.id}/cell/{self.cell.id}/stop",
                        cls="btn btn-ghost btn-sm",
                        disabled=not self.can_stop(),
                    ),
                    Form(
                        Input(type="hidden", value=self.cell.id, name="cell_id"),
                        Button(
                            run_icon_svg,
                            cls="btn btn-ghost btn-sm",
                            disabled=not self.can_run(),
                        ),
                        ws_send=True,
                        hx_ext="ws",
                    ),
                    cls="ml-auto flex items-center gap-2",
                ),
                id=f"cell-header-{self.cell.id}",
                cls="flex items-center px-4 py-2 bg-gray-100 border-b justify-between rounded-t-lg overflow-hidden",
            ),
        )

    def cell_input_form(self):
        return (
            Div(
                Form(
                    self.cell_text_area(),
                    Div(
                        hx_put=f"/blueprint/{self.blueprint.id}/cell/input/{self.cell.id}",
                        hx_trigger=f"keyup[!(shiftKey&&keyCode===13)] changed delay:500ms from:#cell-input-{self.cell.id}",
                        hx_swap=f"#cell-input-form-{self.cell.id}",
                    ),
                    # xxx: shift+enter is being registered as a newline
                    Div(
                        Input(type="hidden", value=self.cell.id, name="cell_id"),
                        ws_send=True,
                        hx_ext="ws",
                        hx_trigger=f"keydown[shiftKey&&keyCode===13] from:#cell-input-{self.cell.id}",
                        hx_swap=f"#cell-input-form-{self.cell.id}",
                    ),
                    id=f"cell-input-form-{self.cell.id}",
                ),
                hx_include="input",
                cls="p-4",
            ),
        )

    def cell_text_area(self):
        return Textarea(
            self.cell.input,
            name="input",
            cls=f"w-full h-24 p-2 font-mono text-sm border rounded focus:outline-none focus:border-blue-500",
            placeholder="Enter your instruction here...",
            id=f"cell-input-{self.cell.id}",
            hidden=self.cell.hidden,
        )

    def cell_output(self):
        if self.cell.output:
            outputs = pickle.loads(self.cell.output)
            outputs = [CellOutputRenderer(output).render() for output in outputs]
        else:
            outputs = []
        return Div(
            Span(f"Out [{self.cell.execution_sequence}]:", cls="text-gray-500 text-sm"),
            Div(
                *outputs,
                id=f"cell-output-{self.cell.id}",
            ),
            cls="px-4 py-2 bg-gray-50 border-t rounded-b-lg overflow-hidden",
        )


def render_react_markdown_raw(output: React):
    return f"""
## Thought process

### Thoughts

{output.thoughts}

### Action

{output.action}
"""


def render_react_markdown(output: React):
    return Div(
        render_react_markdown_raw(output),
        cls="marked prose max-w-none",
    )


def render_react_answer_markdown_raw(output: ReactAnswer):
    return f"""
## Answer

{output.answer}
"""


def render_react_answer_markdown(output: ReactAnswer):
    return Div(
        render_react_answer_markdown_raw(output),
        cls="marked prose max-w-none",
    )


def generate_chart(tool_output: PrometheusTool):
    query = tool_output.output

    if "error" in query.output or query.dataframe is None:
        if "error" in query.output:
            error = query.output["error"]
        else:
            error = "No data to display"
        logger.error("Error in query", error=error)
        # generate a diagram of the error
        fig = px.bar(
            x=["Error"],
            y=[1],
            template="plotly_white",
            title=error,
        )
        return fig.to_html(
            include_plotlyjs=True, full_html=False, config={"displayModeBar": False}
        )

    df = query.dataframe

    y_columns = [col for col in df.columns if col != "timestamp"]
    fig = px.line(
        df, x="timestamp", y=y_columns, template="plotly_white", title=query.title
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig.to_html(
        include_plotlyjs=True, full_html=False, config={"displayModeBar": False}
    )


def render_observation_markdown_raw(output: Observation):
    tool_out = []
    for tool_output in output.tool_outputs:
        if hasattr(tool_output, "markdown"):
            tool_out.append(tool_output.markdown())
        else:
            tool_out.append(yaml.dump(tool_output.model_dump()))
        # if hasattr(tool_output, "time_series"):
        #     image_data = tool_output.time_series(show_base64_image=True)
        #     if image_data:
        #         tool_out.append(
        #             f"![{image_data['title']}](data:{image_data['mime_type']};base64,{image_data['data']})"
        #         )
    return f"""
## Observation

{output.observation}

{"\n".join(tool_out)}
"""


def render_observation_markdown(output: Observation):
    return (
        Div(
            render_observation_markdown_raw(output),
            cls="marked prose max-w-none",
        ),
        *[
            Div(
                Safe(generate_chart(tool_output)),
            )
            for tool_output in output.tool_outputs
            if isinstance(tool_output, PrometheusTool)
        ],
    )


def render_bash_output_markdown(output: str):
    return Div(
        f"""
## Results

```bash
{output}
```
""",
        cls="marked prose max-w-none",
    )


def render_task_plan_markdown(task_plan: TaskPlan):
    return Div(
        f"""
## Task plan

### Goal

{task_plan.goal}

### Subtasks

{"\n".join([f"* {subtask.task}" for subtask in task_plan.subtasks])}
""",
        cls="marked prose max-w-none",
    )


class UnderstandingRenderer:
    @staticmethod
    def render_initial_understanding_markdown(iu: InitialUnderstandingResponse):
        return Div(
            f"""
## Initial understanding

{iu.summary}

{ "**Questions**" if iu.questions else "" }
{"\n".join([f"{i+1}. {question}" for i, question in enumerate(iu.questions)])}
""",
            cls="marked prose max-w-none",
        )

    @staticmethod
    def render_info_gathered_markdown(info_gathered: InfoGathered):
        template = """
## Information Gathering

### Question

{{ info_gathered.question }}

### Command

{% for command in info_gathered.commands %}
```bash
# {{ command.description }}
{{ command.command }}
```

#### Output

```bash
{{ command.result }}
```
{% endfor %}

### Summary

{{ info_gathered.info_gathered }}

"""

        return Div(
            Template(template).render(info_gathered=info_gathered),
            cls="marked prose max-w-none",
        )

    @staticmethod
    def render_potential_solution_markdown(output: dict):
        summary = output.get("summary", "")
        solution = output.get("solution", {})

        rendered = solution.summarize(summary)
        return Div(
            f"""
{rendered}
<br>
""",
            cls="marked prose max-w-none",
        )

    @staticmethod
    def render_non_technical_query_markdown(non_technical_query: NonTechnicalQuery):
        return Div(
            f"""
This is a non-technical query, thus I don't know how to answer it.

**Reason:** {non_technical_query.reason}
""",
            cls="marked prose max-w-none",
        )


def render_notes_output_markdown(output: str):
    return Div(
        output,
        cls="marked prose max-w-none",
    )


def render_facts_markdown(output: Facts):
    tmpl = """
## Facts

{% for fact in output.facts %}
* {{ fact.fact }} ({{ fact.source }})
{% endfor %}
"""
    return Div(
        Template(tmpl).render(output=output),
        cls="marked prose max-w-none",
    )


def render_solution_for_planning_markdown(output: str):
    return Div(
        output,
        cls="marked prose max-w-none",
    )


class CellOutputRenderer:
    cell_output_render_func = {
        "React": render_react_markdown,
        "ReactAnswer": render_react_answer_markdown,
        "Observation": render_observation_markdown,
        "BashOutput": render_bash_output_markdown,
        "NotesOutput": render_notes_output_markdown,
        "InitialUnderstanding": UnderstandingRenderer.render_initial_understanding_markdown,
        "InfoGathered": UnderstandingRenderer.render_info_gathered_markdown,
        "PotentialSolution": UnderstandingRenderer.render_potential_solution_markdown,
        "NonTechnicalQuery": UnderstandingRenderer.render_non_technical_query_markdown,
        "TaskPlan": render_task_plan_markdown,
        "Facts": render_facts_markdown,
        "SolutionForPlanning": render_solution_for_planning_markdown,
    }

    def __init__(self, output: dict):
        self.output = output

    def render(self):
        fn = self.cell_output_render_func.get(self.output["type"])
        if fn:
            return fn(self.output["output"])
        else:
            return None

    @classmethod
    def render_model(cls, model: Any):
        model_cls = model.__class__.__name__
        fn = cls.cell_output_render_func.get(model_cls)
        if fn:
            return fn(model)
        else:
            return None
