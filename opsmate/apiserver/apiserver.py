from fastapi import FastAPI
from typing import List, Literal, Dict, Any
from opsmate.libs.core.types import Model, ExecResults, Task, Metadata, TaskSpec
from opsmate.libs.core.engine import exec_task
from pydantic import BaseModel, Field
from opsmate.libs.providers import Client as ProviderClient, ClientBag
from opsmate.libs.contexts import available_contexts
from opsmate.apiserver.gui import app as fasthtml_app

client_bag = ProviderClient.clients_from_env()
app = FastAPI()


class Health(BaseModel):
    status: Literal["ok", "faulty"] = Field(title="status", default="ok")


class Session(BaseModel):
    uuid: str = Field(title="uuid")


@app.get("/api/v1/health", response_model=Health)
async def health():
    return Health(status="ok")


@app.get("/api/v1/models", response_model=List[Model])
async def models():
    return ProviderClient.models_from_clientbag(client_bag)


class RunRequest(BaseModel):
    model: str = Field(title="name of the llm model to use")
    provider: str = Field(title="name of the provider to use")
    instruction: str = Field(title="instruction to execute")
    contexts: List[str] = Field(title="contexts to use", default=["cli"])
    ask: bool = Field(title="ask", default=False)


@app.post("/api/v1/run", response_model=List[Dict[str, Any]])
async def run(request: RunRequest):

    selected_contexts = get_contexts(request.contexts)

    task = Task(
        metadata=Metadata(name="run"),
        spec=TaskSpec(
            input={},
            contexts=selected_contexts,
            instruction=request.instruction,
            response_model=ExecResults,
        ),
    )

    output = exec_task(
        clients=client_bag,
        task=task,
        ask=request.ask,
        model=request.model,
        provider=request.provider,
    )

    # output = [r.model_dump() for r in output.results]
    result = []
    for r in output.results:
        result.append(
            {
                "executable": r.__class__.__name__,
                **r.model_dump(),
            }
        )
    return result


class ContextNotFound(Exception):
    pass


def get_contexts(contexts: List[str]):
    contexts = list(set(contexts))

    selected_contexts = []
    for ctx_name in contexts:
        for ctx in available_contexts:
            if ctx.metadata.name == ctx_name:
                selected_contexts.append(ctx)
                break
        else:
            raise ContextNotFound(f"Context {ctx_name} not found")

    return selected_contexts


app.mount("/", fasthtml_app)
