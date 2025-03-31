from typing import ClassVar, Any
from pydantic import Field
from opsmate.dino.types import ToolCall, PresentationMixin
import structlog
import asyncio
import os
import inspect
from opsmate.tools.utils import maybe_truncate_text
from opsmate.runtime import LocalRuntime
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)
logger = structlog.get_logger(__name__)


class ShellCommand(ToolCall[str], PresentationMixin):
    """
    ShellCommand tool allows you to run shell commands and get the output.
    """

    description: str = Field(description="Explain what the command is doing")
    command: str = Field(description="The command to run")
    timeout: float = Field(
        description="The estimated time for the command to execute in seconds",
        default=120.0,
    )

    async def __call__(self, context: dict[str, Any] = {}):
        with tracer.start_as_current_span("shell_command") as span:
            envvars = os.environ.copy()
            extra_envvars = context.get("envvars", {})
            max_output_length = context.get("max_output_length", 10000)
            envvars.update(extra_envvars)
            logger.info("running shell command", command=self.command)

            runtime = context.get("runtime", None)
            transit_runtime = True if runtime is None else False

            span.set_attributes(
                {
                    "runtime": runtime.__class__.__name__,
                    "transit_runtime": transit_runtime,
                    "command": self.command,
                    "description": self.description,
                }
            )

            if not await self.confirmation_prompt(context):
                return "Command execution cancelled by user, try something else."

            if runtime is None:
                runtime = LocalRuntime(envvars=envvars)
                await runtime.connect()

            try:
                out = await runtime.run(self.command, timeout=self.timeout)

                span.set_attributes({"output": out})
                return maybe_truncate_text(out, max_output_length)
            except Exception as e:
                err_msg = str(e)
                span.set_attributes({"error": err_msg})
                span.set_status(Status(StatusCode.ERROR))
                return err_msg
            finally:
                if transit_runtime:
                    await runtime.disconnect()

    async def confirmation_prompt(self, context: dict[str, Any] = {}):
        confirmation = context.get("confirmation", None)
        if confirmation is None:
            return True

        if inspect.iscoroutinefunction(confirmation):
            return await confirmation(self)
        else:
            return confirmation(self)

    def markdown(self, context: dict[str, Any] = {}):
        return f"""
### Command

```bash
# {self.description}
{self.command}
```

### Output

```bash
{self.output}
```
"""
