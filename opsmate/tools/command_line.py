from typing import ClassVar, Any
from pydantic import Field
from opsmate.dino.types import ToolCall, PresentationMixin
import structlog
import asyncio
import os
import inspect
from opsmate.tools.utils import maybe_truncate_text

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
        envvars = os.environ.copy()
        extra_envvars = context.get("envvars", {})
        max_output_length = context.get("max_output_length", 10000)
        envvars.update(extra_envvars)
        logger.info("running shell command", command=self.command)

        if not await self.confirmation_prompt(context):
            return "Command execution cancelled by user, try something else."

        try:
            process = await asyncio.create_subprocess_shell(
                self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=envvars,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
            return maybe_truncate_text(stdout.decode(), max_output_length)
        except Exception as e:
            return str(e)

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
