import subprocess
from typing import Optional
from pydantic import Field
from opsmate.dino.types import ToolCall
import structlog
import asyncio

logger = structlog.get_logger(__name__)


class ShellCommand(ToolCall):
    """
    ShellCommand tool allows you to run shell commands and get the output.
    """

    description: str = Field(description="Explain what the command is doing")
    command: str = Field(description="The command to run")
    output: Optional[str] = Field(
        description="The output of the command - DO NOT POPULATE"
    )

    async def __call__(self):
        logger.info("running shell command", command=self.command)
        try:
            process = await asyncio.create_subprocess_shell(
                self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await process.communicate()
            self.output = stdout.decode()
        except Exception as e:
            self.output = str(e)
        return self.output

    def markdown(self):
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
