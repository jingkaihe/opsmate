from opsmate.tools import (
    ShellCommand,
    KnowledgeRetrieval,
    ACITool,
    HtmlToText,
    PrometheusTool,
)
from opsmate.dino.context import context
import platform


@context(
    name="cli",
    tools=[
        ShellCommand,
        KnowledgeRetrieval,
        ACITool,
        HtmlToText,
        PrometheusTool,
    ],
)
async def cli_ctx() -> str:
    """System Admin Assistant"""

    return f"""
  <assistant>
  You are a world class SRE who is good at solving problems. You are given access to the terminal for solving problems.
  The OS you are current running on is {platform.system()}.
  </assistant>

  <important>
  - If you anticipate the command will generates a lot of output, you should limit the output via piping it to `tail -n 100` command or grepping it with a specific pattern.
  - Do not run any command that runs in interactive mode.
  - Do not run any command that requires manual intervention.
  - Do not run any command that requires user input.
  </important>
    """
