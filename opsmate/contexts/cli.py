from opsmate.tools import (
    ShellCommand,
    KnowledgeRetrieval,
    ACITool,
    HtmlToText,
    PrometheusTool,
    Thinking,
)
from opsmate.dino.context import context
from opsmate.runtime import Runtime
from jinja2 import Template


@context(
    name="cli",
    tools=[
        ShellCommand,
        KnowledgeRetrieval,
        ACITool,
        HtmlToText,
        PrometheusTool,
        Thinking,
    ],
)
async def cli_ctx(runtimes: dict[str, Runtime] = {}) -> str:
    """System Admin Assistant"""

    # Pre-fetch all runtime information asynchronously
    runtime_info = {}
    for runtime_name, runtime in runtimes.items():
        runtime_info[runtime_name] = {
            "os_info": await runtime.os_info(),
            "whoami": await runtime.whoami(),
            "runtime_info": await runtime.runtime_info(),
            "has_systemd": await runtime.has_systemd(),
        }

    template = Template(
        """
  <assistant>
  You are a world class SRE who excells at solving problems. You are given access to the terminal for solving problems.
  </assistant>

  You have access to the following runtimes:

  <sys-info>
    {% for runtime_name, info in runtime_info.items() %}
    <runtime name="{{ runtime_name }}">
      <os-info>
      {{ info.os_info }}
      </os-info>
      <whoami>
      {{ info.whoami }}
      </whoami>
      <runtime-info>
      {{ info.runtime_info }}
      </runtime-info>
      <has-systemd>
      {{ info.has_systemd }}
      </has-systemd>
    </runtime>
    {% endfor %}
  </sys-info>

  <important>
  - If you anticipate the command will generates a lot of output, you should limit the output via piping it to `tail -n 100` command or grepping it with a specific pattern.
  - Do not run any command that runs in interactive mode.
  - Do not run any command that requires manual intervention.
  - Do not run any command that requires user input.
  </important>
    """
    )

    rendered_template = template.render(runtime_info=runtime_info)
    return rendered_template
