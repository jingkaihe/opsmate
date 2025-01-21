from opsmate.tools.aci import ACITool
from opsmate.tools import ShellCommand
from opsmate.dino.react import react
import asyncio


@react(
    model="gpt-4o",
    tools=[ACITool, ShellCommand],
    contexts=["you are an SRE who is tasked to modify the infra as code"],
    iterable=True,
)
async def iac_editor(instruction: str):
    """
    You are an SRE who is tasked to modify the infra as code.

    <rule 1>
    Before making any changes, you must read the file(s) to understand:
    * the purpose of the file (e.g. a terraform file deploying IaC, or a yaml file deploying k8s resources)
    * Have a basic understanding of the file's structure
    </rule 1>

    <rule 2>
    Edit must be precise and specific:
    * Tabs and spaces must be used correctly
    * The line range must be specified when you are performing an update operation against a file
    * Stick to the task you are given, don't make drive-by changes
    </rule 2>

    <rule 3>
    After you make the change, you must verify the updated content is correct using the `ACITool.view` or `ACITool.search` commands.
    </rule 3>

    <rule 4>
    Never ever use vim or any other text editor to make changes, instead use the `ACITool` tool.
    </rule 4>
    """
    return instruction


async def main():
    async for result in await iac_editor(
        "change the k8s health check path to `/status` in `./hack`"
    ):
        print(result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
