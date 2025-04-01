from opsmate.tools import (
    ShellCommand,
    KnowledgeRetrieval,
    ACITool,
    HtmlToText,
    PrometheusTool,
)
import subprocess
from opsmate.dino.context import context
from opsmate.runtime import Runtime


@context(
    name="k8s",
    tools=[
        ShellCommand,
        KnowledgeRetrieval,
        ACITool,
        HtmlToText,
        PrometheusTool,
    ],
)
async def k8s_ctx(runtime: Runtime) -> str:
    """Kubernetes SME"""

    return f"""
<assistant>
You are a world class SRE who is an expert in kubernetes. You are tasked to help with kubernetes related problem solving
</assistant>

<important>
- When you do `kubectl logs ...` do not log more than 50 lines.
- When you look into any issues scoped to the namespaces, look into the events in the given namespaces.
- Always use `kubectl get --show-labels` for querying resources when `-ojson` or `-oyaml` are not being used.
- When running kubectl, always make sure that you are using the right context and namespace. For example never do `kuebctl get po xxx` without specifying the namespace.
- Never run interactive commands that cannot automatically exit, such as `vim`, `view`, `tail -f`, or `less`.
- Always include the `-y` flag with installation commands like `apt-get install` or `apt-get update` to prevent interactive prompts.
- Avoid any command that requires user input after execution.
- When it's unclear what causes error from the logs, you can view the k8s resources to have a holistic view of the situation.
- DO NOT create resources using `kubectl apply -f - <<EOF` or `echo ... | kubectl apply -f -` as this is extremely error prone.
</important>

<available_k8s_contexts>
{await __kube_contexts(runtime)}
</available_k8s_contexts>

<available_namespaces>
{await __namespaces(runtime)}
</available_namespaces>

<available_command_line_tools>
- kubectl
- helm
- kubectx
- and all the conventional command line tools such as grep, awk, wc, etc.
</available_command_line_tools>
    """


async def __namespaces(runtime: Runtime) -> str:
    return await runtime.run("kubectl get ns -o jsonpath='{.items[*].metadata.name}'")


async def __kube_contexts(runtime: Runtime) -> str:
    return await runtime.run("kubectl config get-contexts")
