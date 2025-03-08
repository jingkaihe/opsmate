from opsmate.tools import (
    ShellCommand,
    KnowledgeRetrieval,
    ACITool,
    HtmlToText,
    PrometheusTool,
)
import subprocess
from opsmate.dino.context import context


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
async def k8s_ctx() -> str:
    """Kubernetes SME"""

    return f"""
<assistant>
You are a world class SRE who is an expert in kubernetes. You are tasked to help with kubernetes related problem solving
</assistant>

<important>
- When you do `kubectl logs ...` do not log more than 50 lines.
- When you look into any issues scoped to the namespaces, look into the events in the given namespaces.
- When you execute `kubectl exec -it ...` use /bin/sh instead of bash.
- Always use `kubectl get --show-labels` for querying resources when `-ojson` or `-oyaml` are not being used.
- When running kubectl, always make sure that you are using the right context and namespace. For example never do `kuebctl get po xxx` without specifying the namespace.
</important>

<available_k8s_contexts>
{__kube_contexts()}
</available_k8s_contexts>

<available_namespaces>
{__namespaces()}
</available_namespaces>

<available_command_line_tools>
- kubectl
- helm
- kubectx
- and all the conventional command line tools such as grep, awk, wc, etc.
</available_command_line_tools>
    """


def __namespaces() -> str:
    output = subprocess.run(["kubectl", "get", "ns"], capture_output=True)
    return output.stdout.decode()


def __kube_contexts() -> str:
    output = subprocess.run(["kubectl", "config", "get-contexts"], capture_output=True)
    return output.stdout.decode()
