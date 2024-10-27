import yaml
from openai import OpenAI
import instructor
import jinja2
from eval.types import TroubleshootingQuestion, Category, pod_and_container_issues

idea_prompt_template = """
You are tasked to create some ideas for kubernetes related questionaires for SRE candidates.

Here is the topic and description of the questionaire:

<category>
  <name>
  {{name}}
  </name>
  <description>
  {{description}}
  </description>
</category>

Example:

<category>
  <name>pod scheduling</name>
  <description>
Pod Lifecycle: Examine Pod status, including Pending, CrashLoopBackOff, or Evicted.
  </description>
</category>

Ideas:
- name: pod cannot be scheduled due to anti-affinity rules
- name: pod stuck in pending state due to resource request cannot be satisfied
- name: deployment is not rolled out due to image pull issue
- name: pod cannot be scheduled due to manually pinned nodeName does not exist
...

Please come up with {{n}} distinct ideas for questions for this category.
"""


prompt_template = """
You are tasked to create kubernetes related questionaires for SRE candidates.

based on the idea for testing, you are asked to:
* Create a scenario that comes with an issue using kubernetes yaml
* Ask a question about the issue
* Provide commands to verify whether the issue has been fixed

Example:

Idea: pod cannot be deployed due to image pull issue

<example>
steps_to_create_issue:
- namespace: aa062294-571e
  manifest: |
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: nginx-deployment
      namespace: aa062294-571e
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: nginx
      template:
        metadata:
          labels:
            app: nginx
        spec:
          containers:
          - name: nginx
            image: nginc:latest
              ports:
              - containerPort: 80

question: The nginx deployment is not ready, what is the issue?
issue_produced_verification:
# no ready pods
- command: kubectl -n aa062294-571e get deploy nginx-deployment -o jsonpath='{.status.readyReplicas}'
  expected_output: 0
  exit_code: 0
fix_verification:
- command: kubectl get pods -n aa062294-571e -l app=nginx -o jsonpath='{.items[0].status.phase}' | grep -q Running
  expected_output: Running
  exit_code: 0
root_cause: The nginx deployment is not ready because the image is not found
</example>

A few things to note:
* The namespace must be specified in the kubernetes manifest
* kubectl command must have the namespace specified if applicable
* namespace ideally follows the format of <random_name>-<short_unique_id> format, and **must not** reveal the root cause of the issue
* the naming in the kubernetes manifest **must not** reveal the root cause of the issue
* the question **must not** reveal the root cause of the issue
* please do not use any placeholder image registry like `frontend:1.0`, use a real image registry instead

=======

Here is the idea for testing:

<idea>
{{idea}}
</idea>

Please create a scenario.
"""

client = instructor.from_openai(OpenAI())


def make_troubleshooting_questions(category: Category, count: int = 1):
    idea_prompt = jinja2.Template(idea_prompt_template).render(
        name=category.name, description=category.description, n=count
    )

    ideas = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": idea_prompt,
            }
        ],
        response_model=list[str],
        max_retries=5,
    )

    ideas = ideas[:count]

    for idea in ideas:
        print(f"Creating issue for idea: {idea}")
        jinja_prompt_template = jinja2.Template(prompt_template)
        prompt = jinja_prompt_template.render(idea=idea)

        yield client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                }
            ],
            response_model=TroubleshootingQuestion,
            max_retries=5,
        )


def bootstrap(count: int = 3):
    issues = [
        issue
        for category in [pod_and_container_issues]
        for issue in make_troubleshooting_questions(category=category, count=count)
    ]

    with open("./eval/issues.yaml", "w") as f:
        issues_dict = [issue.model_dump() for issue in issues]
        yaml.safe_dump(issues_dict, f, default_flow_style=False, indent=4)
