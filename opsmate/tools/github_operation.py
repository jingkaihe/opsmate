from opsmate.dino.types import ToolCall, PresentationMixin
from pydantic import Field
from typing import ClassVar, Optional, Any
import os
import asyncio
import structlog
from pydantic import BaseModel
import httpx

logger = structlog.get_logger(__name__)


class Result(BaseModel):
    output: Optional[str] = Field(
        description="The output of the tool call",
        default=None,
    )
    error: Optional[str] = Field(
        description="The error of the tool call",
        default=None,
    )


class GithubCloneAndCD(ToolCall[Result], PresentationMixin):
    """
    Clone a github repository and cd into the directory
    """

    # make this configurable in the future
    working_dir: ClassVar[str] = os.path.join(
        os.getenv("HOME"), ".opsmate", "github_repo"
    )

    repo: str = Field(
        ..., description="The github repository in the format of owner/repo"
    )

    @property
    def clone_url(self) -> str:
        return f"https://{self._github_token}@{self._github_domain}/{self.repo}.git"

    @property
    def repo_path(self) -> str:
        return os.path.join(self.working_dir, self.repo.split("/")[-1])

    async def __call__(self, context: dict[str, Any] = {}):
        self._github_token = context.get("github_token", os.getenv("GITHUB_TOKEN"))
        self._github_domain = context.get("github_domain", "github.com")

        logger.info("cloning repository", repo=self.repo, domain=self._github_domain)

        try:
            os.makedirs(self.working_dir, exist_ok=True)
            process = await asyncio.create_subprocess_shell(
                f"git clone {self.clone_url} {self.repo_path}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=60.0)

            # check the exit code
            if process.returncode != 0:
                raise Exception(f"Failed to clone repository: {stdout.decode()}")

            logger.info("changing directory", path=self.repo_path)
            os.chdir(self.repo_path)

            return Result(output=stdout.decode())
        except asyncio.TimeoutError:
            return Result(error="Failed to clone repository due to timeout")
        except Exception as e:
            return Result(error=f"Failed to clone repository: {e}")

    def markdown(self, context: dict[str, Any] = {}):
        if self.output.error:
            return f"Failed to clone repository: {self.output.error}"
        else:
            return f"""
## Repo clone success

Repo name: `{self.repo}`

Repo path: `{self.repo_path}`
"""


class GithubRaisePR(ToolCall[Result], PresentationMixin):
    """
    Raise a PR for a given github repository
    """

    repo: str = Field(..., description="The repository in the format of owner/repo")
    branch: str = Field(..., description="The branch to raise the PR")
    base_branch: str = Field("main", description="The base branch to raise the PR")
    title: str = Field(..., description="The title of the PR")
    body: str = Field(..., description="The body of the PR")

    @property
    def headers(self):
        return {
            "Authorization": f"Bearer {self._github_token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "opsmate / 0.1.0 (https://github.com/jingkaihe/opsmate)",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def __call__(self, context: dict[str, Any] = {}):
        self._github_token = context.get("github_token", os.getenv("GITHUB_TOKEN"))
        self._github_api_url = context.get("github_api_url", "https://api.github.com")

        logger.info(
            "raising PR",
            title=self.title,
            repo=self.repo,
            body=self.body,
            head=self.branch,
        )
        url = f"{self._github_api_url}/repos/{self.repo}/pulls"
        response = await httpx.AsyncClient().post(
            url,
            headers=self.headers,
            json={
                "title": self.title,
                "body": self.body,
                "head": self.branch,
                "base": self.base_branch,
            },
        )

        if response.status_code != 201:
            return Result(error=response.text)

        return Result(output="PR raised successfully")

    def markdown(self, context: dict[str, Any] = {}):
        if self.output.error:
            return f"Failed to raise PR: {self.output.error}"
        else:
            return "PR raised successfully"
