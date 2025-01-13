from opsmate.dino.types import ToolCall, PresentationMixin
import httpx
from typing import Dict, List
from pydantic import Field
import json
import html2text
import os
import shutil


class HttpBase(ToolCall, PresentationMixin):
    """Base class for HTTP tools"""

    url: str = Field(description="The URL to interact with")
    output: str = Field(description="The response from the URL")
    _client: httpx.AsyncClient = None

    def aconn(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient()
        return self._client


class HttpGet(HttpBase):
    """HttpGet tool allows you to get the content of a URL"""

    async def __call__(self):
        resp = await self.aconn().get(self.url)
        return resp.text

    def markdown(self):
        return f"""
### HTTP GET

```bash
{self.url}
```

### Output

```
{self.output}
```
"""


class HttpCall(HttpBase):
    """
    HttpCall tool allows you to call a URL
    Supports POST, PUT, DELETE, PATCH
    """

    data: str = Field(description="The data to post")
    method: str = Field(
        description="The HTTP method to use",
        default="POST",
        choices=["POST", "PUT", "DELETE", "PATCH"],
    )
    content_type: str = Field(
        description="The content type to send",
        default="application/json",
    )
    headers: Dict[str, str] = Field(
        description="The headers to send",
        default={
            "Content-Type": "application/json",
        },
    )

    async def __call__(self):
        if self.content_type == "application/json":
            data = json.loads(self.data)
        else:
            data = self.data
        resp = await self.aconn().request(
            self.method, self.url, json=data, headers=self.headers
        )
        return resp.text

    def markdown(self):
        return f"""
### HTTP {self.method}

```bash
{self.url}
```

### Output

```
{self.output}
```
"""


class HttpToText(HttpBase):
    """HttpToText tool allows you to convert an HTTP response to text"""

    async def __call__(self):
        resp = await self.aconn().get(self.url)
        return html2text.html2text(resp.text)

    def markdown(self):
        return f"""
### HTTP GET

```bash
{self.url}
```

### Output

```
{self.output}
```
"""


class FileRead(ToolCall, PresentationMixin):
    """FileRead tool allows you to read a file"""

    path: str = Field(description="The path to the file to read")

    async def __call__(self):
        with open(self.path, "r") as f:
            return f.read()

    def markdown(self):
        return f"""
### File Read

```bash
{self.path}
```

### Output

```
{self.output}
```
"""


class FileWrite(ToolCall, PresentationMixin):
    """FileWrite tool allows you to write to a file"""

    path: str = Field(description="The path to the file to write")
    data: str = Field(description="The data to write to the file")

    async def __call__(self):
        with open(self.path, "w") as f:
            f.write(self.data)

    def markdown(self):
        return f"""
### File Written

```bash
{self.path}
```

### Data Written

```
{self.data}
```
"""


class FileAppend(ToolCall, PresentationMixin):
    """FileAppend tool allows you to append to a file"""

    path: str = Field(description="The path to the file to append")
    data: str = Field(description="The data to append to the file")

    async def __call__(self):
        with open(self.path, "a") as f:
            f.write(self.data)

    def markdown(self):
        return f"""
### File Appended

```bash
{self.path}
```

### Data Appended

```
{self.data}
```
"""


class ListFiles(ToolCall, PresentationMixin):
    """ListFiles tool allows you to list files in a directory recursively"""

    path: str = Field(description="The path to the directory to list")
    recursive: bool = Field(
        description="Whether to list files recursively", default=True
    )

    async def __call__(self):
        if not self.recursive:
            return os.listdir(self.path)

        file_list: List[str] = []
        for root, _, files in os.walk(self.path):
            rel_path = os.path.relpath(root, self.path)
            if rel_path == ".":
                file_list.extend(files)
            else:
                file_list.extend(os.path.join(rel_path, f) for f in files)
        return "\n".join(file_list)

    def markdown(self):
        return f"""
### List Files

```bash
{self.path}
```

### Files Found
```
{self.output}
```
"""


class FindFiles(ToolCall, PresentationMixin):
    """FindFiles tool allows you to find files in a directory"""

    path: str = Field(description="The path to the directory to search")
    filename: str = Field(description="The filename to search for")

    async def __call__(self):
        found: List[str] = []
        for root, _, files in os.walk(self.path):
            if self.filename in files:
                found.append(os.path.join(root, self.filename))
        return "\n".join(found)

    def markdown(self):
        return f"""
### Find File

```bash
{self.path}
```

### Files Found
```
{self.output}
```
"""


class FileDelete(ToolCall, PresentationMixin):
    """FileDelete tool allows you to delete a file"""

    path: str = Field(description="The path to the file to delete")
    recursive: bool = Field(
        description="Whether to delete the file recursively", default=False
    )

    async def __call__(self):
        if self.recursive:
            shutil.rmtree(self.path)
        else:
            os.remove(self.path)

    def markdown(self):
        return f"""
### File Deleted

```bash
{self.path}
```
"""


class SysStats(ToolCall, PresentationMixin):
    """SysStats tool allows you to get the stats of a file"""

    path: str = Field(description="The path to the file to get stats")

    async def __call__(self):
        stats = os.stat(self.path)
        return str(stats)

    def markdown(self):
        return f"""
### File Stats

```bash
{self.path}
```

### Stats
```
{self.output}
```
"""


class SysEnv(ToolCall, PresentationMixin):
    """SysEnv tool allows you to get the environment variables"""

    env_vars: List[str] = Field(
        description="The environment variables to get",
        default=[],
    )

    async def __call__(self):
        outputs = []
        for var in self.env_vars:
            outputs.append(f"{var}: {os.environ.get(var, 'Not found')}")
        return "\n".join(outputs)

    def markdown(self):
        return f"""
### Env

```bash
{self.env_vars}
```

### Output
```
{self.output}
```
"""
