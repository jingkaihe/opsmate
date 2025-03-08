from opsmate.dino.types import ToolCall, PresentationMixin
import httpx
from typing import Dict, List, Any
from pydantic import Field
import json
import html2text
import os
import shutil
from pydantic import BaseModel
import fnmatch


class HttpResponse(BaseModel):
    status_code: int
    text: str


class HttpBase(ToolCall[HttpResponse], PresentationMixin):
    """Base class for HTTP tools"""

    url: str = Field(description="The URL to interact with")
    _client: httpx.AsyncClient = None

    def aconn(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient()
        return self._client


class HttpGet(HttpBase):
    """HttpGet tool allows you to get the content of a URL"""

    async def __call__(self) -> str:
        resp = await self.aconn().get(self.url)
        return HttpResponse(status_code=resp.status_code, text=resp.text)

    def markdown(self, context: Dict[str, Any] = {}):
        return f"""
### HTTP GET

```bash
{self.url}
```

### Output

resp code: {self.output.status_code}

```
{self.output.text}
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
        return HttpResponse(status_code=resp.status_code, text=resp.text)

    def markdown(self, context: Dict[str, Any] = {}):
        return f"""
### HTTP {self.method}

```bash
{self.url}
```

### Output

resp code: {self.output.status_code}

```
{self.output.text}
```
"""


class HtmlToText(HttpBase):
    """HtmlToText tool allows you to convert an HTTP response to text"""

    async def __call__(self):
        resp = await self.aconn().get(self.url)
        return HttpResponse(
            status_code=resp.status_code, text=html2text.html2text(resp.text)
        )

    def markdown(self, context: Dict[str, Any] = {}):
        return f"""
### HTML to Text

```bash
{self.url}
```

### Output

resp code: {self.output.status_code}

```
{self.output.text}
```
"""


class Fs(ToolCall[str], PresentationMixin):
    """Fs tool allows you to read and write to the filesystem"""

    def markdown(self, context: Dict[str, Any] = {}): ...


class FileRead(Fs):
    """FileRead tool allows you to read a file"""

    path: str = Field(description="The path to the file to read")

    async def __call__(self):
        with open(self.path, "r") as f:
            return f.read()

    def markdown(self, context: Dict[str, Any] = {}):
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


class FileWrite(Fs):
    """FileWrite tool allows you to write to a file"""

    path: str = Field(description="The path to the file to write")
    data: str = Field(description="The data to write to the file")

    async def __call__(self):
        with open(self.path, "w") as f:
            f.write(self.data)

    def markdown(self, context: Dict[str, Any] = {}):
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


class FileAppend(Fs):
    """FileAppend tool allows you to append to a file"""

    path: str = Field(description="The path to the file to append")
    data: str = Field(description="The data to append to the file")

    async def __call__(self):
        with open(self.path, "a") as f:
            f.write(self.data)

    def markdown(self, context: Dict[str, Any] = {}):
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


class FilesList(Fs):
    """FilesList tool allows you to list files in a directory recursively"""

    path: str = Field(description="The path to the directory to list")
    recursive: bool = Field(
        description="Whether to list files recursively", default=True
    )

    async def __call__(self):
        if not self.recursive:
            return "\n".join(os.listdir(self.path))

        file_list: List[str] = []
        for root, _, files in os.walk(self.path):
            rel_path = os.path.relpath(root, self.path)
            if rel_path == ".":
                file_list.extend(files)
            else:
                file_list.extend(os.path.join(rel_path, f) for f in files)
        return "\n".join(file_list)

    def markdown(self, context: Dict[str, Any] = {}):
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


class FilesFind(Fs):
    """FilesFind tool allows you to find files in a directory"""

    path: str = Field(description="The path to the directory to search")
    filename: str = Field(description="The filename pattern to search for")

    async def __call__(self):
        found: List[str] = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if fnmatch.fnmatch(file, self.filename):
                    found.append(os.path.join(root, file))
        return "\n".join(found)

    def markdown(self, context: Dict[str, Any] = {}):
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


class FileDelete(Fs):
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

    def markdown(self, context: Dict[str, Any] = {}):
        return f"""
### File Deleted

```bash
{self.path}
```
"""


class SysStats(Fs):
    """SysStats tool allows you to get the stats of a file"""

    path: str = Field(description="The path to the file to get stats")

    async def __call__(self):
        stats = os.stat(self.path)
        return str(stats)

    def markdown(self, context: Dict[str, Any] = {}):
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


class SysChdir(Fs):
    """SysChdir tool allows you to change the current working directory"""

    path: str = Field(description="The path to change the current working directory to")

    _prev_dir: str = None

    async def __call__(self):
        try:
            self._prev_dir = os.getcwd()
        except Exception as e:
            self._prev_dir = None
        os.chdir(self.path)

    def markdown(self, context: Dict[str, Any] = {}):
        return f"""
### Current Directory

```bash
{self._prev_dir}
```

### New Directory

```bash
{self.path}
```
"""


class SysEnv(Fs):
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

    def markdown(self, context: Dict[str, Any] = {}):
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
