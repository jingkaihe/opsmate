from opsmate.dino.types import ToolCall, PresentationMixin
from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, ClassVar, Self, Any
from collections import defaultdict
from pathlib import Path
from opsmate.tools.utils import maybe_truncate_text
from opsmate.dino import dino
import asyncio
import os
import structlog
import traceback
import re
from jinja2 import Template

logger = structlog.get_logger(__name__)

PATTERN_NOT_FOUND = "Pattern not found"


class Result(BaseModel):
    """
    Result is a model that represents the result of a tool call.
    """

    output: Optional[str] = Field(
        description="The output of the tool call",
        default=None,
    )
    error: Optional[str] = Field(
        description="The error of the tool call",
        default=None,
    )

    operation: Optional[str] = Field(
        description="The operation that is being performed",
        default=None,
    )

    path: Optional[str] = Field(
        description="The path of the file or directory to be operated on",
        default=None,
    )

    old_content: Optional[str] = Field(
        description="The old content to be replaced by the new content",
        default=None,
    )

    content: Optional[str] = Field(
        description="The new content that is being operated on",
        default=None,
    )

    insert_line_number: Optional[int] = Field(
        description="The line number to insert the content at, only applicable for the `insert` command. Note the line number is 0-indexed.",
        default=None,
    )

    line_start: Optional[int] = Field(
        description="The start line number to be operated on, only applicable for the 'update' command. Note the line number is 0-indexed.",
        default=None,
    )
    line_end: Optional[int] = Field(
        description="The end line number to be operated on, only applicable for the 'update' command. Note the line number is 0-indexed.",
        default=None,
    )


class ACITool(ToolCall[Result], PresentationMixin):
    """
    # ACITool

    File system utility with the following commands:

    search <file|dir> <content>           # Search in file/directory
    view <file|dir> [start] [end]        # View file (optional 0-indexed line range) or directory
    create <file> <content>          # Create new file
    update <file> <old> <new> [start] [end]       # Replace content (old must be unique), with optional 0-indexed line range
    append <file> <line> <content>   # Insert at line number
    undo <file>                      # Undo last file change

    Notes:
    - Line numbers are 0-indexed
    - Directory view: 2-depth, ignores dotfiles
    - Empty new content in update deletes old content
    """

    _file_history: ClassVar[Dict[Path, List[str]]] = defaultdict(list)

    # number of lines before and after the match
    search_context_window: ClassVar[int] = 4

    command: str = Field(
        description="The command to execute",
        choices=["search", "view", "create", "update", "insert", "undo"],
    )

    path: str = Field(description="The path of the file or directory to be operated on")

    insert_line_number: Optional[int] = Field(
        description="The line number to insert the content at, only applicable for the `insert` command. Note the line number is 0-indexed.",
        default=None,
    )

    line_start: Optional[int] = Field(
        description="The start line number to be operated on, only applicable to the 'view' and 'update' commands. Note the line number is 0-indexed.",
        default=None,
    )
    line_end: Optional[int] = Field(
        description="The end line number to be operated on, only applicable to the 'view' and 'update' commands. Note the line number is 0-indexed.",
        default=None,
    )

    content: Optional[str] = Field(
        description="The content to be added to the file, only applicable for the `search`, `create`, `update` and `insert` commands.",
        default=None,
    )

    old_content: Optional[str] = Field(
        description="The old content to be replaced by the new content, only applicable for the `update` command.",
        default=None,
    )

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if self.command != "create":
            if not Path(self.path).exists():
                raise ValueError(f"File or directory {self.path} does not exist")
        else:
            if Path(self.path).exists():
                raise ValueError(f"File or directory {self.path} already exists")
        return self

    @model_validator(mode="after")
    def validate_search_command(self) -> Self:
        if self.command == "search":
            if self.content is None:
                self.content = ""
        return self

    @model_validator(mode="after")
    def validate_view_command(self) -> Self:
        if self.command == "view":
            if self.line_start is None and self.line_end is not None:
                raise ValueError("line_start is required when line_end is provided")
            if self.line_end is None and self.line_start is not None:
                raise ValueError("line_end is required when line_start is provided")

            if self.line_start is not None and self.line_end is not None:
                if self.line_start < 0 or self.line_end < self.line_start:
                    raise ValueError(
                        "line_end must be greater than or equal to line_start"
                    )

        return self

    @model_validator(mode="after")
    def validate_create_command(self) -> Self:
        if self.command == "create":
            if self.content is None:
                raise ValueError("content is required for the create command")
        return self

    @model_validator(mode="after")
    def validate_update_command(self) -> Self:
        if self.command == "update":
            if self.old_content is None:
                raise ValueError("old_content is required for the update command")
            if self.content is None:
                raise ValueError("new_content is required for the update command")
        return self

    @model_validator(mode="after")
    def validate_insert_command(self) -> Self:
        if self.command == "insert":
            if self.content is None:
                raise ValueError("content is required for the insert command")
            if self.insert_line_number is None:
                raise ValueError(
                    "insert_line_number is required for the insert command"
                )
        return self

    @model_validator(mode="after")
    def validate_undo_command(self) -> Self:
        if self.command == "undo":
            if self.path is None:
                raise ValueError("path is required for the undo command")
        return self

    async def __call__(self) -> Result:
        logger.info(
            "executing command",
            command=self.command,
            path=self.path,
            content=self.content,
            old_content=self.old_content,
            insert_line_number=self.insert_line_number,
            line_start=self.line_start,
            line_end=self.line_end,
        )
        if self.command == "search":
            return await self.search()
        elif self.command == "view":
            return await self.view()
        elif self.command == "create":
            return await self.create()
        elif self.command == "update":
            return await self.update()
        elif self.command == "insert":
            return await self.insert()
        elif self.command == "undo":
            return await self.undo()
        else:
            raise ValueError(f"Invalid command: {self.command}")

    async def create(self) -> Result:
        try:
            Path(self.path).write_text(self.content)
            self._file_history[Path(self.path)].append(self.content)
        except Exception as e:
            return Result(
                error=str(e),
                operation="create",
                path=self.path,
                content=self.content,
            )
        return Result(
            output="File created successfully",
            operation="create",
            path=self.path,
            content=self.content,
        )

    async def view(self) -> Result:
        if Path(self.path).is_file():
            return await self._view_file()
        elif Path(self.path).is_dir():
            return await self._view_directory()
        else:
            return Result(
                error=f"The path {self.path} is not a file or directory",
                operation="view",
                path=self.path,
            )

    async def _view_file(self) -> Result:
        try:
            with open(self.path, "r") as f:
                lines = f.readlines()

            # Handle line range if specified
            if self.line_start is not None and self.line_end is not None:
                if self.line_end >= len(lines):
                    raise ValueError(
                        f"end line number {self.line_end} is out of range (file has {len(lines)} lines)"
                    )
                lines = lines[self.line_start : self.line_end + 1]

            # Format lines with line numbers (0-indexed)
            numbered_contents = ""
            for i, line in enumerate(lines):
                line_number = i if self.line_start is None else i + self.line_start
                numbered_contents += f"{line_number:4d} | {line}"

            return Result(
                output=maybe_truncate_text(numbered_contents),
                operation="view",
                path=self.path,
            )
        except Exception as e:
            return Result(
                error=f"Failed to view file: {e}",
                operation="view",
                path=self.path,
            )

    async def _view_directory(self) -> Result:
        try:
            process = await asyncio.create_subprocess_shell(
                rf"find {self.path} -maxdepth 2 -not -path '*/\.*' -not -name '.*' | sort",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            # Add 5 second timeout to communicate
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5.0)
            return Result(
                output=maybe_truncate_text(stdout.decode()),
                operation="view",
                path=self.path,
            )
        except asyncio.TimeoutError:
            # Kill the process if it times out
            process.kill()
            return Result(
                error="Directory listing timed out after 5 seconds",
                operation="view",
                path=self.path,
            )
        except Exception as e:
            return Result(
                error=f"Failed to view directory: {e}",
                operation="view",
                path=self.path,
            )

    async def insert(self) -> Result:
        """
        Insert content into a file at a specific line number.
        """
        try:
            with open(self.path, "r") as f:
                lines = [line.rstrip() for line in f.readlines()]
            if self.insert_line_number < 0 or self.insert_line_number >= len(lines):
                raise ValueError(
                    f"end line number {self.insert_line_number} is out of range (file has {len(lines)} lines)"
                )
            lines.insert(self.insert_line_number, self.content)

            new_content = "\n".join(lines)
            Path(self.path).write_text(new_content)
            self._file_history[Path(self.path)].append(new_content)

            return Result(
                output="Content inserted successfully",
                operation="insert",
                path=self.path,
                content=self.content,
                insert_line_number=self.insert_line_number,
            )
        except Exception as e:
            return Result(
                error=f"Failed to insert content into file: {e}",
                operation="insert",
                path=self.path,
                content=self.content,
                insert_line_number=self.insert_line_number,
            )

    async def update(self) -> Result:
        """
        Replace the old content with the new content within a specified line range.
        """
        path = Path(self.path)
        file_content = path.read_text()
        lines = file_content.splitlines()

        # Determine the range of lines to operate on
        start = self.line_start if self.line_start is not None else 0
        end = self.line_end if self.line_end is not None else len(lines) - 1

        # Validate line range
        if start < 0 or end >= len(lines) or start > end:
            return Result(
                error="Invalid line range specified",
                operation="update",
                path=self.path,
                old_content=self.old_content,
                line_start=start,
                line_end=end,
            )

        # Join the lines within the specified range
        range_content = "\n".join(lines[start : end + 1])

        # Check for occurrences within the specified range
        occurrences = range_content.count(self.old_content)
        if occurrences == 0:
            return Result(
                error="Old content not found in the specified line range",
                operation="update",
                path=self.path,
                old_content=self.old_content,
                line_start=start,
                line_end=end,
            )
        elif occurrences > 1:
            return Result(
                error="Old content occurs more than once in the specified line range, please make sure its uniqueness",
                operation="update",
                path=self.path,
                old_content=self.old_content,
                line_start=start,
                line_end=end,
            )

        # Replace the old content with the new content within the range
        updated_range_content = range_content.replace(self.old_content, self.content)

        # Update the lines with the modified content
        lines[start : end + 1] = updated_range_content.splitlines()

        # Write the updated content back to the file
        new_file_content = "\n".join(lines)
        path.write_text(new_file_content)
        self._file_history[path].append(new_file_content)

        return Result(
            output="Content updated successfully",
            operation="update",
            path=self.path,
        )

    async def undo(self) -> Result:
        """
        Undo the last file operation.
        """
        path = Path(self.path)
        if len(self._file_history[path]) <= 1:
            return Result(
                error="There is no history of file operations",
                operation="undo",
                path=self.path,
            )
        self._file_history[path].pop()
        latest_content = self._file_history[path][-1]
        path.write_text(latest_content)
        return Result(
            output="File rolled back to previous state",
            operation="undo",
            path=self.path,
        )

    async def search(self) -> Result:
        """
        Search for a pattern in a file or directory using regex.
        """
        path = self.path
        if path == ".":
            path = os.getcwd()
        path_type_check = Path(path)
        logger.info("searching", path=path, content=self.content)
        try:
            if path_type_check.is_file():
                result = await self._search_file(path)
                return Result(output=maybe_truncate_text(result))

            elif path_type_check.is_dir():
                results = ""
                for root, _dirs, files in os.walk(path):
                    for file in files:
                        if file.startswith(".") or root.startswith("."):
                            continue
                        result = await self._search_file(os.path.join(root, file))
                        if result and result != PATTERN_NOT_FOUND:
                            results += f"{root}/{file}\n---\n{result}\n"
                logger.info("search results", results=results)
                return Result(
                    output=maybe_truncate_text(results),
                    operation="search",
                    path=self.path,
                    content=self.content,
                )

            else:
                return Result(
                    error=f"Path {path} is not a file or directory",
                    operation="search",
                    path=self.path,
                    content=self.content,
                )

        except asyncio.TimeoutError:
            return Result(
                error="Search timed out after 5 seconds",
                operation="search",
                path=self.path,
                content=self.content,
            )
        except Exception as e:
            return Result(
                error=f"Failed to search: {e}",
                operation="search",
                path=self.path,
                content=self.content,
            )

    async def _search_file(self, filename: str) -> str:
        try:
            cmd = f"grep -En -A {self.search_context_window} -B {self.search_context_window} '{self.content}' {filename}"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5.0)
            result = stdout.decode().strip()

            if not result:
                return PATTERN_NOT_FOUND

            formatted_lines = []
            for line in result.splitlines():
                # line_num, content = re.split(r"[:-]", line)
                match = re.split(r"([:-])", line, maxsplit=1)
                if len(match) >= 3:
                    line_num, sep, content = match
                else:
                    raise ValueError(f"Failed to split line: {line}")
                line_num, sep, content = match

                if sep == ":":
                    sep = "|"

                formatted_lines.append(f"{int(line_num)-1:4d} {sep} {content.rstrip()}")

            return "\n".join(formatted_lines)
        except Exception as e:
            logger.error(
                "Failed to search file", error=e, traceback=traceback.format_exc()
            )
            return f"Failed to search file: {e}"

    def markdown(self, context: dict[str, Any] = {}):
        match self.command:
            case "search":
                return self._render_search_markdown()
            case "view":
                return self._render_view_markdown()
            case "create":
                return self._render_create_markdown()
            case "update":
                return self._render_update_markdown()
            case "insert":
                return self._render_insert_markdown()
            case "undo":
                return self._render_undo_markdown()
            case _:
                return self.output

    def _render_search_markdown(self) -> str:
        template = Template(
            """
## ACI Results

action: `search`

search: `{{ output.content }}`

path: `{{ output.path }}`

{% if output.error %}
### Error
```
{{ output.error }}
```
{% else %}
### Output
```
{{ output.output }}
```
{% endif %}
"""
        )
        return template.render(output=self.output)

    def _render_view_markdown(self) -> str:
        template = Template(
            """
## ACI Results

action: `view`

path: `{{ output.path }}`

{% if output.line_start %}
line_start: `{{ output.line_start }}`
line_end: `{{ output.line_end }}`
{% endif %}

{% if output.error %}
### Error
```
{{ output.error }}
```
{% else %}
### Output
```
{{ output.output }}
```
{% endif %}
"""
        )
        return template.render(output=self.output)

    def _render_create_markdown(self) -> str:
        template = Template(
            """
## ACI Results

action: `create`

path: `{{ output.path }}`

{% if output.error %}
### Error
```
{{ output.error }}
```
{% else %}
### Output
```
{{ output.output }}
```
{% endif %}
"""
        )
        return template.render(output=self.output)

    def _render_insert_markdown(self) -> str:
        template = Template(
            """
## ACI Results

action: `insert`

path: `{{ output.path }}`

content: `{{ output.content }}`

line_number: `{{ output.insert_line_number }}`

{% if output.error %}
### Error
```
{{ output.error }}
```
{% else %}
### Output
```
{{ output.output }}
```
{% endif %}
"""
        )
        return template.render(output=self.output)

    def _render_undo_markdown(self) -> str:
        template = Template(
            """
## ACI Results

action: `undo`

path: `{{ output.path }}`

{% if output.error %}
### Error
```
{{ output.error }}
```
{% else %}
### Output
```
{{ output.output }}
```
{% endif %}
"""
        )
        return template.render(output=self.output)

    def _render_update_markdown(self) -> str:
        template = Template(
            """
## ACI Results

action: `update`

path: `{{ output.path }}`

old_content: `{{ old_content }}`

new_content: `{{ content }}`

{% if output.line_start %}
line_start: `{{ output.line_start }}`

line_end: `{{ output.line_end }}`
{% endif %}

{% if output.error %}
### Error
```
{{ output.error }}
```
{% else %}
### Output
```
{{ output.output }}
```
{% endif %}
"""
        )
        return template.render(
            output=self.output, old_content=self.old_content, content=self.content
        )


@dino(
    model="gpt-4o",
    response_model=ACITool,
)
async def coder(instruction: str):
    """
    You are a world class file system editor specialised in the `ACITool` tool.
    You will be given instructions to perform search, view, create, update,
    insert, and undo operations on files and directories.

    You will be returned the ACITool object to be executed.
    """
    return instruction
