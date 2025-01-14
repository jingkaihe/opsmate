from pydantic import BaseModel, Field
from typing import Dict, Callable, Optional, TypeVar, ParamSpec, ClassVar, List
from opsmate.dino.types import ToolCall
import asyncio
import structlog
from functools import wraps
import os
import sys
import importlib
import inspect

logger = structlog.get_logger(__name__)
# Type variables for better type hints
T = TypeVar("T")
P = ParamSpec("P")


class Metadata(BaseModel):
    """Metadata for a plugin"""

    name: str = Field(description="The name of the plugin")
    description: str = Field(description="The description of the plugin")
    version: str = Field(description="The version of the plugin", default="0.1.0")
    author: Optional[str] = Field(
        description="The author of the plugin", default="unknown"
    )

    is_async: bool = Field(description="Whether the plugin is async", default=False)
    source: str = Field(description="The source of the plugin")


class Plugin(BaseModel):
    """A plugin"""

    name: str = Field(description="The name of the plugin")
    metadata: Metadata = Field(description="The metadata of the plugin")
    func: Callable[P, T] = Field(description="The callable of the plugin")

    async def execute(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if self.metadata.is_async:
            return await self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)


class PluginRegistry(BaseModel):
    """Function-based plugin registry with directory loading support"""

    _plugins: ClassVar[Dict[str, Plugin]] = {}
    _tools: ClassVar[Dict[str, ToolCall]] = {}
    _tool_sources: ClassVar[Dict[str, str]] = {}

    @classmethod
    def auto_discover(
        cls,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        author: Optional[str] = None,
    ):
        """auto-discover the function as a plugin"""

        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            plugin_name = name or func.__name__
            plugin_description = description or func.__doc__
            is_async = asyncio.iscoroutinefunction(func)

            # Get the caller's frame and file path
            caller_frame = inspect.currentframe().f_back
            source_file = inspect.getfile(caller_frame)
            abs_source_path = os.path.abspath(source_file)

            metadata = Metadata(
                name=plugin_name,
                description=plugin_description,
                version=version,
                author=author,
                is_async=is_async,
                source=abs_source_path,  # Use the absolute path from caller
            )

            if cls._plugins.get(plugin_name):
                conflict_source = cls._plugins[plugin_name].metadata.source
                raise ValueError(
                    f"Plugin {plugin_name} already exists at {conflict_source}"
                )

            cls._plugins[plugin_name] = Plugin(
                name=plugin_name,
                metadata=metadata,
                func=func,
            )

            logger.info(f"Discovered plugin {plugin_name}")

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def discover(cls, *plugin_dirs: str, ignore_conflicts: bool = False):
        """discover plugins in a directory"""
        cls._load_builtin(ignore_conflicts=True)
        for plugin_dir in plugin_dirs:
            cls._discover(plugin_dir, ignore_conflicts)

    @classmethod
    def _discover(cls, plugin_dir: str, ignore_conflicts: bool = False):
        """discover plugins in a directory"""

        if not os.path.exists(plugin_dir):
            logger.warning("Plugin directory does not exist", plugin_dir=plugin_dir)
            return

        logger.info(
            "adding the plugin directory to the sys path",
            plugin_dir=os.path.abspath(plugin_dir),
        )
        sys.path.append(os.path.abspath(plugin_dir))

        for filename in os.listdir(plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                plugin_path = os.path.join(plugin_dir, filename)
                cls._load_plugin_file(plugin_path, ignore_conflicts)

        sys.path.remove(os.path.abspath(plugin_dir))

    @classmethod
    def _load_plugin_file(cls, plugin_path: str, ignore_conflicts: bool = False):
        """load a plugin file"""
        logger.info("loading plugin file", plugin_path=plugin_path)
        try:
            module_name = os.path.basename(plugin_path).replace(".py", "")
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                logger.error("failed to load plugin file", plugin_path=plugin_path)
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls._load_dtools(module, ignore_conflicts)

            logger.info("loaded plugin file", plugin_path=plugin_path)
        except Exception as e:
            logger.error("failed to load plugin file", plugin_path=plugin_path, error=e)
            if not ignore_conflicts:
                raise e

    @classmethod
    def _load_builtin(
        cls,
        ignore_conflicts: bool = True,
        builtin_modules: List[str] = ["opsmate.tools"],
    ):
        logger.debug("loading builtin tools")
        for builtin_module in builtin_modules:
            logger.debug("loading builtin tools from", builtin_module=builtin_module)
            module = importlib.import_module(builtin_module)
            cls._load_dtools(module, ignore_conflicts)

    @classmethod
    def _load_dtools(cls, module, ignore_conflicts: bool = False):
        """load dtools from a module"""
        for item_name, item in inspect.getmembers(module):
            if inspect.isclass(item) and issubclass(item, ToolCall):
                logger.debug("loading dtool", dtool=item_name)
                if item_name in cls._tools:
                    conflict_source = cls._tool_sources[item_name]
                    logger.warning(
                        "tool already exists",
                        tool=item_name,
                        conflict_source=conflict_source,
                    )
                    if not ignore_conflicts:
                        raise ValueError(
                            f"Tool {item_name} already exists at {conflict_source}"
                        )
                cls._tools[item_name] = item
                cls._tool_sources[item_name] = module.__file__

    @classmethod
    def get_plugin(cls, plugin_name: str) -> Plugin:
        """get a plugin by name"""
        return cls._plugins.get(plugin_name)

    @classmethod
    def get_plugins(cls) -> List[Metadata]:
        """get all plugins"""
        return list(cls._metadata.values())

    @classmethod
    def get_tool(cls, tool_name: str) -> ToolCall:
        """get a tool by name"""
        return cls._tools.get(tool_name)

    @classmethod
    def get_tools(cls) -> Dict[str, ToolCall]:
        """get all tools"""
        return cls._tools

    @classmethod
    def get_tools_from_list(cls, tool_names: List[str]) -> List[ToolCall]:
        """get tools from a list of tool names"""
        result = []
        for tool_name in tool_names:
            if tool_name not in cls._tools:
                raise ValueError(f"Tool {tool_name} not found")
            result.append(cls._tools[tool_name])
        return result

    @classmethod
    def clear(cls):
        """clear all plugins"""
        cls._plugins = {}
