import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.markup import escape
from typing import List, Tuple, Optional, Dict, Callable, Any
import inspect
import shutil
import os


def get_console(color: bool = False) -> Console:
    env_color = os.environ.get("OPSMATE_COLOR", "").lower()
    if env_color in ("1", "true", "yes", "on", "Y", "y"):
        color = True
    elif env_color in ("0", "false", "no", "off", "N", "n"):
        color = False
    
    return Console(color_system="auto" if color else None)

console = get_console()

class OpsmateHelpFormatter(click.HelpFormatter):
    def write_usage(self, prog, args='', prefix='Usage: '):
        usage_text = f"{prefix}{prog} {args}"
        self._write_usage = usage_text
    
    def write_heading(self, heading):
        self.write(f"\n{heading}:\n")
    
    def write_paragraph(self):
        self.write("\n")
    
    def write_text(self, text):
        if text:
            self.write(f"{text}\n")


class OpsmateGroup(click.Group):
    def format_help(self, ctx, formatter):
        original_formatter = click.HelpFormatter()
        super().format_help(ctx, original_formatter)
        help_text = original_formatter.getvalue()
        
        sections = self._parse_help_sections(help_text)
        self._render_rich_help(ctx, sections)
        
        return ""
    
    def _parse_help_sections(self, help_text: str) -> Dict[str, List[str]]:
        sections = {
            "Usage": [],
            "Description": [],
            "Commands": [],
            "Options": []
        }
        
        current_section = None
        for line in help_text.splitlines():
            line = line.rstrip()
            
            if line.endswith(':') and not line.startswith(' '):
                import re
                match = re.match(r'^(\S.*?):$', line)
                if match:
                    section_name = match.group(1).strip()
                current_section = section_name
                continue
                
            if current_section and line:
                sections.setdefault(current_section, []).append(line)
        
        return sections
    
    def _render_rich_help(self, ctx, sections: Dict[str, List[str]]):
        from opsmate import __version__
        title = f"[bold cyan]Opsmate CLI[/bold cyan] [dim]v{__version__}[/dim]"
        console.print(Panel(title, expand=False))
        
        if "Description" in sections and sections["Description"]:
            description = "\n".join(sections["Description"])
            console.print(Markdown(f"## Description\n\n{description}"))
        
        if "Usage" in sections and sections["Usage"]:
            usage_text = " ".join(sections["Usage"])
            console.print(f"\n[bold]Usage:[/bold] [yellow]{usage_text}[/yellow]")
        
        if "Commands" in sections and sections["Commands"]:
            console.print("\n[bold]Commands:[/bold]")
            
            for line in sections["Commands"]:
                if line.strip():
                    parts = line.strip().split('  ', 1)
                    if len(parts) == 2:
                        cmd, desc = parts[0].strip(), parts[1].strip()
                        console.print(f"  [cyan]{cmd}[/cyan]")
                        console.print(f"      {desc}\n")
        
        if "Options" in sections and sections["Options"]:
            console.print("\n[bold]Options:[/bold]")
            
            options_dict = {}
            current_option = None
            
            for line in sections["Options"]:
                stripped_line = line.strip()
                
                if stripped_line.startswith("-"):
                    parts = stripped_line.split("  ", 1)
                    opt_name = parts[0].strip()
                    initial_desc = parts[1].strip() if len(parts) > 1 else ""
                    
                    current_option = opt_name
                    if current_option not in options_dict:
                        options_dict[current_option] = []
                    
                    if initial_desc:
                        options_dict[current_option].append(initial_desc)
                elif stripped_line and current_option is not None:
                    if current_option in options_dict:
                        options_dict[current_option].append(stripped_line)
            
            for option, desc_lines in options_dict.items():
                full_desc = " ".join(desc_lines)
                full_desc = full_desc.replace("…", "").replace("\n", " ")
                
                console.print(f"  [green]{option}[/green]")
                console.print(f"      {full_desc}\n")
        
        console.print(
            "\n[dim italic]Tip: Use 'opsmate COMMAND --help' for more information about a specific command.[/dim italic]"
        )

    def get_command(self, ctx, cmd_name):
        cmd = super().get_command(ctx, cmd_name)
        
        if cmd is not None:
            return cmd
        
        from difflib import get_close_matches
        
        available_commands = self.list_commands(ctx)
        close_matches = get_close_matches(cmd_name, available_commands, n=3, cutoff=0.6)
        
        console.print(f"\n[bold red]Error:[/bold red] No such command '[yellow]{cmd_name}[/yellow]'.")
        
        if close_matches:
            console.print("\nDid you mean:")
            for match in close_matches:
                console.print(f"  [cyan]{match}[/cyan]")
        
        console.print("\nRun '[green]opsmate --help[/green]' for a list of all available commands.")
        
        return None


class OpsmateCommand(click.Command):
    def format_help(self, ctx, formatter):
        original_formatter = click.HelpFormatter()
        super().format_help(ctx, original_formatter)
        help_text = original_formatter.getvalue()
        
        sections = self._parse_help_sections(help_text)
        self._render_rich_command_help(ctx, sections)
        
        return ""
    
    def _parse_help_sections(self, help_text: str) -> Dict[str, List[str]]:
        sections = {
            "Usage": [],
            "Description": [],
            "Options": [],
            "Arguments": []
        }
        
        current_section = None
        for line in help_text.splitlines():
            line = line.rstrip()
            
            if line.endswith(':') and not line.startswith(' '):
                section_name = line[:-1].strip()
                current_section = section_name
                continue
                
            if current_section and line:
                sections.setdefault(current_section, []).append(line)
        
        return sections
    
    def _render_rich_command_help(self, ctx, sections: Dict[str, List[str]]):
        command_name = ctx.command.name
        full_command = f"opsmate {command_name}"
        
        if "Usage" in sections and sections["Usage"]:
            usage_text = " ".join(sections["Usage"])
            usage_text = usage_text.replace("opsmate-cli", "opsmate")
            console.print(f"\n[bold]Usage:[/bold] [yellow]{usage_text}[/yellow]")
        else:
            console.print(f"\n[bold]Usage:[/bold] [yellow]opsmate {command_name} [OPTIONS][/yellow]")

        description = ""
        if "Description" in sections and sections["Description"]:
            description = "\n".join(sections["Description"]).strip()
        if not description:
            description = self._get_command_description(command_name) 
        if description:
             console.print(f"\n  {description}\n")

        if "Arguments" in sections and sections["Arguments"]:
            console.print("\n[bold]Arguments:[/bold]")
            for line in sections["Arguments"]:
                if line.strip():
                    parts = line.strip().split('  ', 1)
                    if len(parts) == 2:
                        arg, desc = parts[0].strip(), parts[1].strip()
                        console.print(f"  [magenta]{arg}[/magenta]")
                        console.print(f"      {desc}\n")

        if "Options" in sections and sections["Options"]:
            console.print("\n[bold]Options:[/bold]")

            option_info = {}
            for param in ctx.command.params:
                if isinstance(param, click.Option):
                    default_value = param.get_default(ctx, call=False)
                    show_default_explicit = param.show_default
                    show_default_calculated = False
                    if show_default_explicit is not None:
                        show_default_calculated = show_default_explicit
                    elif default_value is not None:
                        show_default_calculated = True

                    if show_default_calculated:
                        for opt in param.opts:
                            option_info[opt] = default_value

            options_dict = {}
            current_option_key = None
            for line in sections["Options"]:
                stripped_line = line.strip()
                if stripped_line.startswith("-"):
                    parts = stripped_line.split("  ", 1)
                    current_option_key = parts[0].strip()
                    initial_desc = parts[1].strip() if len(parts) > 1 else ""
                    options_dict[current_option_key] = [initial_desc] if initial_desc else []
                elif stripped_line and current_option_key is not None:
                    if current_option_key in options_dict:
                        options_dict[current_option_key].append(stripped_line)

            for option_key, desc_lines in options_dict.items():
                parts = option_key.split(' ')
                option_names_part = parts[0]
                if ',' in option_names_part and len(parts) > 1 and parts[1].startswith('-'):
                    option_names_part += f" {parts[1]}"
                elif ',' in option_names_part and not option_names_part.endswith(','):
                    last_comma_index = option_names_part.rfind(',')
                    potential_name = (
                        option_names_part[: last_comma_index + 1]
                        + option_names_part[last_comma_index + 1 :].split('=')[0]
                    )
                    if all(p.strip().startswith('-') for p in potential_name.split(',')):
                        option_names_part = potential_name

                opts_in_key = [opt.strip() for opt in option_names_part.split(',') if opt.strip()]

                full_desc = " ".join(desc_lines).replace("…", "").strip()
                default_value = None
                matched_opt = None

                for opt in opts_in_key:
                    if opt in option_info:
                        default_value = option_info[opt]
                        matched_opt = opt
                        break

                default_str = None
                if matched_opt is not None:
                    if isinstance(default_value, bool):
                        default_str = f"[default: {str(default_value)}]"
                    elif isinstance(default_value, (list, tuple)):
                        if not default_value:
                            default_str = "[default: ]"
                        else:
                            default_str = f"[default: {','.join(str(x) for x in default_value)}]"
                    elif default_value == "":
                        default_str = '[default: ""]'
                    elif default_value is None:
                        default_str = "[default: None]"
                    else:
                        if isinstance(default_value, str):
                            default_str = f'[default: "{default_value}"]'
                        else:
                            default_str = f"[default: {default_value}]"
                    
                    default_str = escape(default_str)

                console.print(f"  [green]{option_key}[/green]")

                if default_str:
                    console.print(f"      {full_desc} [dim]{default_str}[/dim]\n")
                else:
                    console.print(f"      {full_desc}\n")

        self._show_command_examples(ctx)

    def _get_command_description(self, command_name: str) -> str:
        command_descriptions = {
            "worker": "Start the Opsmate worker for processing background jobs.",
            "solve": "Ask Opsmate to solve a problem or answer a question.",
            "run": "Run a task or command with the Opsmate assistant.",
            "chat": "Start an interactive chat session with the Opsmate assistant.",
            "serve": "Start the Opsmate web server for API access.",
            "ingest": "Ingest knowledge from a source into the Opsmate knowledge base.",
            "ingest-prometheus-metrics-metadata": "Import Prometheus metrics metadata into the knowledge base.",
            "schedule-embeddings-reindex": "Schedule periodic reindexing of embeddings table.",
            "list-contexts": "List all available contexts that can be used with Opsmate.",
            "list-tools": "List all available tools that can be used with Opsmate.",
            "list-models": "List all available language models that can be used with Opsmate.",
            "list-runtimes": "List all available runtime environments for tool execution.",
            "version": "Display the current version of Opsmate.",
            "db-migrate": "Apply database migrations to update schema to latest version.",
            "db-rollback": "Roll back database migrations to a previous version.",
            "db-revisions": "List all available database migration revisions.",
            "reset": "Reset the Opsmate database and embeddings storage.",
            "install": "Install packages or plugins for Opsmate.",
            "uninstall": "Remove installed packages or plugins from Opsmate.",
        }
        
        return command_descriptions.get(
            command_name, 
            f"Command '{command_name}' for the Opsmate CLI."
        )

    def _show_command_examples(self, ctx):
        command_name = ctx.command.name
        
        predefined_examples = {
            "worker": [
                ("Start worker with default settings", "opsmate worker"),
                ("Use specific queue", "opsmate worker -q embeddings"),
                ("Run multiple workers", "opsmate worker -w 5"),
                ("Specify model", "opsmate worker --model claude-3-5-sonnet-20240620"),
            ],
            "solve": [
                ("Basic usage", "opsmate solve 'Why is my CPU usage spiking?'"),
                ("With context", "opsmate solve --context ~/logs.txt 'Debug these error logs'"),
                ("With specific model", "opsmate solve --model claude-3-5-sonnet-20240620 'Analyze performance'"),
            ],
            "run": [
                ("Basic usage", "opsmate run 'List all processes using more than 10% CPU'"),
                ("With review flag", "opsmate run -r 'Check disk space usage'"),
                ("With system prompt", "opsmate run -s 'You are a Linux expert' 'Find large files'"),
            ],
            "chat": [
                ("Start chat session", "opsmate chat"),
                ("With specific model", "opsmate chat --model gpt-4o"),
                ("With context", "opsmate chat --context production-logs"),
            ],
            "version": [
                ("Show version", "opsmate version"),
            ],
            "serve": [
                ("Start the web server", "opsmate serve"),
                ("Specify port", "opsmate serve --port 8080"),
                ("With debug mode", "opsmate serve --debug"),
            ],
            "list-contexts": [
                ("List available contexts", "opsmate list-contexts"),
            ],
            "list-tools": [
                ("List available tools", "opsmate list-tools"),
            ],
            "list-models": [
                ("List available models", "opsmate list-models"),
            ],
            "ingest": [
                ("Basic usage", "opsmate ingest"),
                ("With source path", "opsmate ingest --source fs:////path/to/kb"),
                ("From GitHub repo", "opsmate ingest --source github:///owner/repo"),
            ],
            "install": [
                ("Install package", "opsmate install package-name"),
                ("Install with upgrade", "opsmate install -U package-name"),
                ("Install in editable mode", "opsmate install -e ."),
            ],
            "test": [
                ("Run all tests", "opsmate test"),
                ("Run with verbosity", "opsmate test -v"),
                ("Run specific tests", "opsmate test --pattern 'test_cli*'"),
            ],
            "embeddings": [
                ("Generate embeddings", "opsmate embeddings 'Text to embed'"),
                ("With specific model", "opsmate embeddings --model text-embedding-3-large 'Text to embed'"),
            ],
            "categorize": [
                ("Categorize content", "opsmate categorize 'Content to categorize'"),
                ("With specific categories", "opsmate categorize --categories 'tech,finance,health' 'Content to categorize'"),
            ],
            "prepare-context": [
                ("Prepare a context", "opsmate prepare-context --name my-context"),
                ("From specific source", "opsmate prepare-context --name my-context --source-path /path/to/data"),
            ],
            "summarize": [
                ("Summarize content", "opsmate summarize 'Long text to summarize'"),
                ("From file", "opsmate summarize --file report.txt"),
                ("With word limit", "opsmate summarize --max-words 100 'Long text to summarize'"),
            ],
            "generate": [
                ("Generate content", "opsmate generate 'Write a bash script to backup files'"),
                ("With specific format", "opsmate generate --format markdown 'Create a project readme'"),
            ],
            "shell-completion": [
                ("Generate shell completion for bash", "opsmate shell-completion bash > ~/.opsmate-completion.bash"),
                ("Generate shell completion for zsh", "opsmate shell-completion zsh > ~/.zfunc/_opsmate"),
            ],
            "upgrade": [
                ("Check for upgrades", "opsmate upgrade --check"),
                ("Perform upgrade", "opsmate upgrade"),
            ],
            "config": [
                ("Show current configuration", "opsmate config show"),
                ("Set configuration value", "opsmate config set model.default gpt-4o"),
            ]
        }
        
        if command_name not in predefined_examples:
            examples = self._generate_examples(ctx)
        else:
            examples = predefined_examples[command_name]
        
        if examples:
            console.print("\n[bold]Examples:[/bold]")
            examples_table = Table(show_header=False, box=None, pad_edge=False)
            examples_table.add_column("Description", style="dim")
            examples_table.add_column("Command", style="yellow")
            
            for desc, cmd in examples:
                examples_table.add_row(f"{desc}:", f"$ {cmd}")
            
            console.print(examples_table)

    def _generate_examples(self, ctx) -> List[Tuple[str, str]]:
        command = ctx.command
        examples = []        
        command_name = command.name
        
        examples.append(("Basic usage", f"opsmate {command_name}"))
        
        args = [param for param in command.params if isinstance(param, click.Argument)]
        if args:
            arg_examples = []
            for arg in args:
                if 'TEXT' in str(arg.type).upper():
                    arg_examples.append(f"'example-{arg.name}'")
                elif 'INT' in str(arg.type).upper():
                    arg_examples.append("42")
                elif 'FLOAT' in str(arg.type).upper():
                    arg_examples.append("3.14")
                elif 'BOOL' in str(arg.type).upper():
                    arg_examples.append("true")
                elif 'PATH' in str(arg.type).upper():
                    arg_examples.append("/path/to/file")
                else:
                    arg_examples.append(f"<{arg.name}>")
            
            if arg_examples:
                examples.append(
                    ("With arguments", f"opsmate {command_name} {' '.join(arg_examples)}")
                )
        
        options = [param for param in command.params if isinstance(param, click.Option)]
        if options:
            flag_options = [opt for opt in options if opt.is_flag]
            if flag_options and len(flag_options) > 0:
                flag_example = f"opsmate {command_name}"
                flag_names = []
                
                for i, opt in enumerate(flag_options[:2]):
                    option_name = next((n for n in opt.opts if n.startswith('-') and not n.startswith('--')), opt.opts[0])
                    flag_example += f" {option_name}"
                    flag_names.append(option_name.lstrip('-'))
                
                examples.append((f"With {'flag' if len(flag_options[:2]) == 1 else 'flags'} {', '.join(flag_names)}", flag_example))
            
            value_options = [opt for opt in options if not opt.is_flag and not opt.hidden]
            if value_options and len(value_options) > 0:
                important_keywords = ['model', 'output', 'file', 'path', 'name', 'config', 'format']
                
                selected_option = None
                for keyword in important_keywords:
                    matching_options = [opt for opt in value_options if keyword in ' '.join(opt.opts)]
                    if matching_options:
                        selected_option = matching_options[0]
                        break
                
                if not selected_option and value_options:
                    selected_option = value_options[0]
                
                if selected_option:
                    option_name = next((n for n in selected_option.opts if n.startswith('--')), selected_option.opts[0])
                    
                    if 'PATH' in str(selected_option.type).upper():
                        value = '/path/to/file'
                    elif 'DIR' in str(selected_option.type).upper():
                        value = '/path/to/directory'
                    elif 'INT' in str(selected_option.type).upper():
                        value = '42'
                    elif 'TEXT' in str(selected_option.type).upper() or 'STRING' in str(selected_option.type).upper():
                        if 'file' in option_name or 'path' in option_name:
                            value = '/path/to/file'
                        elif 'dir' in option_name:
                            value = '/path/to/directory'
                        elif 'name' in option_name:
                            value = 'example-name'
                        elif 'model' in option_name:
                            value = 'claude-3-sonnet'
                        elif 'url' in option_name:
                            value = 'https://example.com'
                        else:
                            value = 'value'
                    
                    examples.append((
                        f"With {option_name.lstrip('-')} option", 
                        f"opsmate {command_name} {option_name} {value}"
                    ))
        
        return examples
