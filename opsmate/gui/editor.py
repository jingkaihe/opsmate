from fasthtml.common import *
from opsmate.gui.models import Cell
import json

editor_script = Script(
    """
// Global map to keep track of editor instances
window.editorInstances = window.editorInstances || {};

function initEditor(id, default_value) {
    // Clean up any existing editor instance for this element
    if (window.editorInstances[id]) {
        // window.editorInstances[id].destroy();
        // window.editorInstances[id].container.remove();
        window.editorInstances[id] = null;
    }

    let editor;
    let completionTippy;
    let currentCompletion = '';

    editor = ace.edit(id);
    editor.setTheme("ace/theme/monokai");
    editor.session.setMode("ace/mode/markdown");
    editor.setOptions({
        fontSize: "14px",
        showPrintMargin: false,
        showGutter: true,
        highlightActiveLine: true,
        // maxLines: Infinity,
        wrap: true
    });

    editor.setValue(default_value);

    // Store the editor instance for later cleanup
    window.editorInstances[id] = editor;

    window.addEventListener('resize', function() {
        editor.resize();
    });
    completionTippy = tippy(document.getElementById('editor'), {
        content: 'Loading...',
        trigger: 'manual',
        placement: 'top-start',
        arrow: true,
        interactive: true
    });

    // Override the default tab behavior
    editor.commands.addCommand({
        name: 'insertCompletion',
        bindKey: {win: 'Tab', mac: 'Tab'},
        exec: function(editor) {
            if (currentCompletion) {
                editor.insert(currentCompletion);
                currentCompletion = '';
                completionTippy.hide();
            } else {
                editor.indent();
            }
        }
    });
}

"""
)


def CodeEditor(cell: Cell):
    return (
        Div(
            # Toolbar(),
            Div(
                Div(
                    id=f"cell-input-{cell.id}",
                    cls="w-full h-64",
                    name="input",
                    value=cell.input,
                ),
                Script(
                    f"""
                    // Initial load
                    document.body.addEventListener('DOMContentLoaded', function(evt) {{
                        if (document.getElementById('cell-input-{cell.id}')) {{
                            initEditor('cell-input-{cell.id}', {json.dumps(cell.input)});
                        }}
                    }});
                """
                ),
                cls="flex-grow w-full",
            ),
            cls="flex flex-col h-auto w-full",
            hidden=cell.hidden,
            id=f"cell-input-container-{cell.id}",
        ),
    )
