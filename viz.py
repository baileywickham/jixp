"""Render a model_summary() as a self-contained interactive HTML explorer.

The viewer is a Vite + React + TypeScript app in viewer/; its single-file
build (viewer/dist/index.html, committed) is the template here. This module
just injects the title and the summary JSON into that template — the page
embeds the JSON (usable by other tools) and renders a collapsible tree with
one node per *source* block, so repeat shows a single node with multiplied
params and the page stays small for arbitrarily large models.

Rebuild the template after changing the viewer:
    npm --prefix viewer install && npm --prefix viewer run build
"""

import json
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "viewer" / "dist" / "index.html"


def render_html(summary, title="jixp model"):
    """Return a self-contained HTML page exploring the model summary."""
    if not _TEMPLATE_PATH.exists():
        raise RuntimeError(
            f"viewer template missing at {_TEMPLATE_PATH} — build it with: "
            "npm --prefix viewer install && npm --prefix viewer run build"
        )
    template = _TEMPLATE_PATH.read_text()
    data = json.dumps(summary).replace("</", "<\\/")
    return template.replace("__TITLE__", title).replace("__DATA__", data)
