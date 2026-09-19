"""Working on both Gradio 5 and Gradio 6.

`gradio>=5.0` in the requirements means whoever installs today gets 6, and 6
removed things 5 required. Rather than pin one version and go stale, or pin one
and break the other, the handful of places that differ go through here.

What changed in 6, and what this does about it:

* ``Chatbot(type="messages")`` — the messages format is now the only one and the
  argument is gone. Several other Chatbot arguments went with it.
* ``Blocks(css=..., theme=...)`` — moved to ``launch()``.
* ``launch(show_api=...)`` — gone.

Every one of these is a hard ``TypeError``, not a warning, so a window built for
5 does not open at all on 6. The fix is to pass only what the installed version
accepts, which is a thing that can be asked rather than guessed.
"""

from __future__ import annotations

import inspect
from typing import Any

import gradio as gr

#: 5 and 6 are both in the wild. Anything newer is treated like 6.
MAJOR = int(str(gr.__version__).split(".")[0] or 5)

#: Where css and theme belong: the Blocks constructor, or launch().
STYLE_AT_LAUNCH = MAJOR >= 6

#: Set on a Blocks by :func:`blocks` so :func:`launch` can pass its styling on.
_STYLING = "_teacher_deferred_styling"


def accepted(function, given: dict[str, Any]) -> dict[str, Any]:
    """Only the arguments this version of ``function`` actually takes.

    Dropping an argument silently would be wrong for a setting that matters, so
    callers pass things here that are presentation only — a copy button, an API
    page — and nothing that changes what the application does.
    """
    try:
        allowed = inspect.signature(function).parameters
    except (TypeError, ValueError):  # pragma: no cover - a C-level callable
        return given
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in allowed.values()):
        return given
    return {key: value for key, value in given.items() if key in allowed}


def chatbot(**kwargs) -> gr.Chatbot:
    """A Chatbot in the messages format, however this version spells that."""
    if MAJOR < 6:
        kwargs.setdefault("type", "messages")
    return gr.Chatbot(**accepted(gr.Chatbot.__init__, kwargs))


def blocks(**kwargs) -> gr.Blocks:
    """A Blocks whose css and theme end up wherever this version wants them."""
    styling = {key: kwargs.pop(key) for key in ("css", "theme") if key in kwargs}
    if not STYLE_AT_LAUNCH:
        kwargs.update(styling)
        styling = {}
    app = gr.Blocks(**accepted(gr.Blocks.__init__, kwargs))
    # Stashed rather than passed twice at the call site; launch() picks it up.
    setattr(app, _STYLING, styling)
    return app


def launch(app: gr.Blocks, **kwargs) -> None:
    """Launch, passing the styling this version expects here and nothing it refuses."""
    kwargs.update(getattr(app, _STYLING, {}))
    app.launch(**accepted(gr.Blocks.launch, kwargs))
