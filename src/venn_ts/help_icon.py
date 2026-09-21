"""Compact help icon whose tooltip belongs to the widget's model graph."""

from bokeh.core.properties import Instance
from bokeh.models import Tooltip, Widget


class HelpIcon(Widget):
    """Display a description-style icon with a tooltip on hover or focus.

    Add this widget to a layout normally; no separate document root or
    document-ready callback is needed for its tooltip.
    """

    tooltip = Instance(Tooltip)

    def __init__(self, content=None, *, tooltip=None, **kwargs):
        if content is not None and tooltip is not None:
            raise ValueError("Supply content or tooltip, not both")
        if tooltip is None:
            if content is None:
                raise ValueError("Supply content or tooltip")
            tooltip = Tooltip(content=content, position="right", interactive=False)
        kwargs.setdefault("width", 18)
        kwargs.setdefault("height", 18)
        kwargs.setdefault("margin", 0)
        kwargs.setdefault("align", "center")
        super().__init__(tooltip=tooltip, **kwargs)
