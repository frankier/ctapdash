"""Bokeh transport for the shared browser channel selector."""

from bokeh.core.properties import List, String
from bokeh.models import Widget


class ChannelSelector(Widget):
    metadata_json = String(default="{}")
    available = List(String, default=[])
    bads_json = String(default="{}")
    value = List(String, default=[])
