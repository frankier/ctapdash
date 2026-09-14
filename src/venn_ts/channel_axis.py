"""Fixed-width channel labels shared by the viewport and navigation axes."""
from bokeh.core.properties import Bool, Dict, Float, String
from bokeh.models import LinearAxis


class ChannelAxis(LinearAxis):
    truncate_labels = Bool(default=False, help="Keep labels horizontal and truncate to the fixed width")
    avoid_overlap = Bool(default=False, help="Select labels with endpoint priority")
    channel_labels = Dict(Float, String, default={})
