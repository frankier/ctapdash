from bokeh.document import Document
from bokeh.layouts import row
from bokeh.models import Tooltip
import pytest

from venn_ts.help_icon import HelpIcon


def test_tooltip_follows_widget_document_membership():
    help_icon = HelpIcon("Select one or more options")
    layout = row(help_icon)
    doc = Document()
    doc.add_root(layout)
    assert help_icon.tooltip.document is doc
    assert help_icon.tooltip not in doc.roots
    doc.to_json()
    doc.remove_root(layout)
    assert help_icon.tooltip.document is None


def test_tooltips_are_independent_and_accept_custom_content():
    first = HelpIcon("First")
    second = HelpIcon("Second")
    assert first.tooltip is not second.tooltip
    tooltip = Tooltip(content="Custom", position="bottom")
    assert HelpIcon(tooltip=tooltip).tooltip is tooltip
    with pytest.raises(ValueError):
        HelpIcon("Ambiguous", tooltip=tooltip)
