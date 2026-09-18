"""Standalone rendering of the help widget, without inherited icon styles."""

from bokeh.embed import file_html
from bokeh.layouts import row
from bokeh.resources import INLINE
from playwright.sync_api import expect

from venn_ts.help_icon import HelpIcon


def test_help_icon_renders_and_shows_tooltip(page, tmp_path):
    html = tmp_path / "help.html"
    html.write_text(file_html(row(HelpIcon("Choose layers")), INLINE))
    page.goto(html.as_uri())
    button = page.get_by_role("button", name="Help", exact=True)
    expect(button).to_be_visible()

    def assert_adjacent():
        assert button.evaluate("""el => {
            const tooltip = el.nextElementSibling
            return tooltip?.hasAttribute('popover') &&
                tooltip.shadowRoot?.querySelector('.bk-tooltip-content')?.textContent === 'Choose layers'
        }""")

    assert_adjacent()
    icon = button.locator(":scope > *")
    mask = icon.evaluate("el => getComputedStyle(el).maskImage")
    assert mask.startswith('url("data:image/svg+xml'), mask
    assert icon.evaluate("el => el.getBoundingClientRect().width") == 18
    assert (
        icon.evaluate("el => getComputedStyle(el).backgroundColor")
        != "rgba(0, 0, 0, 0)"
    )
    button.hover()
    tooltip = page.get_by_text("Choose layers", exact=True)
    expect(tooltip).to_be_visible()
    assert_adjacent()
    page.mouse.move(500, 400)
    expect(tooltip).to_be_hidden()
    assert_adjacent()
    button.focus()
    expect(tooltip).to_be_visible()
    assert_adjacent()
    button.press("Escape")
    expect(tooltip).to_be_hidden()
