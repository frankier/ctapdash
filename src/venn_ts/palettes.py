"""Bitset-indexed comparison colors and exact Venn menu previews."""

import json
from urllib.parse import quote

from bokeh.core.properties import String
from bokeh.models import Widget

# Copied from the local palette design file; no runtime file dependency.
PALETTES = {
    "cym_paint": [
        "#ffffff",
        "#00aeef",
        "#fcd300",
        "#5dd447",
        "#80022e",
        "#563088",
        "#be5023",
        "#7b7246",
    ],
    "cym_sub": [
        "#ffffff",
        "#00ffff",
        "#ffff00",
        "#00ff00",
        "#ff00ff",
        "#0000ff",
        "#ff0000",
        "#000000",
    ],
    "rbg_add": [
        "#000000",
        "#ff0000",
        "#0000ff",
        "#ff00ff",
        "#00ff00",
        "#ffff00",
        "#00ffff",
        "#ffffff",
    ],
}


def hull_color(background):
    """Move each background component two hex shades toward mid-gray."""
    values = [int(background[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{v + 34 if v < 128 else v - 34:02x}" for v in values)


def palette_icon(colors):
    circles = [(25, 23), (43, 23), (34, 39)]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="68" height="62" viewBox="0 0 68 62"><defs>'
    ]
    for bit, (x, y) in enumerate(circles):
        for inside in (0, 1):
            parts.append(
                f'<mask id="m{bit}{inside}" maskUnits="userSpaceOnUse" x="0" y="0" width="68" height="62"><rect width="68" height="62" fill="{"black" if inside else "white"}"/><circle cx="{x}" cy="{y}" r="19" fill="{"white" if inside else "black"}"/></mask>'
            )
    parts.append(f'</defs><rect width="68" height="62" rx="4" fill="{colors[0]}"/>')
    for index in range(1, 8):
        for bit in range(3):
            parts.append(f'<g mask="url(#m{bit}{int(bool(index & (1 << bit)))})">')
        parts.append(
            f'<rect width="68" height="62" fill="{colors[index]}"/></g></g></g>'
        )
    parts.append("</svg>")
    return "data:image/svg+xml," + quote("".join(parts))


class PaletteSelector(Widget):
    value = String(default="cym_paint")
    options_json = String(default="[]")

    def __init__(self, **kwargs):
        kwargs.setdefault(
            "options_json",
            json.dumps(
                [
                    {"name": name, "icon": palette_icon(colors)}
                    for name, colors in PALETTES.items()
                ]
            ),
        )
        super().__init__(**kwargs)
