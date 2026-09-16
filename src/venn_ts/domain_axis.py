"""Original-time labels and batched epoch/break overlays."""

from bokeh.core.properties import Bool, Float, List, String
from bokeh.models import CustomJSTickFormatter, Renderer


class DomainMarkers(Renderer):
    """One canvas renderer for any number of original-time boundaries."""

    positions = List(Float, default=[])
    labels = List(String, default=[])
    colors = List(String, default=[])
    dashed = Bool(default=False)
    show_labels = Bool(default=True)


def configure_domain_axis(plot, domain, *, epoch_visible=False, epochs=True):
    starts = [s.display_start for s in domain.segments]
    originals = [s.original_start for s in domain.segments]
    plot.xaxis.axis_label = None if epochs else "Time (s)"
    plot.xaxis.formatter = CustomJSTickFormatter(
        args=dict(starts=starts, originals=originals, tolerance=domain.interval * 1e-5),
        code="""
        let lo = 0, hi = starts.length
        while (lo + 1 < hi) {
            const mid = Math.floor((lo + hi) / 2)
            if (starts[mid] <= tick + tolerance) lo = mid
            else hi = mid
        }
        return Number((originals[lo] + tick - starts[lo]).toPrecision(8)).toString()
    """,
    )
    positions, labels = [], []
    for previous, segment in zip(domain.segments, domain.segments[1:]):
        previous_end = previous.original_start + previous.count * domain.interval
        if abs(previous_end - segment.original_start) < domain.interval * 1e-5:
            continue
        positions.append(segment.display_start)
        labels.append("↶" if segment.original_start < previous_end else "//")
    plot.renderers.append(
        DomainMarkers(
            positions=positions,
            labels=labels,
            colors=["#777777"] * len(positions),
            show_labels=epochs,
            level="annotation",
            name="domain-break",
        )
    )
    if not epochs:
        return []
    markers = [DomainMarkers(dashed=True, level="annotation", name="epoch-start")]
    update_epoch_markers(plot, domain, epoch_visible, markers)
    plot.renderers.extend(markers)
    return markers


def update_epoch_markers(plot, domain, visible, markers, palette=None):
    """Update arrays without creating or removing annotation models."""
    positions = domain.epoch_starts()
    markers[0].update(
        positions=list(positions),
        labels=[
            "".join("ABC"[side] for side in sorted(sides))
            for sides in positions.values()
        ],
        colors=[
            (
                palette
                or [
                    "white",
                    "red",
                    "blue",
                    "#7b3294",
                    "green",
                    "orange",
                    "cyan",
                    "black",
                ]
            )[sum(1 << side for side in sides)]
            for sides in positions.values()
        ],
        visible=visible,
    )
