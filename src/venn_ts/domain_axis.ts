import {Renderer, RendererView} from "@bokehjs/models/renderers/renderer"
import type {Context2d} from "@bokehjs/core/util/canvas"
import * as p from "@bokehjs/core/properties"

export class DomainMarkersView extends RendererView {
    declare model: DomainMarkers

    override get needs_clip(): boolean {
        return false
    }

    override connect_signals(): void {
        super.connect_signals()
        this.connect(this.model.change, () => this.request_paint())
    }

    protected override _paint(ctx: Context2d): void {
        const frame = this.plot_view.frame.bbox
        const xs = this.coordinates.x_scale.v_compute(this.model.positions)
        const groups = new Map<string, Set<number>>()
        for (let i = 0; i < xs.length; i++) {
            if (xs[i] < frame.left || xs[i] > frame.right) continue
            const color = this.model.colors[i]
            if (!groups.has(color)) groups.set(color, new Set())
            // At overview zoom many boundaries share a pixel. Draw it only once.
            groups.get(color)!.add(Math.round(xs[i]))
        }
        ctx.save()
        ctx.beginPath()
        ctx.rect(frame.left, frame.top, frame.width, frame.height)
        ctx.clip()
        ctx.globalAlpha = this.model.dashed ? 0.65 : 0.5
        ctx.lineWidth = 1
        ctx.setLineDash(this.model.dashed ? [6, 4] : [2, 4])
        for (const [color, pixels] of groups) {
            ctx.strokeStyle = color
            ctx.beginPath()
            for (const x of pixels) {
                ctx.moveTo(x, frame.top)
                ctx.lineTo(x, frame.bottom)
            }
            ctx.stroke()
        }
        ctx.restore()
        if (!this.model.show_labels) return
        ctx.save()
        ctx.beginPath()
        ctx.rect(frame.left, Math.max(0, frame.top - 26), frame.width, 26)
        ctx.clip()
        ctx.font = "11px sans-serif"
        ctx.textBaseline = "middle"
        // Epoch labels sit just right of a boundary, break labels just left.
        // Cull crowded text independently of the full set of vertical lines.
        let previous_right = -Infinity
        for (let i = 0; i < xs.length; i++) {
            const x = xs[i]
            if (x < frame.left || x > frame.right) continue
            const label = this.model.labels[i]
            const width = ctx.measureText(label).width
            const left = this.model.dashed ? x + 3 : x - width - 3
            if (left < previous_right + 4) continue
            ctx.fillStyle = this.model.colors[i]
            ctx.fillText(label, left, frame.top - 13)
            previous_right = left + width
        }
        ctx.restore()
    }
}

export namespace DomainMarkers {
    export type Attrs = p.AttrsOf<Props>
    export type Props = Renderer.Props & {
        positions: p.Property<number[]>
        labels: p.Property<string[]>
        colors: p.Property<string[]>
        dashed: p.Property<boolean>
        show_labels: p.Property<boolean>
    }
}
export interface DomainMarkers extends DomainMarkers.Attrs {}
export class DomainMarkers extends Renderer {
    static __module__ = "venn_ts.domain_axis"
    declare properties: DomainMarkers.Props
    declare __view_type__: DomainMarkersView
    static {
        this.prototype.default_view = DomainMarkersView
        this.define<DomainMarkers.Props>(({List, Float, Str, Bool}) => ({
            positions: [List(Float), []],
            labels: [List(Str), []],
            colors: [List(Str), []],
            dashed: [Bool, false],
            show_labels: [Bool, true],
        }))
    }
}
