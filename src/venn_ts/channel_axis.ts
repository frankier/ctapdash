import {LinearAxis, LinearAxisView} from "@bokehjs/models/axes/linear_axis"
import type {Extents, TickCoords} from "@bokehjs/models/axes/axis"
import type {Context2d} from "@bokehjs/core/util/canvas"
import * as p from "@bokehjs/core/properties"

type Label = {index: number, y: number, height: number, lines: string[], rotated: boolean}

export class ChannelAxisView extends LinearAxisView {
  declare model: ChannelAxis

  protected _tick_label_extents(): number[] { return [52] }

  protected _draw_major_labels(ctx: Context2d, extents: Extents, ticks: TickCoords): void {
    const [xs, ys] = this.scoords(ticks.major)
    const values = ticks.major[this.dimension]
    const labels = this.model.major_label_overrides
    const side = this.normals[0]
    const frame = this.plot_view.frame.bbox
    ctx.save()
    this.visuals.major_label_text.set_value(ctx)
    ctx.textBaseline = "middle"
    const threshold = ctx.measureText("A888").width
    const candidates: Label[] = []
    for (let i = 0; i < values.length; i++) {
      const text = labels instanceof Map ? (labels as Map<number | string, unknown>).get(values[i]) : labels[values[i]]
      if (typeof text !== "string") continue
      let gap = frame.height
      for (let j = 0; j < ys.length; j++) {
        if (j !== i) gap = Math.min(gap, Math.abs(ys[i] - ys[j]))
      }
      // Dense overview ticks still get candidates. Select among their actual
      // text bounds below rather than dropping every crowded label.
      const room = this.model.avoid_overlap ? Math.max(12, gap - 4) : gap - 4
      if (room < 12) continue
      const rotated = !this.model.truncate_labels && this.model.channel_labels.has(values[i]) && ctx.measureText(text).width > threshold
      const lines: string[] = []
      if (!rotated) {
        let line = text
        if (this.model.truncate_labels && ctx.measureText(line).width > 48) {
          const chars = Array.from(line)
          while (chars.length > 0 && ctx.measureText(chars.join("") + "...").width > 48) chars.pop()
          line = chars.join("") + "..."
        }
        lines.push(line)
      } else {
        const rest = Array.from(text)
        while (rest.length > 0 && lines.length < 3) {
          let line = ""
          while (rest.length > 0 && ctx.measureText(line + rest[0]).width <= room) line += rest.shift()
          if (lines.length == 2 && rest.length > 0) {
            while (line.length > 0 && ctx.measureText(line + "...").width > room) line = line.slice(0, -1)
            line += "..."
          }
          lines.push(line)
          if (line == "") break
        }
      }
      const height = rotated ? Math.max(...lines.map(line => ctx.measureText(line).width)) : 12
      // Endpoint labels can be closer to the canvas edge than half a glyph.
      const y = this.model.avoid_overlap
        ? Math.max(frame.top + height/2, Math.min(frame.bottom - height/2, ys[i])) : ys[i]
      candidates.push({index: i, y, height, lines, rotated})
    }
    let selected = candidates
    if (this.model.avoid_overlap && candidates.length > 0) {
      const ordered = candidates.slice().sort((a, b) => ys[a.index] - ys[b.index])
      const channels = ordered.filter(label => this.model.channel_labels.has(values[label.index]))
      const endpoints = channels.length > 0 ? channels : ordered
      selected = []
      const fits = (label: Label) => selected.every(other =>
        Math.abs(label.y - other.y) >= (label.height + other.height)/2 + 4)
      for (const label of [endpoints[0], endpoints[endpoints.length - 1]]) {
        if (fits(label)) selected.push(label)
      }
      // Earliest finishing intervals maximize the number of remaining labels
      // around the reserved first and last channel names.
      for (const label of ordered.slice().sort((a, b) => (a.y + a.height/2) - (b.y + b.height/2))) {
        if (fits(label)) selected.push(label)
      }
    }
    for (const label of selected) {
      ctx.save()
      const x = xs[label.index] + side * (extents.tick + 5)
      ctx.translate(x + (label.rotated ? side * 24 : 0), label.y)
      if (label.rotated) {
        ctx.textAlign = "center"
        ctx.rotate(-Math.PI / 2)
        label.lines.forEach((line, j) => ctx.fillText(line, 0, (j - (label.lines.length - 1)/2)*14))
      } else {
        ctx.textAlign = side < 0 ? "right" : "left"
        ctx.fillText(label.lines[0], 0, 0, 48)
      }
      ctx.restore()
    }
    ctx.restore()
  }
}
export namespace ChannelAxis {
  export type Attrs = p.AttrsOf<Props>
  export type Props = LinearAxis.Props & {
    channel_labels: p.Property<Map<number, string>>
    avoid_overlap: p.Property<boolean>
    truncate_labels: p.Property<boolean>
  }
}
export interface ChannelAxis { channel_labels: Map<number, string>, avoid_overlap: boolean, truncate_labels: boolean }
export class ChannelAxis extends LinearAxis {
  declare properties: ChannelAxis.Props
  static __module__ = "venn_ts.channel_axis"
  static {
    this.prototype.default_view = ChannelAxisView
    this.define<ChannelAxis.Props>(({Mapping, Float, Str, Bool}) => ({
      channel_labels: [Mapping(Float, Str), new Map()],
      avoid_overlap: [Bool, false],
      truncate_labels: [Bool, false],
    }))
  }
}
