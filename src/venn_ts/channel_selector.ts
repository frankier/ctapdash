import {Widget, WidgetView} from "@bokehjs/models/widgets/widget"
import type * as p from "@bokehjs/core/properties"

interface SelectorElement extends HTMLElement {
    configure(metadata: unknown, available: string[], bads: unknown): void
    selectedNames: string[]
}

export class ChannelSelectorView extends WidgetView {
    declare model: ChannelSelector
    private selector?: SelectorElement
    private timer?: ReturnType<typeof setTimeout>

    async lazy_initialize(): Promise<void> {
        await super.lazy_initialize()
        await customElements.whenDefined("channel-selector")
    }
    connect_signals(): void {
        super.connect_signals()
        const {metadata_json, available, bads_json, value} = this.model.properties
        this.on_change([metadata_json, available, bads_json], () => this.configure())
        this.on_change(value, () => {
            if (this.selector) this.selector.selectedNames = this.model.value
        })
    }
    private configure(): void {
        this.selector?.configure(
            JSON.parse(this.model.metadata_json),
            this.model.available,
            JSON.parse(this.model.bads_json),
        )
    }
    render(): void {
        super.render()
        this.selector = document.createElement("channel-selector") as SelectorElement
        this.configure()
        this.shadow_el.append(this.selector)
        this.selector.addEventListener("channel-selection-change", () => this.publish())
        this.publish() // Restore tab selection into the Bokeh document on first mount.
    }
    private publish(): void {
        clearTimeout(this.timer)
        this.timer = setTimeout(() => {
            if (!this.selector) return
            const names = this.selector.selectedNames
            if (JSON.stringify(names) !== JSON.stringify(this.model.value)) this.model.value = names
        }, 100)
    }
    remove(): void {
        clearTimeout(this.timer)
        this.selector?.remove()
        super.remove()
    }
}
export namespace ChannelSelector {
    export type Attrs = p.AttrsOf<Props>
    export type Props = Widget.Props & {
        metadata_json: p.Property<string>
        available: p.Property<string[]>
        bads_json: p.Property<string>
        value: p.Property<string[]>
    }
}
export interface ChannelSelector extends ChannelSelector.Attrs {}
export class ChannelSelector extends Widget {
    static __module__ = "venn_ts.channel_selector"
    declare properties: ChannelSelector.Props
    declare __view_type__: ChannelSelectorView
    static {
        this.prototype.default_view = ChannelSelectorView
        this.define<ChannelSelector.Props>(({Str, List}) => ({
            metadata_json: [Str, "{}"],
            available: [List(Str), []],
            bads_json: [Str, "{}"],
            value: [List(Str), []],
        }))
    }
}
