import {Widget, WidgetView} from "@bokehjs/models/widgets/widget"
import {Tooltip, TooltipView} from "@bokehjs/models/ui/tooltip"
import {BuiltinIcon, BuiltinIconView} from "@bokehjs/models/ui/icons/builtin_icon"
import {build_view} from "@bokehjs/core/build_views"
import type * as p from "@bokehjs/core/properties"

class AdjacentTooltipView extends TooltipView {
    protected override _reposition(): void {
        // Bokeh computes viewport coordinates but moves tooltips to body on
        // every update. Restore this tooltip beside its trigger afterwards.
        super._reposition()
        this.target.after(this.el)
        if (this.model.visible && this.model.position != null) {
            // Moving a popover closes it; reopen it in the top layer so fixed
            // positioning still works inside Bokeh's contained layouts.
            this.el.showPopover?.()
        }
    }
}

export class HelpIconView extends WidgetView {
    declare model: HelpIcon
    private tooltip_view: TooltipView
    private icon_view: BuiltinIconView

    override children_views() {
        return [...super.children_views(), this.tooltip_view, this.icon_view]
    }

    override async lazy_initialize(): Promise<void> {
        await super.lazy_initialize()
        this.tooltip_view = await build_view(
            this.model.tooltip,
            {parent: this},
            () => AdjacentTooltipView,
        )
        this.icon_view = await build_view(new BuiltinIcon({icon_name: "help", size: 18}), {
            parent: this,
        })
    }

    override render(): void {
        super.render()
        const button = document.createElement("button")
        button.type = "button"
        button.setAttribute("aria-label", "Help")
        button.style.cssText =
            "display:block;width:18px;height:18px;padding:0;border:0;background:transparent;color:inherit;cursor:help"
        this.icon_view.render()
        button.append(this.icon_view.el)
        this.shadow_el.append(button)
        this.tooltip_view.target = button
        let hovered = false
        let focused = false
        const update = () => (this.model.tooltip.visible = hovered || focused)
        button.addEventListener("mouseenter", () => {
            hovered = true
            update()
        })
        button.addEventListener("mouseleave", () => {
            hovered = false
            update()
        })
        button.addEventListener("focus", () => {
            focused = true
            update()
        })
        button.addEventListener("blur", () => {
            focused = false
            update()
        })
        button.addEventListener("keydown", (event) => {
            if (event.key === "Escape") this.model.tooltip.visible = false
        })
        this.tooltip_view.render()
        this.tooltip_view.r_after_render()
    }

    override remove(): void {
        this.icon_view.remove()
        this.tooltip_view.remove()
        super.remove()
    }
}

export namespace HelpIcon {
    export type Attrs = p.AttrsOf<Props>
    export type Props = Widget.Props & {tooltip: p.Property<Tooltip>}
}
export interface HelpIcon extends HelpIcon.Attrs {}
export class HelpIcon extends Widget {
    static __module__ = "venn_ts.help_icon"
    declare properties: HelpIcon.Props
    declare __view_type__: HelpIconView
    static {
        this.prototype.default_view = HelpIconView
        this.define<HelpIcon.Props>(({Ref}) => ({tooltip: [Ref(Tooltip)]}))
    }
}
