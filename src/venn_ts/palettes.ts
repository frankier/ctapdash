import {Widget, WidgetView} from "@bokehjs/models/widgets/widget"
import type * as p from "@bokehjs/core/properties"

export class PaletteSelectorView extends WidgetView {
    declare model: PaletteSelector
    connect_signals(): void {
        super.connect_signals()
        this.on_change(this.model.properties.value, () => this.render())
    }
    render(): void {
        super.render()
        const style = document.createElement("style")
        style.textContent = `
            :host {overflow: visible; position: relative;}
            button {display:flex; align-items:center; gap:10px; width:100%; padding:4px 8px;
                cursor:pointer; border:1px solid #888; border-radius:4px; background:#fff; color:#222;}
            button:hover, button:focus-visible {outline:2px solid #888;}
            img {width:51px; height:47px;}
            .menu {position:absolute; z-index:100; width:100%; background:white; box-shadow:0 3px 8px #777;}
            .menu[hidden] {display:none;}
            .label {margin-bottom:4px;}
        `
        const label = document.createElement("div")
        label.className = "label"
        label.textContent = "Palette"
        const options: {name: string; icon: string}[] = JSON.parse(this.model.options_json)
        const make_button = (option: {name: string; icon: string}) => {
            const button = document.createElement("button")
            button.type = "button"
            const image = document.createElement("img")
            image.src = option.icon
            image.alt = ""
            button.append(image, document.createTextNode(option.name))
            return button
        }
        const current = make_button(options.find((o) => o.name == this.model.value) ?? options[0])
        current.setAttribute("aria-label", `Palette: ${this.model.value}`)
        current.setAttribute("aria-expanded", "false")
        current.setAttribute("aria-haspopup", "menu")
        const menu = document.createElement("div")
        menu.className = "menu"
        menu.role = "menu"
        menu.hidden = true
        current.onclick = () => {
            menu.hidden = !menu.hidden
            current.setAttribute("aria-expanded", String(!menu.hidden))
            if (!menu.hidden) (menu.firstElementChild as HTMLElement)?.focus()
        }
        for (const option of options) {
            const button = make_button(option)
            button.role = "menuitemradio"
            button.setAttribute("aria-checked", String(option.name == this.model.value))
            button.onclick = () => {
                menu.hidden = true
                current.setAttribute("aria-expanded", "false")
                this.model.value = option.name
                this.shadow_el.querySelector<HTMLButtonElement>("button")?.focus()
            }
            menu.append(button)
        }
        menu.onkeydown = (event) => {
            const buttons = [...menu.querySelectorAll("button")]
            const index = buttons.indexOf(this.shadow_el.activeElement as HTMLButtonElement)
            if (event.key == "Escape") {
                menu.hidden = true
                current.setAttribute("aria-expanded", "false")
                current.focus()
            } else if (event.key == "ArrowDown" || event.key == "ArrowUp") {
                event.preventDefault()
                buttons[
                    (index + (event.key == "ArrowDown" ? 1 : buttons.length - 1)) % buttons.length
                ].focus()
            }
        }
        this.shadow_el.append(style, label, current, menu)
    }
}
export namespace PaletteSelector {
    export type Attrs = p.AttrsOf<Props>
    export type Props = Widget.Props & {value: p.Property<string>; options_json: p.Property<string>}
}
export interface PaletteSelector extends PaletteSelector.Attrs {}
export class PaletteSelector extends Widget {
    static __module__ = "venn_ts.palettes"
    declare properties: PaletteSelector.Props
    declare __view_type__: PaletteSelectorView
    static {
        this.prototype.default_view = PaletteSelectorView
        this.define<PaletteSelector.Props>(({Str}) => ({
            value: [Str, "rbg_add"],
            options_json: [Str, "[]"],
        }))
    }
}
