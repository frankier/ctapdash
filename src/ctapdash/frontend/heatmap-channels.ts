import "./channel-selector.js"
import type {ChannelSelector} from "./channel-selector.js"
import type {Metadata, Groups} from "./channel-selector.js"

class HeatmapChannels extends HTMLElement {
    private controller?: AbortController
    connectedCallback(): void {
        this.controller?.abort()
        this.controller = new AbortController()
        const signal = this.controller.signal
        const metadata = JSON.parse(this.querySelector('script[type="application/json"]')!.textContent!) as Metadata
        const selector = this.querySelector<ChannelSelector>("channel-selector")!
        const input = this.querySelector<HTMLInputElement>('input[name="channels"]')!
        const step = document.querySelector<HTMLSelectElement>("#statistics-step")!
        const dialog = this.querySelector<HTMLDialogElement>("dialog")!
        const update = () => {
            const steps = step.value === "all" ? Object.keys(metadata.steps) : [step.value]
            const available = [...new Set(steps.flatMap(s => metadata.steps[s] ?? []))]
            const bads: Groups = Object.fromEntries(steps.map(s => [s, metadata.bads[s] ?? []]))
            selector.configure(metadata, available, bads)
            input.value = JSON.stringify(selector.selectedNames.filter(name => selector.availableNames.includes(name)))
        }
        update()
        step.addEventListener("change", update, {signal, capture: true})
        this.querySelector<HTMLButtonElement>('[data-open]')!.addEventListener("click", () => dialog.showModal(), {signal})
        this.querySelector<HTMLButtonElement>('[data-close]')!.addEventListener("click", () => dialog.close(), {signal})
        selector.addEventListener("channel-selection-change", () => {
            input.value = JSON.stringify(selector.selectedNames.filter(name => selector.availableNames.includes(name)))
            document.querySelector("#channel-statistics-table")?.dispatchEvent(new Event("channels-changed", {bubbles: true}))
        }, {signal})
    }
    disconnectedCallback(): void { this.controller?.abort() }
}
if (!customElements.get("heatmap-channels")) customElements.define("heatmap-channels", HeatmapChannels)
