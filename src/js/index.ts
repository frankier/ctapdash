// Browser bundle entry point. The templates load nothing but this module and
// the Tailwind output; every frontend dependency is an import here so esbuild
// resolves and version-pins them from package-lock.json instead of the
// hand-downloaded copies that used to live in static/vendor.
import Alpine from "alpinejs"
import htmx from "htmx.org"
import "htmx-ext-sse"
import "htmx-ext-ws"
import {TabulatorFull} from "tabulator-tables"

// Registers <channel-selector> and <heatmap-channels>, which the statistics
// and Venndiff pages use.
import "./channel-selector.js"
import "./heatmap-channels.js"

// The vendored builds exposed themselves as globals. Nothing in this repo
// relies on that, but htmx and Alpine are documented extension points and
// keeping the names avoids a surprise for anything added later.
declare global {
    interface Window {
        htmx: typeof htmx
        Alpine: typeof Alpine
        Tabulator: typeof TabulatorFull
    }
}

window.htmx = htmx
window.Alpine = Alpine
window.Tabulator = TabulatorFull

// The CDN build of Alpine started itself; the module build does not.
Alpine.start()

const settingsButton = document.querySelector<HTMLButtonElement>("#source-settings-button")
const settingsDialog = document.querySelector<HTMLDialogElement>("#source-settings-dialog")
const settingsContent = document.querySelector<HTMLElement>("#source-settings-content")
if (settingsButton && settingsDialog && settingsContent) {
    let changed = false
    settingsButton.addEventListener("click", async () => {
        await htmx.ajax("get", settingsButton.dataset.setupUrl!, {
            target: settingsContent,
            swap: "innerHTML",
        })
        settingsDialog.showModal()
    })
    document
        .querySelector("#source-settings-close")
        ?.addEventListener("click", () => settingsDialog.close())
    document.body.addEventListener("sources-changed", () => {
        changed = true
    })
    settingsDialog.addEventListener("close", () => {
        if (!changed) return
        const selected = document.querySelector<HTMLSelectElement>("#source-select")?.value
        const names = [
            ...settingsContent.querySelectorAll<HTMLTableCellElement>(
                "#setup tbody tr td:first-child",
            ),
        ].map((cell) => cell.textContent?.trim())
        if (selected && !names.includes(selected)) window.location.assign("/")
        else window.location.reload()
    })
}
