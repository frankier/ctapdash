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

const logTable = document.querySelector<HTMLElement>("#log-table")
const logTableData = document.querySelector<HTMLScriptElement>("#log-table-data")
if (logTable && logTableData) {
    const {columns, rows} = JSON.parse(logTableData.textContent || "{}") as {
        columns: string[]
        rows: string[][]
    }
    const availableHeight = () =>
        Math.max(1, window.innerHeight - logTable.getBoundingClientRect().top)
    const table = new TabulatorFull(logTable, {
        data: rows.map((row) =>
            Object.fromEntries(row.map((value, index) => [`col${index}`, value])),
        ),
        columns: columns.map((title, index) => {
            const heading = document.createElement("span")
            heading.textContent = title
            return {
                title: heading.innerHTML,
                field: `col${index}`,
                formatter: "plaintext",
            }
        }),
        layout: "fitData",
        height: availableHeight(),
        columnDefaults: {maxWidth: 320, tooltip: true},
        pagination: true,
        paginationCounter: "rows",
    })
    let ready = false
    let fitting = false
    const fitFirstPage = () => {
        if (!ready || fitting || table.getPage() !== 1 || !logTable.getClientRects().length) return
        fitting = true
        try {
            table.setHeight(availableHeight())
            const holder = logTable.querySelector<HTMLElement>(".tabulator-tableholder")
            const row = logTable.querySelector<HTMLElement>(".tabulator-row")
            if (holder && row?.offsetHeight) {
                const pageSize = Math.max(1, Math.floor(holder.clientHeight / row.offsetHeight))
                if (pageSize !== table.getPageSize()) table.setPageSize(pageSize)
            }
        } finally {
            fitting = false
        }
    }
    table.on("tableBuilt", () => {
        ready = true
        fitFirstPage()
    })
    table.on("pageLoaded", (page) => {
        if (page === 1) fitFirstPage()
    })
    window.addEventListener("resize", () => {
        fitFirstPage()
    })
    document.querySelector("#log-table-tab")?.addEventListener("click", () => {
        requestAnimationFrame(fitFirstPage)
    })
}

const settingsButton = document.querySelector<HTMLAnchorElement>("#source-settings-button")
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
