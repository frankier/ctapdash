export type Channel = {name: string; type: string}
export type Groups = Record<string, string[]>
export type Head = {points: Record<string, [number, number]>; outlines: number[][][]; message: string}
export type Metadata = {channels: Channel[]; groups: Groups; regions: Groups; head: Head; steps: Groups; bads: Groups; stateKey: string}

const stateEvent = "ctap-channel-state"
const storagePrefix = "ctap-channel-selection:"
const emptyHead: Head = {points: {}, outlines: [], message: "No usable sensor positions are available."}

function stored(key: string, names: string[]): string[] {
    try {
        const raw = sessionStorage.getItem(storagePrefix + key)
        if (raw) {
            const value = JSON.parse(raw)
            if (Array.isArray(value.names) && Array.isArray(value.known)) {
                return names.filter(n => value.names.includes(n) || !value.known.includes(n))
            }
        }
    } catch { /* Storage may be disabled; the component remains usable. */ }
    return names
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string): HTMLElementTagNameMap[K] {
    const node = document.createElement(tag)
    if (text !== undefined) node.textContent = text
    return node
}
function button(text: string, action: () => void): HTMLButtonElement {
    const node = element("button", text)
    node.type = "button"
    node.onclick = action
    return node
}
function inside(point: number[], polygon: number[][]): boolean {
    let yes = false
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const a = polygon[i], b = polygon[j]
        if ((a[1] > point[1]) !== (b[1] > point[1]) &&
            point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]) yes = !yes
    }
    return yes
}

export class ChannelSelector extends HTMLElement {
    private root = this.attachShadow({mode: "open"})
    private _channels: Channel[] = []
    private _groups: Groups = {}
    private regions: Groups = {}
    private _head: Head = emptyHead
    private _available = new Set<string>()
    private _selected = new Set<string>()
    private selectionInitialized = false
    private availabilityInitialized = false
    private _bads: Groups = {}
    private key = ""
    private query = ""
    private grouping = "type"
    private view = "list"
    private lassoMode = "select"
    private queued = false
    private controller?: AbortController
    unavailableReason = "Unavailable in the displayed processing steps"

    get channels(): Channel[] { return this._channels }
    set channels(value: Channel[]) {
        this._channels = value
        if (!this.selectionInitialized) this._selected = new Set(value.map(c => c.name))
        if (!this.availabilityInitialized) this._available = new Set(value.map(c => c.name))
        this.schedule()
    }
    get groups(): Groups { return this._groups }
    set groups(value: Groups) { this._groups = value; this.schedule() }
    get head(): Head { return this._head }
    set head(value: Head) { this._head = value; this.schedule() }
    get selectedNames(): string[] { return this._channels.map(c => c.name).filter(n => this._selected.has(n)) }
    set selectedNames(value: string[]) { this.selectionInitialized = true; this._selected = new Set(value); this.schedule() }
    get availableNames(): string[] { return [...this._available] }
    set availableNames(value: string[]) { this.availabilityInitialized = true; this._available = new Set(value); this.schedule() }
    get badsByStep(): Groups { return this._bads }
    set badsByStep(value: Groups) { this._bads = value; this.schedule() }

    configure(metadata: Metadata, available: string[], bads: Groups): void {
        const names = metadata.channels.map(c => c.name)
        if (this.key !== metadata.stateKey || !this._channels.length) {
            this._selected = new Set(stored(metadata.stateKey, names))
        } else {
            const known = new Set(this._channels.map(c => c.name))
            names.filter(n => !known.has(n)).forEach(n => this._selected.add(n))
        }
        this.selectionInitialized = true
        this.availabilityInitialized = true
        this.key = metadata.stateKey
        this._channels = metadata.channels
        this._groups = metadata.groups
        this.regions = metadata.regions
        this._head = metadata.head
        this._available = new Set(available)
        this._bads = bads
        this.schedule()
    }

    connectedCallback(): void {
        this.controller?.abort()
        this.controller = new AbortController()
        window.addEventListener(stateEvent, ((event: CustomEvent) => {
            if (event.detail.key === this.key && event.detail.source !== this) {
                this.selectedNames = event.detail.names
                this.emit()
            }
        }) as EventListener, {signal: this.controller.signal})
        this.schedule()
    }
    disconnectedCallback(): void { this.controller?.abort() }
    private schedule(): void {
        if (this.queued) return
        this.queued = true
        queueMicrotask(() => { this.queued = false; if (this.isConnected) this.render() })
    }
    private emit(): void {
        this.dispatchEvent(new CustomEvent("channel-selection-change", {
            bubbles: true, composed: true, detail: {selectedNames: this.selectedNames},
        }))
    }
    private change(names: string[], selected: boolean): void {
        let changed = false
        for (const name of names) {
            if (!this._available.has(name) || this._selected.has(name) === selected) continue
            changed = true
            if (selected) this._selected.add(name)
            else this._selected.delete(name)
        }
        if (!changed) return
        this.selectionInitialized = true
        const selectedNames = this.selectedNames
        try {
            sessionStorage.setItem(storagePrefix + this.key,
                JSON.stringify({names: selectedNames, known: this._channels.map(c => c.name)}))
        } catch { /* In-memory selection still works. */ }
        window.dispatchEvent(new CustomEvent(stateEvent, {detail: {key: this.key, names: selectedNames, source: this}}))
        this.schedule()
        this.emit()
    }
    private badDescription(name: string): string {
        const steps = Object.entries(this._bads).filter(([, names]) => names.includes(name)).map(([step]) => step)
        return steps.length ? `Bad in step${steps.length > 1 ? "s" : ""} ${steps.join(", ")}` : ""
    }
    private label(name: string): string {
        return [name, this.badDescription(name), this._available.has(name) ? "" : this.unavailableReason].filter(Boolean).join(" — ")
    }
    private filtered(): Channel[] {
        const search = this.query.toLocaleLowerCase()
        return this._channels.filter(c => c.name.toLocaleLowerCase().includes(search))
    }
    private render(): void {
        const scrollTop = this.root.querySelector(".list")?.scrollTop ?? 0
        const active = this.root.activeElement as HTMLInputElement | null
        const focusKey = active?.dataset.focus
        const cursor = active?.type === "search" ? active.selectionStart : null
        this.root.replaceChildren()
        const style = element("style")
        style.textContent = `
            :host{display:block;width:560px;max-width:100%;font:14px system-ui;color:#172033;background:white;box-sizing:border-box}
            *{box-sizing:border-box} button,input,select{font:inherit} button,select,input[type=search]{border:1px solid #94a3b8;border-radius:5px;padding:5px 9px;background:white;color:inherit}
            button{cursor:pointer} button:hover{background:#eef2ff} button[aria-selected=true]{background:#dbeafe;border-color:#2563eb}
            button:focus-visible,input:focus-visible,select:focus-visible,[role=checkbox]:focus-visible{outline:3px solid #2563eb;outline-offset:2px}
            .toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}.list{max-height:350px;overflow:auto}
            fieldset{border:1px solid #cbd5e1;border-radius:5px;margin:8px 0;padding:8px} legend{font-weight:600}
            .channel{display:inline-flex;align-items:center;gap:5px;margin:4px 10px 4px 0;min-width:110px}.unavailable{color:#64748b}.bad{color:#9a3412;font-size:12px}
            svg{width:100%;max-height:410px;touch-action:none} .sensor{cursor:pointer}.sensor[aria-disabled=true]{opacity:.35;cursor:default}
            .sensor circle{fill:white;stroke:#475569;stroke-width:.001}.sensor[aria-checked=true] circle{fill:#2563eb;stroke:#1e40af}
            .sensor text{font-size:.006px;fill:#172033}.sensor.bad circle{stroke:#c2410c;stroke-width:.002}
            p{margin:8px 0;color:#475569} [hidden]{display:none!important}
        `
        this.root.append(style)
        const tabs = element("div"); tabs.className = "toolbar"; tabs.setAttribute("role", "tablist"); tabs.setAttribute("aria-label", "Channel selection view")
        for (const [id, name] of [["list", "Grouped list"], ["head", "Head diagram"]]) {
            const tab = button(name, () => { this.view = id; this.schedule() })
            tab.id = `tab-${id}`; tab.dataset.focus = tab.id
            tab.setAttribute("role", "tab"); tab.setAttribute("aria-selected", String(this.view === id)); tab.setAttribute("aria-controls", "panel")
            tab.tabIndex = this.view === id ? 0 : -1
            tab.onkeydown = event => {
                if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
                    event.preventDefault(); this.view = event.key === "Home" ? "list" : event.key === "End" ? "head" : this.view === "list" ? "head" : "list"
                    this.render(); this.root.querySelector<HTMLButtonElement>(`#tab-${this.view}`)?.focus()
                }
            }
            tabs.append(tab)
        }
        this.root.append(tabs)
        const count = element("p", `${this.selectedNames.filter(n => this._available.has(n)).length} / ${this._available.size} available channels selected`)
        count.setAttribute("role", "status"); this.root.append(count)
        const panel = element("div"); panel.id = "panel"; panel.setAttribute("role", "tabpanel"); panel.setAttribute("aria-labelledby", `tab-${this.view}`)
        this.root.append(panel)
        if (this.view === "list") this.renderList(panel)
        else this.renderHead(panel)
        const list = this.root.querySelector(".list")
        if (list) list.scrollTop = scrollTop
        if (focusKey) {
            const replacement = [...this.root.querySelectorAll<HTMLElement>("[data-focus]")].find(n => n.dataset.focus === focusKey)
            replacement?.focus({preventScroll: true})
            if (cursor !== null && replacement instanceof HTMLInputElement && replacement.type === "search") replacement.setSelectionRange(cursor, cursor)
        }
    }
    private renderList(panel: HTMLElement): void {
        const toolbar = element("div"); toolbar.className = "toolbar"
        const search = element("input"); search.type = "search"; search.placeholder = "Search channels"; search.setAttribute("aria-label", "Search channels"); search.dataset.focus = "search"; search.value = this.query
        search.oninput = () => { this.query = search.value; this.schedule() }
        const grouping = element("select"); grouping.setAttribute("aria-label", "Group channels by"); grouping.dataset.focus = "grouping"
        for (const [value, text] of [["type", "Channel type"], ["region", "Head region"], ["custom", "Custom groups"]]) {
            if (value === "custom" && !Object.keys(this._groups).length) continue
            const option = element("option", text); option.value = value; grouping.append(option)
        }
        grouping.value = this.grouping; grouping.onchange = () => { this.grouping = grouping.value; this.schedule() }
        toolbar.append(search, grouping,
            button("Select all", () => this.change(this.filtered().map(c => c.name), true)),
            button("Clear", () => this.change(this.filtered().map(c => c.name), false)))
        panel.append(toolbar)
        const filtered = this.filtered(); const matching = new Set(filtered.map(c => c.name))
        const groups: Groups = Object.create(null)
        if (this.grouping === "type") {
            for (const ch of filtered) (groups[ch.type] ??= []).push(ch.name)
        } else {
            const source = this.grouping === "region" ? this.regions : this._groups
            const assigned = new Set(Object.values(source).flat())
            for (const [name, names] of Object.entries(source)) groups[name] = names.filter(n => matching.has(n))
            groups["Ungrouped"] = filtered.map(c => c.name).filter(n => !assigned.has(n))
        }
        const list = element("div"); list.className = "list"; panel.append(list)
        if (!filtered.length) list.append(element("p", "No matching channels."))
        for (const [group, names] of Object.entries(groups)) {
            if (!names.length) continue
            const field = element("fieldset"), legend = element("legend"), label = element("label"), checkbox = element("input")
            const available = names.filter(n => this._available.has(n)), selected = available.filter(n => this._selected.has(n))
            checkbox.type = "checkbox"; checkbox.checked = available.length > 0 && selected.length === available.length
            checkbox.indeterminate = selected.length > 0 && selected.length < available.length
            checkbox.disabled = !available.length; checkbox.dataset.focus = `group:${group}`
            checkbox.onchange = () => this.change(available, checkbox.checked)
            label.append(checkbox, document.createTextNode(` ${group} (${selected.length}/${available.length})`)); legend.append(label); field.append(legend)
            for (const name of names) {
                const row = element("label"); row.className = "channel" + (this._available.has(name) ? "" : " unavailable"); row.title = this.label(name)
                const input = element("input"); input.type = "checkbox"; input.checked = this._selected.has(name); input.disabled = !this._available.has(name)
                input.dataset.focus = `channel:${name}`; input.setAttribute("aria-label", this.label(name)); input.onchange = () => this.change([name], input.checked)
                row.append(input, document.createTextNode(name))
                const bad = this.badDescription(name)
                if (bad) { const badge = element("span", "⚠"); badge.className = "bad"; badge.setAttribute("aria-label", bad); row.append(badge) }
                field.append(row)
            }
            list.append(field)
        }
    }
    private renderHead(panel: HTMLElement): void {
        if (!Object.keys(this._head.points).length) { panel.append(element("p", this._head.message)); return }
        const toolbar = element("div"); toolbar.className = "toolbar"
        const mode = element("select"); mode.setAttribute("aria-label", "Lasso action"); mode.dataset.focus = "lasso-mode"
        for (const value of ["select", "deselect"]) { const option = element("option", `${value === "select" ? "Select" : "Deselect"} enclosed sensors`); option.value = value; mode.append(option) }
        mode.value = this.lassoMode; mode.onchange = () => { this.lassoMode = mode.value }
        toolbar.append(mode); panel.append(toolbar, element("p", "Click a sensor to toggle it. Drag around sensors to select or deselect them. Channels without positions remain in the list."))
        const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg")
        svg.setAttribute("aria-label", "Head sensor positions; nose at top"); svg.setAttribute("role", "group")
        const all = [...Object.values(this._head.points), ...this._head.outlines.flat()]
        const xs = all.map(p => p[0]), ys = all.map(p => p[1]), margin = .018
        const x = Math.min(...xs) - margin, y = Math.min(...ys) - margin
        svg.setAttribute("viewBox", `${x} ${y} ${Math.max(...xs) - x + margin} ${Math.max(...ys) - y + margin}`)
        for (const points of this._head.outlines) {
            const line = document.createElementNS(ns, "polyline"); line.setAttribute("points", points.map(p => p.join(",")).join(" "))
            line.setAttribute("fill", "none"); line.setAttribute("stroke", "#64748b"); line.setAttribute("stroke-width", ".001"); svg.append(line)
        }
        for (const [name, [px, py]] of Object.entries(this._head.points)) {
            const sensor = document.createElementNS(ns, "g"); sensor.classList.add("sensor"); if (this.badDescription(name)) sensor.classList.add("bad")
            sensor.setAttribute("role", "checkbox"); sensor.setAttribute("aria-label", this.label(name)); sensor.setAttribute("aria-checked", String(this._selected.has(name))); sensor.setAttribute("aria-disabled", String(!this._available.has(name)))
            sensor.setAttribute("tabindex", this._available.has(name) ? "0" : "-1"); sensor.dataset.focus = `sensor:${name}`
            const title = document.createElementNS(ns, "title"); title.textContent = this.label(name)
            const circle = document.createElementNS(ns, "circle"); circle.setAttribute("cx", String(px)); circle.setAttribute("cy", String(py)); circle.setAttribute("r", ".003")
            const text = document.createElementNS(ns, "text"); text.setAttribute("x", String(px + .004)); text.setAttribute("y", String(py)); text.textContent = name + (this.badDescription(name) ? " ⚠" : "")
            sensor.append(title, circle, text); sensor.onclick = () => this.change([name], !this._selected.has(name))
            sensor.onkeydown = event => { if ([" ", "Enter"].includes(event.key)) { event.preventDefault(); this.change([name], !this._selected.has(name)) } }
            svg.append(sensor)
        }
        let polygon: number[][] | null = null
        const trace = document.createElementNS(ns, "polyline"); trace.setAttribute("fill", "#2563eb22"); trace.setAttribute("stroke", "#2563eb"); trace.setAttribute("stroke-width", ".001"); trace.setAttribute("pointer-events", "none"); svg.append(trace)
        const point = (event: PointerEvent): number[] => { const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM()!.inverse()); return [p.x, p.y] }
        svg.onpointerdown = event => {
            if (event.button !== 0 || (event.target as Element).closest(".sensor")) return
            event.preventDefault(); svg.setPointerCapture(event.pointerId); polygon = [point(event)]
        }
        svg.onpointermove = event => { if (polygon) { polygon.push(point(event)); trace.setAttribute("points", polygon.map(p => p.join(",")).join(" ")) } }
        svg.onpointerup = event => {
            if (!polygon) return
            const enclosed = polygon.length >= 3 ? Object.entries(this._head.points).filter(([, p]) => inside(p, polygon!)).map(([name]) => name) : []
            polygon = null; trace.setAttribute("points", ""); svg.releasePointerCapture(event.pointerId); this.change(enclosed, this.lassoMode === "select")
        }
        svg.onpointercancel = () => { polygon = null; trace.setAttribute("points", "") }
        panel.append(svg)
    }
}
if (!customElements.get("channel-selector")) customElements.define("channel-selector", ChannelSelector)
