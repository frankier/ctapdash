// These packages ship ESM builds with no type declarations. Only the shapes
// this bundle touches are declared; extend as needed if they get used more.
declare module "alpinejs" {
    const Alpine: {start(): void}
    export default Alpine
}

declare module "htmx-ext-sse"
declare module "htmx-ext-ws"

declare module "tabulator-tables" {
    const TabulatorFull: new (
        element: HTMLElement,
        options: Record<string, unknown>,
    ) => {
        setHeight(height: number): void
        getPage(): number
        getPageSize(): number
        setPageSize(size: number): void
        on(event: "tableBuilt", callback: () => void): void
        on(event: "pageLoaded", callback: (page: number) => void): void
    }
    export {TabulatorFull}
}
