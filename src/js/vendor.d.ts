// These packages ship ESM builds with no type declarations. Only the shapes
// this bundle touches are declared; extend as needed if they get used more.
declare module "alpinejs" {
    const Alpine: { start(): void }
    export default Alpine
}

declare module "htmx-ext-sse"
declare module "htmx-ext-ws"

declare module "tabulator-tables" {
    const TabulatorFull: unknown
    export { TabulatorFull }
}
