# Vendored frontend dependencies

These were previously loaded from CDNs. They are checked in so the packaged
desktop app works offline and so builds are reproducible.

To refresh, re-download from the URL below, update the version and checksum
here, and check the result actually still works — none of these are pinned by
a lockfile.

| File | Version | Source | SHA-256 |
|---|---|---|---|
| `tabulator.min.css` | 6.3.1 | `https://unpkg.com/tabulator-tables@6.3.1/dist/css/tabulator.min.css` | `a46d8051944c745cae8a7976b4fb9d93d894d20876a4521cc4f6f035cfef52ea` |
| `tabulator.min.js` | 6.3.1 | `https://unpkg.com/tabulator-tables@6.3.1/dist/js/tabulator.min.js` | `e952272c3b2afa4ebb60cef5db8cbe9cbaabaa52b50c3cd3d22993ca5215a6ff` |
| `tailwind-browser.js` | 4 | `https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4` | `6d8c473ef2f8ad63feafc0bd76502dda31501a6c135dc4c6173f6268cde595be` |
| `htmx.min.js` | 2.0.7 | `https://cdn.jsdelivr.net/npm/htmx.org@2.0.7/dist/htmx.min.js` | `60231ae6ba9db3825eb15a261122d5f55921c4d53b66bf637dc18b4ee27c79f9` |
| `htmx-ext-sse.js` | 2.2.2 | `https://cdn.jsdelivr.net/npm/htmx-ext-sse@2.2.2` | `b32dacd8e5bdd41a0223b5c56374fd76e4c4377e3f3a3bfe873bb41cbc7585eb` |
| `alpine.min.js` | 3.15.0 | `https://cdn.jsdelivr.net/npm/alpinejs@3.15.0/dist/cdn.min.js` | `e041f1b639d1e6b2fc2736d8d7638a409afcd444a6ec90446f8f4e44fa36f406` |

`tailwind-browser.js` compiles utility classes in the browser on every page
load. That works offline, which is what the packaged app needs, but it costs
~280 KB of JS and a brief flash of unstyled content. Replacing it with a
build-time `tailwindcss` CLI step would fix both, at the cost of adding a Node
toolchain to CI.
