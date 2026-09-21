# CTAP Dashboard

A dashboard for viewing the outputs of [CTAP](https://version.helsinki.fi/hipercog/Methods/ctap) pipelines.

## Installing

Download the build for your platform from the
[latest release](../../releases/latest), unpack it, and run `ctapdash`
(`CTAP Dashboard.app` on macOS).

The builds are not code-signed, so:

- **macOS**: Gatekeeper will refuse to open a downloaded app. Clear the
  quarantine flag once, after moving it to `/Applications`:
  ```bash
  xattr -dr com.apple.quarantine "/Applications/CTAP Dashboard.app"
  ```
- **Windows**: SmartScreen shows "Windows protected your PC". Choose
  _More info_ → _Run anyway_.
  Keep `ctapdash.exe.config` beside `ctapdash.exe`: it lets the native window
  load its bundled .NET assemblies after extraction from a downloaded ZIP.
  For older builds reporting `Failed to resolve Python.Runtime.Loader.Initialize`,
  right-click the downloaded ZIP, choose _Properties_ → _Unblock_ → _Apply_,
  then extract it again into a fresh directory.

On Windows and macOS the dashboard opens in its own window. On Linux it starts
a local server and opens your usual browser, because the native window there
would need a system WebKitGTK installation.

## Configuration

By default, simply opening the program shows a setup page where you can set out your CTAP output directories.
Sources added that way last only for that run — use _Save configuration_ to write them to a file you can pass with `--config` next time.

---

# Advanced usage

## Command line usage

You can also write a configuration file manually, e.g. `conf.toml`:

```toml
[sources]
source_name = "/path/to/TAPPED"
```

Then either pass it on the command line:

```bash
ctapdash --config /path/to/conf.toml
```

or set `CTAPDASH_SETTINGS=/path/to/conf.toml` in the environment.

| Option         | Effect                                             |
| -------------- | -------------------------------------------------- |
| `--port N`     | Serve on a fixed port instead of a free one        |
| `--no-window`  | Serve only; don't open a window or a browser       |
| `--no-browser` | Don't open a browser                               |
| `--reload`     | Restart on source changes (browser only)           |
| `--debug`      | Show tracebacks in the browser, implies `--reload` |

Freshness is checked at registration using `.set` mtimes. External `.fdt` changes
and removed recordings are not detected. Re-add an existing source in Setup to
check it again after editing `.set` files; there is no filesystem watcher.

`ctapdash-warm --config conf.toml` warms only the metadata pickle caches.

## Developing

```bash
uv sync --group build
npm ci
uv run python -m ctapdash.build
uv run ctapdash --config conf.toml
```

`--debug` implies `--reload`, so `uv run ctapdash --config conf.toml --debug`
restarts the server whenever you edit a source file. Python changes restart the
server directly; edits to the TypeScript sources of the Bokeh extension and the
shared browser components restart the server and recompile their bundles first,
so `src/venn_ts` is picked up without a manual `ctapdash.build` run. Both options
serve in the browser: uvicorn's reloader runs the application in a subprocess,
which cannot own the main thread as the native window needs. `--smoke-test`
turns reload off.

`uv run uvicorn ctapdash.webapp:create_app --factory` also works if you want a
plain ASGI server. Build browser assets first when starting through ASGI directly.
Node.js and npm are needed during development and packaging. `ctapdash.build`
bundles `src/js` with esbuild, compiles `src/css` with the Tailwind CLI, and
invokes `venn_ts.build` for the Bokeh extension. The libraries the pages need
(htmx and its extensions, Alpine, Tabulator, Tailwind) are npm dependencies
compiled into `static/generated/index.js` and `static/generated/index.css`;
nothing is fetched at runtime. Normal CLI startup rebuilds changed sources;
frozen builds ship the compiled assets and skip compilation. `npm run watch`
rebuilds on change and `npm run check` type-checks the components.

### Building a release locally

```bash
uv run pyinstaller ctapdash.spec --noconfirm --clean
./dist/ctapdash/ctapdash --smoke-test
```

`--smoke-test` starts the server, requests the pages and static trees, and
forces the lazily-imported MNE and matplotlib code paths. It is the check that
catches PyInstaller problems, since those are runtime import failures rather
than build failures. It works unfrozen too (`uv run ctapdash --smoke-test`),
so you can compare the two directly when something breaks only in the bundle.

CI builds all four targets on every push and attaches them to a GitHub
Release on tags matching `v*`.

### Tests and dummy data

```bash
uv sync --group test
uv run playwright install chromium
uv run pytest
```

The Playwright tests in `tests/e2e` launch the real dashboard with a temporary
TOML configuration and synthetic CTAP output. They cover every page, including
loaded EEG plots, channel statistics, QC images, and logs. Run just the browser
checks with `uv run pytest tests/e2e` (add `--headed` to watch them).

To generate the same small dataset for manual use:

```bash
uv run python -m tests.dummy_data /tmp/ctap-dummy
uv run ctapdash --config /tmp/ctap-dummy/conf.toml
```

No reference participant files are required or copied. The generated `TAPPED`
directory mirrors the numbered processing steps, logs, and quality-control
layout of CTAP output.
