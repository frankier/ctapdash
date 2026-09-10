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
  *More info* → *Run anyway*.

On Windows and macOS the dashboard opens in its own window. On Linux it starts
a local server and opens your usual browser, because the native window there
would need a system WebKitGTK installation.

## Configuring

The dashboard needs to know where your CTAP output directories are. Write a
configuration file, e.g. `conf.toml`:

```toml
[sources]
source_name = "/path/to/TAPPED"
```

Then either pass it on the command line:

```bash
ctapdash --config /path/to/conf.toml
```

or set `CTAPDASH_SETTINGS=/path/to/conf.toml` in the environment (`--config`
wins if both are given).

If you give neither, the dashboard opens on a setup page where you can add
directories interactively. Sources added that way last only for that run —
use *Save configuration* to write them to a file you can pass with `--config`
next time. There is deliberately no automatic config location.

### Other options

| Option | Effect |
|---|---|
| `--port N` | Serve on a fixed port instead of a free one |
| `--no-window` | Serve only; don't open a window or a browser |
| `--no-browser` | Don't open a browser |
| `--debug` | Show tracebacks in the browser |

The dashboard warms metadata, transposed samples, pyramids, and statistics when
sources are loaded or added in Setup. A bottom-right indicator shows the active
operation and completed/total build jobs, pushed over an HTMX WebSocket. It is
hidden while scanning, idle, failed, or disconnected; only actual build progress
is displayed.

Freshness is checked at registration using `.set` mtimes. External `.fdt` changes
and removed recordings are not detected. Re-add an existing source in Setup to
check it again after editing `.set` files; there is no filesystem watcher.

`ctapdash-warm --config conf.toml` warms only the metadata pickle caches.

## Developing

```bash
uv sync --group build
uv run ctapdash --config conf.toml
```

`uv run uvicorn ctapdash.webapp:create_app --factory` also works if you want a
plain ASGI server.

### Accessing recording caches

All cache locations are defined in `ctapdash.io.paths`. Existing cache filenames
are preserved, including metadata pickles next to their `.set` files.

```python
from ctapdash.io.paths import ObservationData

observation = ObservationData.from_source("my-source", participant="participant-id")
recording = observation.get_recording(1)  # processing step number
set_path = recording.paths.set
transpose_path = recording.paths.transpose
metadata = recording.read_metadata()
samples = recording.open_transpose(return_xarray=True)
tree, groups = recording.open_pyramid(range=True)
```

For paths outside request context, use
`RecordingData(DatasetPaths(dataset_root).recording(set_path))` from
`ctapdash.io.recording` and `ctapdash.io.paths`. The dataset root is explicit,
so nested recordings resolve correctly. Opening an unavailable artifact raises
`FileNotFoundError`; server callers first await the corresponding
`app.state.cache_hurrier.hurry(kind, (dataset_root, set_path))` job. Dataset stats
are shared through `await DATASET_STATS.get(dataset_root)` from
`ctapdash.io.stats_cache`. This waits for the registered stats job on a miss
and retains a read-only mmap-backed xarray per dataset and server process.
`get_cached(root)` returns an existing entry or `None`. Consumers must not
mutate or close shared datasets; re-registration and shutdown invalidate them.
The comparison viewer uses whole-recording min/max stats for channel geometry,
including when recordings have different lengths. Setup does not scan recording
or pyramid values, although missing stats still require worker sample reads
and rendering visible tiles reads their data.
Standalone metadata reads retain their own freshness checks.

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
