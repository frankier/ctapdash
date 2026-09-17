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
use _Save configuration_ to write them to a file you can pass with `--config`
next time. There is deliberately no automatic config location.

### Other options

| Option         | Effect                                             |
| -------------- | -------------------------------------------------- |
| `--port N`     | Serve on a fixed port instead of a free one        |
| `--no-window`  | Serve only; don't open a window or a browser       |
| `--no-browser` | Don't open a browser                               |
| `--reload`     | Restart on source changes (browser only)           |
| `--debug`      | Show tracebacks in the browser, implies `--reload` |

The dashboard warms metadata, transposed samples, pyramids, and statistics when
sources are loaded or added in Setup. A bottom-right indicator shows the active
operation and completed/total build jobs, pushed over an HTMX WebSocket. It is
hidden while scanning, idle, failed, or disconnected; only actual build progress
is displayed.

Freshness is checked at registration using `.set` mtimes. External `.fdt` changes
and removed recordings are not detected. Re-add an existing source in Setup to
check it again after editing `.set` files; there is no filesystem watcher.

`ctapdash-warm --config conf.toml` warms only the metadata pickle caches.

## VennDiff domains and epochs

VennDiff compares continuous and epoched EEGLab recordings on their original
recording timeline. **Domain → Union** (the default) shows every time present in
either file, including one-sided data. **Intersection** concatenates the shared
intervals. Intervals absent from the selected domain take no horizontal space;
ticks show original seconds, with `//` for forward jumps and `↶` for repeated time.

In union, overlapping epochs have independent A/B `‹ 1/2 ›` controls above their
intervals. In intersection, each overlapping epoch pair appears sequentially,
sorted by original start time. **Show epoch starts** defaults off in union and
on in intersection; each mode remembers its checkbox setting for the session.
Red/blue dashed lines mark A/B starts, and purple marks coincident starts.

Epoch starts come directly from the sample offsets in `recording.events[:, 0]`,
converted to seconds using the recording's sampling rate. Continuous recordings
use their sample timeline as one interval. Alignment requires compatible sampling
grids; other event metadata and boundary deletions are not considered.
Recordings use bounded reads from external `.fdt` files and pyramids.

Pyramids use a shared sample-zero grid: factor 64 summarizes samples 0–63,
64–127, and so on. Epoch offsets come from `recording.events[:, 0]`.
Each epoch retains its own partial first/last buckets; overlapping epochs remain
separate. At cached zoom levels, VennDiff slices these summaries directly and
clips drawing to the selected intervals, without raw reads or boundary
reaggregation. A clipped continuous bucket still summarizes its full source
bucket. Line pyramids retain LTTB reduction with aligned bucket slots.
Older pyramid caches rebuild automatically when the dataset is registered;
raw data remains available while caches are being built.

## Developing

```bash
uv sync --group build
npm ci
uv run python -m ctapdash.build
uv run ctapdash --config conf.toml
```

`--debug` implies `--reload`, so `uv run ctapdash --config conf.toml --debug`
restarts the server whenever you edit a source file. Both options serve in the
browser: uvicorn's reloader runs the application in a subprocess, which cannot
own the main thread as the native window needs. `--smoke-test` turns reload off.

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

### Channel selector

The statistics heatmap and Venndiff share `<channel-selector>`, with searchable
type/region groups and a clickable, lasso-selectable head diagram. Selection
is remembered per dataset and participant in the current browser tab. Unavailable
channels retain their selection for later steps. Bad-channel markers describe the
steps reporting them and never exclude channels automatically. Heatmap colors are
normalized over the selected channels in the displayed steps.

`ctapdash.channels.participant_channel_metadata(dataset, bads_by_step=...)`
accepts optional per-step bad-channel overrides. Omitted
steps use recording `info["bads"]`; an explicit empty list clears that step's markers.
The lower-level `channel_metadata` accepts `(step, Ctap*EEGLAB)` pairs. Both read
metadata only, preserve EEGLAB type labels, and isolate private MNE projection and
region helpers in `ctapdash.channels`. Invalid positions fall back to the checklist.

Import `static/generated/channel-selector.js` as an ES module to reuse the widget.
Call `selector.configure(metadata, availableNames, badsByStep)` with the adapter's
metadata (including a dataset/participant `stateKey`). Individual properties are
`channels`, `groups`, `head`, `selectedNames`, `availableNames`, and `badsByStep`.
Property updates do not emit selection events. User gestures emit a bubbling,
composed `channel-selection-change` event with `detail.selectedNames` in metadata
order, including selection intent for unavailable channels. Consumers should
intersect with `availableNames` when choosing which channels to display.

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
