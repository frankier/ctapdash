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

`ctapdash-warm --config conf.toml` pre-converts every `.set` file to the
faster `.fif` cache format, so the first view of each participant is quick.

## Developing

```bash
uv sync --group build
uv run ctapdash --config conf.toml
```

`uv run uvicorn ctapdash.webapp:create_app --factory` also works if you want a
plain ASGI server.

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
