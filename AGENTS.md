# AGENTS.md

## Project

Dashboard for viewing the outputs of CTAP pipelines, available as a local web app or desktop build.

## Layout and tooling

- `src/ctapdash/`: Python app, Jinja templates, static assets, CTAP data loading, and plotting.
- `src/css/` and `src/js/`: frontend source; `src/venn_ts/`: Bokeh extension with Python and TypeScript components.
- `tests/`: Python tests and Playwright browser tests; `tests/dummy_data.py` generates sample CTAP output.
- `uv` manages Python dependencies. `npm` builds Tailwind CSS and bundles TypeScript with esbuild; `python -m ctapdash.build` also builds the Bokeh extension. The app uses Starlette, HTMX, Alpine.js, and Bokeh.

## Style — DRY and reuse first

Follow the patterns already in the codebase rather than inventing new ones.
Be succinct!
Prefer Tailwind utilities over custom CSS. Prefer CSS or Alpine.js over custom JavaScript; choose between CSS and Alpine.js based on which is easier for the task. Keep layouts responsive.

## Formatting

`prek run --all-files` runs checks configured in `prek.toml`.
