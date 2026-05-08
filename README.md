# CTAP Dashboard

A dashboard for viewing the outputs of [CTAP](https://version.helsinki.fi/hipercog/Methods/ctap) pipelines.

*Step 1.* Write your configuration file to e.g. `conf.toml`:

```toml
[sources]
source_name = "/path/to/TAPPED"
```

*Step 2.* Then run the dashboard with the following command:

```bash
CTAPDASH_SETTINGS="/path/to/conf.toml" uv run uvicorn main:app
```
