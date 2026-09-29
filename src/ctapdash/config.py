import json
import os
import tempfile
import tomllib
from dataclasses import dataclass, field
from os import environ
from pathlib import Path

from platformdirs import user_config_path


ENV_VAR = "CTAPDASH_SETTINGS"
DEBUG_ENV_VAR = "CTAPDASH_DEBUG"


def debug_from_env():
    """Read the debug flag from the environment.

    uvicorn's reloader runs the application in a spawned subprocess, so the CLI
    passes --debug on to it this way rather than as an argument.
    """
    return environ.get(DEBUG_ENV_VAR, "").lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Sources and the mode chosen at startup."""

    sources: dict[str, str] = field(default_factory=dict)
    loaded_from: Path | None = None
    managed: bool = True

    @property
    def configured(self):
        return bool(self.sources)


SETTINGS = Settings()


def load_from_file(path):
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    SETTINGS.sources = {k: str(v) for k, v in data.get("sources", {}).items()}
    SETTINGS.loaded_from = path
    SETTINGS.managed = False


def load_from_env():
    """Load from CTAPDASH_SETTINGS if it is set. Returns whether it was."""
    value = environ.get(ENV_VAR)
    if not value:
        return False
    load_from_file(value)
    return True


def managed_config_path():
    return user_config_path("ctapdash", appauthor=False) / "config.toml"


def load_managed():
    path = managed_config_path()
    if path.exists():
        load_from_file(path)
    else:
        SETTINGS.sources = {}
        SETTINGS.loaded_from = None
    SETTINGS.managed = True


def load_startup():
    """Use an explicit environment path, otherwise the managed config."""
    if not load_from_env():
        load_managed()


def dumps_toml(sources=None):
    # TOML basic strings and JSON strings agree on escaping for everything we
    # emit here (source names and filesystem paths), so json.dumps is a correct
    # and dependency-free writer.
    lines = ["[sources]"]
    for name, directory in (SETTINGS.sources if sources is None else sources).items():
        lines.append(f"{json.dumps(name)} = {json.dumps(str(directory))}")
    return "\n".join(lines) + "\n"


def save_managed_sources(sources):
    """Commit a complete source map, leaving memory unchanged on write failure."""
    if not SETTINGS.managed:
        raise ValueError("Configuration is read only")
    path = managed_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".config-",
            suffix=".toml",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(dumps_toml(sources))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    SETTINGS.sources = sources
    SETTINGS.loaded_from = path
