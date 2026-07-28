import json
import tomllib
from dataclasses import dataclass, field
from os import environ
from pathlib import Path


ENV_VAR = "CTAPDASH_SETTINGS"


@dataclass
class Settings:
    """Application configuration, mutable at runtime by the setup UI.

    Deliberately has no default location: sources come from --config, from
    CTAPDASH_SETTINGS, or from the in-app picker. Picker-selected sources live
    only in this object and are lost on exit unless explicitly saved.
    """

    sources: dict[str, str] = field(default_factory=dict)
    loaded_from: Path | None = None

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


def load_from_env():
    """Load from CTAPDASH_SETTINGS if it is set. Returns whether it was."""
    value = environ.get(ENV_VAR)
    if not value:
        return False
    load_from_file(value)
    return True


def dumps_toml():
    # TOML basic strings and JSON strings agree on escaping for everything we
    # emit here (source names and filesystem paths), so json.dumps is a correct
    # and dependency-free writer.
    lines = ["[sources]"]
    for name, directory in SETTINGS.sources.items():
        lines.append(f"{json.dumps(name)} = {json.dumps(str(directory))}")
    return "\n".join(lines) + "\n"


def save_to_file(path):
    path = Path(path)
    path.write_text(dumps_toml())
    SETTINGS.loaded_from = path
    return path
