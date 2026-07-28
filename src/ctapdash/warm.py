import argparse
import sys
from pathlib import Path

from ctapdash import config
from ctapdash.config import SETTINGS
from ctapdash.io import read_eeglab, cached_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ctapdash-warm",
        description="Pre-convert EEGLAB .set files to the faster .fif cache format.",
    )
    parser.add_argument("--config", type=Path, metavar="PATH", help="TOML configuration file")
    parser.add_argument("--clean", action="store_true", help="Discard existing caches first")
    args = parser.parse_args(argv)

    if args.config:
        config.load_from_file(args.config)
    elif not config.load_from_env():
        parser.error(f"no configuration: pass --config or set ${config.ENV_VAR}")

    for source in SETTINGS.sources.values():
        source_path = Path(source)
        for subdir in source_path.iterdir():
            if not subdir.name[0].isnumeric():
                continue
            for path in subdir.iterdir():
                if path.suffix != ".set":
                    continue
                if args.clean:
                    cached_path(path, raw=True).unlink(missing_ok=True)
                    cached_path(path, raw=False).unlink(missing_ok=True)
                if read_eeglab(path, warm=True):
                    print("Caching:", path)
                else:
                    print("Already cached:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
