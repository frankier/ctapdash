"""Thin application adapter over the shared desktop runner."""

from pathlib import Path

from workingtitle.desktop import run_cli

from ctapdash import config
from ctapdash.build import ASSETS
from ctapdash.desktop import APP, SESSION
from ctapdash.smoke import smoke_test


def add_arguments(parser):
    parser.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help=f"TOML configuration file. Overrides ${config.ENV_VAR}. "
        "Without either, the dashboard uses its managed configuration.",
    )


def prepare(args):
    """Load the requested configuration before assets or the app are built.

    The returned override is inherited by the reload subprocess, so its factory
    finds the same file. ValueError becomes a normal argparse error.
    """
    if args.config:
        if not args.config.exists():
            raise ValueError(f"no such configuration file: {args.config}")
        config.load_from_file(args.config)
        return {config.ENV_VAR: str(args.config)}
    config.load_startup()
    return {}


def main(argv=None):
    return run_cli(
        APP,
        argv,
        add_arguments=add_arguments,
        prepare=prepare,
        assets=ASSETS,
        smoke_test=smoke_test,
        session=SESSION,
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
