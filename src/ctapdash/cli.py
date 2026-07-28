import argparse
import sys
from pathlib import Path

from ctapdash import config, desktop


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ctapdash",
        description="A dashboard for viewing the outputs of CTAP pipelines.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help=f"TOML configuration file. Overrides ${config.ENV_VAR}. "
        "Without either, the dashboard starts on its setup page.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=0, help="0 (the default) picks a free port"
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Serve only; do not open a native window",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open the system browser"
    )
    parser.add_argument("--debug", action="store_true", help="Show tracebacks in the browser")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Start up, self-check, and exit. Used to validate frozen builds.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config:
        if not args.config.exists():
            parser.error(f"no such configuration file: {args.config}")
        config.load_from_file(args.config)
    else:
        config.load_from_env()

    from ctapdash.webapp import create_app

    app = create_app(debug=args.debug)
    sock = desktop.bind_socket(args.host, args.port)

    if args.smoke_test:
        from ctapdash.smoke import run_smoke_test

        return run_smoke_test(app, sock)

    if args.no_window or not desktop.native_window_supported():
        desktop.run_browser(app, sock, open_browser=not args.no_browser)
        return 0

    try:
        desktop.run_window(app, sock)
    except Exception as err:
        # A missing or broken webview backend should degrade to the browser,
        # not to nothing at all.
        print(f"Could not open a native window ({err}); falling back to the browser.",
              file=sys.stderr)
        desktop.run_browser(app, sock, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
