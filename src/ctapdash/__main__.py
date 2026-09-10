import sys

from ctapdash.cli import main

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    sys.exit(main())
