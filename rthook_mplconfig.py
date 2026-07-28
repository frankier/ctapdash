"""Keep matplotlib's font cache out of the application directory.

Without this, a frozen app run from a read-only or shared location either
fails to cache fonts or rebuilds the cache on every launch.
"""

import os
import sys
import tempfile
from pathlib import Path

if getattr(sys, "frozen", False) and not os.environ.get("MPLCONFIGDIR"):
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    config_dir = base / "ctapdash" / "matplotlib"
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    else:
        os.environ["MPLCONFIGDIR"] = str(config_dir)
