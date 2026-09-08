"""Skip the on-startup extension build in frozen apps.

Frozen builds ship a venn_ts bundle produced at packaging time (see
ctapdash.spec) and contain no TypeScript sources or node runtime, so the
out-of-date check in ctapdash.cli would either rebuild nothing or fail. The
bundling process sets this environment variable for the frozen process to
make the check a no-op.
"""

import os

os.environ.setdefault("CTAPDASH_SKIP_EXTENSION_BUILD", "1")
