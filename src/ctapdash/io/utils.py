from contextlib import contextmanager
from pathlib import Path
import tempfile
from shutil import rmtree


@contextmanager
def atomic_write(destination: Path, dir=False, overwrite=False):
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    if not dir:
        staging = staging / destination.name
    try:
        yield staging
    except BaseException:
        rmtree(staging, ignore_errors=True)
        raise
    if destination.exists() or destination.is_symlink():
        if overwrite:
            rmtree(destination)
        else:
            raise FileExistsError(destination)
    staging.rename(destination)