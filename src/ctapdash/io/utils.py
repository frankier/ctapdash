from contextlib import contextmanager
from pathlib import Path
import tempfile
from shutil import rmtree


@contextmanager
def atomic_write(destination: Path, dir=False, overwrite=False):
    destination = Path(destination)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    staging = temporary if dir else temporary / destination.name
    try:
        yield staging
        if destination.exists() or destination.is_symlink():
            if not overwrite:
                raise FileExistsError(destination)
            if destination.is_dir():
                rmtree(destination)
        staging.replace(destination)
    finally:
        rmtree(temporary, ignore_errors=True)
