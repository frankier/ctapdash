"""A recording and its derived artifacts, independent of MNE's object model."""
from dataclasses import dataclass

from ctapdash.io.paths import RecordingPaths


@dataclass
class RecordingData:
    paths: RecordingPaths
    metadata_validated: bool = False

    def read_metadata(self):
        from ctapdash.io.eeglab import read_eeglab
        validated = self.metadata_validated or self._registered_ready("metadata")
        return read_eeglab(self.paths.set, validated=validated)

    def open_transpose(self, return_xarray=False):
        self._registered_ready("transpose")
        self._require(self.paths.transpose)
        return self.read_metadata().mmap(
            return_xarray=return_xarray,
            data_fname=self.paths.transpose,
            ctapdash_order=True,
        )

    def open_pyramid(self, range=False):
        from ctapdash.io.pyramid import load_pyramid
        self._registered_ready("rangepyramid" if range else "pyramid")
        path = self.paths.rangepyramid if range else self.paths.pyramid
        self._require(path)
        return load_pyramid(path, range=range)

    def _registered_ready(self, kind):
        if self.metadata_validated:
            return True
        from ctapdash.io.cache import REGISTERED
        warmer = REGISTERED.get(str(self.paths.dataset.root))
        if warmer is None:
            return False
        if not warmer.ready(kind, (self.paths.dataset.root, self.paths.relative)):
            raise FileNotFoundError(f"Cache {kind} is not ready for {self.paths.set}")
        return True

    @staticmethod
    def _require(path):
        if not path.exists():
            raise FileNotFoundError(f"Cache artifact is not ready: {path}")
