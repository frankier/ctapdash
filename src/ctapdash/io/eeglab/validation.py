"""Cheap container checks; the EEGLAB reader validates the actual contents."""

from pathlib import Path
import struct

from scipy.io.matlab import MatReadError, matfile_version


def set_appears_valid(path) -> bool:
    path = Path(path)
    try:
        with path.open("rb") as stream:
            header = stream.read(128)
            if len(header) < 20 or (0 not in header[:4] and len(header) < 128):
                # Truncated MATLAB header.
                return False
            stream.seek(0)
            major, minor = matfile_version(stream)
            if minor != 0:
                # Invalid MATLAB version.
                return False
            if major == 0:
                # MAT v4 has no magic string. Validate its five header integers
                # rather than accepting SciPy's zero-byte detection alone.
                size = path.stat().st_size
                for order, machine in (("<", 0), (">", 1)):
                    mopt, rows, cols, imag, name_len = struct.unpack(
                        order + "5i", header[:20]
                    )
                    if (
                        mopt // 1000 == machine
                        and mopt // 100 % 10 == 0
                        and mopt // 10 % 10 <= 5
                        and mopt % 10 <= 2
                        and rows >= 0
                        and cols >= 0
                        and imag in (0, 1)
                        and 1 <= name_len <= size - 20
                    ):
                        stream.seek(20 + name_len - 1)
                        if stream.read(1) == b"\0":
                            return True
                # Invalid MATLAB v4 header.
                return False
            if header[126:128] not in (b"IM", b"MI"):
                # Invalid MATLAB endian marker.
                return False
            if major == 2:
                from h5py import is_hdf5

                if not is_hdf5(path):
                    # MATLAB v7.3 file has no HDF5 signature.
                    return False
        return True
    except OSError, ValueError, MatReadError, struct.error:
        # Unreadable file or invalid MATLAB container.
        return False
