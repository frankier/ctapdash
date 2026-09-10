import mne
import numpy as np
import pytest
from scipy import stats

from ctapdash.stats import describe_mne as describe


@pytest.fixture
def info():
    return mne.create_info(["Fz", "Cz"], sfreq=100, ch_types="eeg")


def test_raw_returns_scipy_statistics_by_channel(info):
    data = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 8.0, 10.0]])
    raw = mne.io.RawArray(data, info, verbose="error")

    result = describe(raw)
    expected = stats.describe(data, axis=-1, ddof=0)

    assert dict(result.sizes) == {"channel": 2}
    assert list(result.data_vars) == [
        "nobs",
        "min",
        "max",
        "mean",
        "variance",
        "skewness",
        "kurtosis",
    ]
    assert result.channel.values.tolist() == ["Fz", "Cz"]
    np.testing.assert_array_equal(result["nobs"].values, [expected.nobs] * 2)
    np.testing.assert_allclose(result["min"].values, expected.minmax[0])
    np.testing.assert_allclose(result["max"].values, expected.minmax[1])
    np.testing.assert_allclose(result["mean"].values, expected.mean)
    np.testing.assert_allclose(result["variance"].values, expected.variance)
    np.testing.assert_allclose(result["skewness"].values, expected.skewness)
    np.testing.assert_allclose(result["kurtosis"].values, expected.kurtosis)


def test_epochs_flattens_epochs_and_times_per_channel(info):
    data = np.array(
        [
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
            [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
        ]
    )
    epochs = mne.EpochsArray(data, info, verbose="error")

    result = describe(epochs)

    expected = data.swapaxes(0, 1).reshape(2, -1)
    np.testing.assert_allclose(result["mean"].values, expected.mean(axis=-1))
    np.testing.assert_array_equal(result["nobs"].values, [6, 6])


