"""pandas DataFrame input: string, category, boolean and datetime columns are encoded, one column per input column."""
import numpy as np
import pandas as pd
import pytest

from tabdpt import TabDPTClassifier, TabDPTRegressor
from tabdpt.utils import DataFrameEncoder

from device_utils import pick_device

DEVICE = pick_device()


def test_encoder_output():
    X = pd.DataFrame(
        {
            "num": pd.array([1, None, 3], dtype="Int64"),
            "color": ["red", None, "blue"],
            "size": pd.Categorical(["S", "M", "S"]),
            "flag": [True, False, True],
            # 01:00 in Rome in January is midnight UTC
            "when": pd.to_datetime(["2020-01-01 01:00", None, "2020-01-01 03:00"]).tz_localize("Europe/Rome"),
            "duration": pd.to_timedelta([1, 2, None], unit="h"),
        }
    )
    midnight_2020 = 1577836800.0
    nan = np.nan
    expected = np.array(
        [
            # categories are numbered in sorted order; missing values stay NaN
            [1.0, 1.0, 1.0, 1.0, midnight_2020, 3600.0],
            [nan, nan, 0.0, 0.0, nan, 7200.0],
            [3.0, 0.0, 1.0, 1.0, midnight_2020 + 7200, nan],
        ]
    )
    encoder = DataFrameEncoder().fit(X)
    np.testing.assert_array_equal(encoder.transform(X), expected)
    # columns are matched by name, unseen categories become missing
    shuffled = X[X.columns[::-1]].assign(color=["green", "red", "blue"])
    np.testing.assert_array_equal(encoder.transform(shuffled)[:, 1], [nan, 1.0, 0.0])
    with pytest.raises(ValueError, match="missing columns"):
        encoder.transform(X.drop(columns="flag"))


def test_encoder_rejects_unsupported_dtype():
    with pytest.raises(TypeError, match="'z'"):
        DataFrameEncoder().fit(pd.DataFrame({"a": [1.0, 2.0], "z": [1 + 2j, 3j]}))


@pytest.mark.parametrize("Estimator", [TabDPTClassifier, TabDPTRegressor])
def test_dataframe_matches_encoded_array(Estimator):
    """A DataFrame gives the same predictions as the numeric array it encodes to."""
    rng = np.random.default_rng(0)
    n = 300
    codes = rng.integers(0, 3, n).astype(float)
    codes[rng.random(n) < 0.1] = np.nan
    X_array = np.column_stack([rng.normal(size=n), codes, rng.integers(0, 2, n)])
    labels = np.array(["a", "b", "c"], dtype=object)  # sorted like the codes
    X_df = pd.DataFrame(
        {
            "x": X_array[:, 0],
            "letter": np.where(np.isnan(codes), None, labels[np.nan_to_num(codes).astype(int)]),
            "flag": X_array[:, 2].astype(bool),
        }
    )
    y = X_array[:, 0] + np.nan_to_num(codes)
    if Estimator is TabDPTClassifier:
        y = (y > np.median(y)).astype(int)

    def fit_predict(X, X_test):
        est = Estimator(device=DEVICE, use_flash=False, compile=False, missing_indicators=True, verbose=False)
        assert est.fit(X[:200], y[:200]) is est
        return est.predict(X_test, seed=0)

    expected = fit_predict(X_array, X_array[200:])
    np.testing.assert_allclose(fit_predict(X_df, X_df[200:]), expected)
    # test columns in another order are matched by name
    np.testing.assert_allclose(fit_predict(X_df, X_df[200:][["flag", "x", "letter"]]), expected)


def test_non_numeric_array_points_to_dataframe():
    X = np.array([["a", 1.0], ["b", 2.0]], dtype=object)
    with pytest.raises(ValueError, match="DataFrame"):
        TabDPTClassifier(device=DEVICE, use_flash=False, compile=False, verbose=False).fit(X, np.array([0, 1]))
