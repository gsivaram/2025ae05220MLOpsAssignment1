import numpy as np
import pandas as pd
import pytest

from heartrisk import config, data


def test_raw_file_has_expected_shape_and_missing_markers(raw_df):
    assert raw_df.shape == (303, 14)
    assert list(raw_df.columns) == config.RAW_COLUMNS
    # The '?' placeholders must be parsed as NaN, not kept as strings.
    assert raw_df["ca"].isna().sum() == 4
    assert raw_df["thal"].isna().sum() == 2
    assert pd.api.types.is_numeric_dtype(raw_df["ca"])


def test_clean_output_schema(clean_df):
    assert list(clean_df.columns) == config.FEATURES + [config.TARGET]
    assert "num" not in clean_df
    assert set(clean_df[config.TARGET].unique()) == {0, 1}
    assert clean_df[config.TARGET].value_counts().to_dict() == {0: 164, 1: 139}


@pytest.mark.parametrize("num, expected", [(0, 0), (1, 1), (2, 1), (4, 1)])
def test_target_is_binarised(raw_row, num, expected):
    out = data.clean(pd.DataFrame([{**raw_row, "num": num}]))
    assert out[config.TARGET].iloc[0] == expected


@pytest.mark.parametrize("col, bad_value", [("cp", 9), ("thal", 5), ("sex", 2), ("slope", 0)])
def test_out_of_domain_codes_become_missing(raw_row, col, bad_value):
    out = data.clean(pd.DataFrame([{**raw_row, col: bad_value}]))
    assert np.isnan(out[col].iloc[0])


@pytest.mark.parametrize("col", ["chol", "trestbps"])
def test_impossible_zero_readings_become_missing(raw_row, col):
    out = data.clean(pd.DataFrame([{**raw_row, col: 0}]))
    assert np.isnan(out[col].iloc[0])


def test_missing_values_are_kept_for_pipeline_imputation(raw_row):
    out = data.clean(pd.DataFrame([{**raw_row, "ca": np.nan}]))
    assert len(out) == 1 and np.isnan(out["ca"].iloc[0])


def test_rows_without_target_are_dropped(raw_row):
    out = data.clean(pd.DataFrame([raw_row, {**raw_row, "num": np.nan, "age": 40}]))
    assert len(out) == 1


def test_duplicates_are_dropped(raw_row):
    out = data.clean(pd.DataFrame([raw_row, raw_row]))
    assert len(out) == 1


def test_non_numeric_garbage_is_coerced(raw_row):
    out = data.clean(pd.DataFrame([{**raw_row, "chol": "abc"}]))
    assert np.isnan(out["chol"].iloc[0])


def test_missing_column_raises(raw_row):
    bad = pd.DataFrame([raw_row]).drop(columns="thal")
    with pytest.raises(ValueError, match="thal"):
        data.clean(bad)


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    data.clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_missing_report_lists_only_columns_with_gaps(clean_df):
    assert data.missing_report(clean_df).to_dict() == {"ca": 4, "thal": 2}


def test_download_skips_existing_file(tmp_path, monkeypatch):
    dest = tmp_path / "raw.data"
    dest.write_text("cached")

    def fail(*_a, **_k):
        raise AssertionError("network must not be used when the file exists")

    monkeypatch.setattr(data.urllib.request, "urlopen", fail)
    assert data.download_raw(dest=dest) == dest
    assert dest.read_text() == "cached"


def test_download_force_fetches_and_writes(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    dest = tmp_path / "sub" / "raw.data"
    data.download_raw(dest=dest, force=True)
    assert data.load_raw(dest).shape == (1, 14)


def test_forced_download_falls_back_to_cached_file_when_offline(tmp_path, monkeypatch):
    dest = tmp_path / "raw.data"
    dest.write_text("cached")

    def offline(*_a, **_k):
        raise OSError("network unreachable")

    monkeypatch.setattr(data.urllib.request, "urlopen", offline)
    assert data.download_raw(dest=dest, force=True) == dest
    assert dest.read_text() == "cached"


def test_download_without_cache_propagates_network_error(tmp_path, monkeypatch):
    def offline(*_a, **_k):
        raise OSError("network unreachable")

    monkeypatch.setattr(data.urllib.request, "urlopen", offline)
    with pytest.raises(OSError):
        data.download_raw(dest=tmp_path / "missing.data", force=True)
