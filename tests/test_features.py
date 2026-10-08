"""Testnivå 1: kod. Enhetstester av feature-koden, utan modell och utan riktig data."""

import pandas as pd
import pytest

from churn.data import skapa_features


@pytest.fixture
def liten_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "manadskostnad": [299.0, 149.0],
            "data_gb_per_manad": [9.0, 0.0],
            "region": [" stockholm", "Väst"],
        }
    )


def test_kostnad_per_gb(liten_df):
    ut = skapa_features(liten_df)
    assert ut["kostnad_per_gb"].tolist() == pytest.approx([29.9, 149.0])


def test_region_normaliseras(liten_df):
    assert skapa_features(liten_df)["region"].tolist() == ["Stockholm", "Väst"]


@pytest.mark.parametrize(
    "indata, forvantat",
    [
        ("STOCKHOLM", "Stockholm"),
        ("stockholm", "Stockholm"),
        ("StOcKhOlM", "Stockholm"),
        ("VÄST", "Väst"),
        ("  VÄST  ", "Väst"),
    ],
)
def test_region_med_olika_skiftlage_normaliseras(indata, forvantat):
    df = pd.DataFrame(
        {
            "manadskostnad": [299.0],
            "data_gb_per_manad": [9.0],
            "region": [indata],
        }
    )
    assert skapa_features(df)["region"].iloc[0] == forvantat


def test_indata_andras_inte(liten_df):
    original = liten_df.copy()
    skapa_features(liten_df)
    pd.testing.assert_frame_equal(liten_df, original)