"""Testnivå 2: data. Ser datan ut som vi tror?"""

from churn.data import FEATURES, MAL, las_data, skapa_features


def test_forvantade_kolumner_finns():
    df = skapa_features(las_data())
    assert set(FEATURES + [MAL]) <= set(df.columns)


def test_malvariabeln_ar_binar():
    assert set(las_data()[MAL].unique()) <= {0, 1}


def test_kund_id_ar_unikt():
    df = las_data()
    dubbletter = df[df["kund_id"].duplicated(keep=False)]
    assert dubbletter.empty, f"Dubblerade kund_id:\n{dubbletter}"


def test_alder_i_rimligt_intervall():
    df = las_data()
    utanfor = df[~df["alder"].between(18, 100)]
    assert utanfor.empty, f"Ålder utanför 18–100:\n{utanfor[['kund_id', 'alder']]}"