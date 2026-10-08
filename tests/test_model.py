"""Testnivå 3: modell. Kontrakt, reproducerbarhet och prestanda."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from churn.data import FEATURES, dela_upp, las_data, skapa_features
from churn.model import trana_och_utvardera

ROC_AUC_TROSKEL = 0.70


@pytest.fixture(scope="module")
def resultat():
    return trana_och_utvardera()


def test_modellkontrakt(resultat):
    modell, _ = resultat
    X, _ = dela_upp(skapa_features(las_data().head(5)))
    assert list(X.columns) == FEATURES
    sannolikhet = modell.predict_proba(X)
    assert sannolikhet.shape == (5, 2)
    assert np.all((sannolikhet >= 0) & (sannolikhet <= 1))


def test_reproducerbar(resultat):
    _, matvarden = resultat
    _, igen = trana_och_utvardera()
    assert igen == matvarden


def test_roc_auc_over_troskel(resultat):
    _, matvarden = resultat
    assert matvarden["roc_auc"] >= ROC_AUC_TROSKEL


def test_modellen_slar_dummy(resultat):
    _, matvarden = resultat
    X, y = dela_upp(skapa_features(las_data()))
    dummy = DummyClassifier(strategy="prior").fit(X, y)
    dummy_auc = roc_auc_score(y, dummy.predict_proba(X)[:, 1])
    assert matvarden["roc_auc"] > dummy_auc + 0.05