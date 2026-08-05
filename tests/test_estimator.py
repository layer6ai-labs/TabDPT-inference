"""TabDPTEstimator init and weight loading, exercised against the real in-repo v1.2 checkpoint."""
import pytest
import torch
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from tabdpt import TabDPTClassifier
from tabdpt import estimator
from tabdpt.estimator import TabDPTEstimator
from tabdpt.utils import Log1pScaler

from device_utils import pick_device

DEVICE = pick_device()

V12_MAX_FEATURES = 128
V12_MAX_NUM_CLASSES = 16


def build(**kwargs):
    """Construct a base estimator using the default HF-downloaded weights."""
    return TabDPTEstimator(mode="cls", device=DEVICE, **kwargs)


# --- Real weight loading + init ---

def test_real_v12_load_and_init():
    """The v1.2 checkpoint loads, exposes the right dims, and populates real tensors."""
    model = TabDPTClassifier(device=DEVICE)
    assert model.max_features == V12_MAX_FEATURES
    assert model.max_num_classes == V12_MAX_NUM_CLASSES
    assert isinstance(model.scaler, StandardScaler)  # default normalizer

    thinking = model.model.thinking_embed
    assert thinking.shape[0] == model.model.n_thinking_rows > 0
    assert torch.isfinite(thinking).all()
    assert thinking.abs().sum().item() > 0


def test_default_downloads_from_hf():
    est = build()
    assert est.path.endswith(estimator._MODEL_NAME)


# --- Normalizer selection ---

@pytest.mark.parametrize(
    "normalizer,scaler_cls",
    [
        ("standard", StandardScaler),
        ("minmax", MinMaxScaler),
        ("robust", RobustScaler),
        ("power", PowerTransformer),
        ("quantile-uniform", QuantileTransformer),
        ("quantile-normal", QuantileTransformer),
        ("log1p", Log1pScaler),
        (None, type(None)),
    ],
)
def test_normalizer_selects_scaler(normalizer, scaler_cls):
    est = build(normalizer=normalizer)
    assert isinstance(est.scaler, scaler_cls)


# --- Invalid args ---

def test_invalid_normalizer_raises():
    with pytest.raises(ValueError):
        build(normalizer="not-a-scaler")


def test_invalid_feature_reduction_raises():
    with pytest.raises(AssertionError):
        build(feature_reduction="bad")


def test_invalid_faiss_metric_raises():
    with pytest.raises(AssertionError):
        build(faiss_metric="bad")
