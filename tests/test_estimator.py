"""TabDPTEstimator init and weight loading, exercised against the real in-repo checkpoint."""
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

MAX_FEATURES = 128
MAX_NUM_CLASSES = 16


def build(**kwargs):
    """Construct a base estimator using the default HF-downloaded weights."""
    return TabDPTEstimator(mode="cls", device=DEVICE, **kwargs)


# --- Real weight loading + init ---

def test_real_load_and_init():
    """The checkpoint loads, exposes the right dims, and populates real tensors."""
    model = TabDPTClassifier(device=DEVICE)
    assert model.max_features == MAX_FEATURES
    assert model.max_num_classes == MAX_NUM_CLASSES
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


# --- Separable model loading ---

def test_load_model_is_separately_callable():
    """``_load_model`` can be re-run (or overridden) after construction to swap the network."""
    est = build()
    first_model = est.model
    est._load_model()
    assert est.model is not first_model  # a fresh module, same checkpoint
    assert est.max_features == MAX_FEATURES  # unaffected: derived once, from the first load


def test_load_model_can_be_overridden_to_share_a_network():
    """A caller can skip the per-instance download/build entirely by overriding ``_load_model``."""
    donor = build()

    class _SharedWeightEstimator(TabDPTEstimator):
        def _load_model(self) -> None:
            self.model = donor.model  # reuse the donor's network instead of loading one

    est = _SharedWeightEstimator(mode="cls", device=DEVICE)
    assert est.model is donor.model


def test_get_params_reads_every_constructor_argument():
    """``clip_sigma`` and ``model_weight_path`` are stored on ``self``, as sklearn's contract requires."""
    est = build(clip_sigma=4.0)
    params = est.get_params()
    assert params["clip_sigma"] == 4.0
    assert params["model_weight_path"] is None
