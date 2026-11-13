"""Unit tests for TabDPT KV Caching implementation.

Validates:
1. Exact mathematical equivalence between full-sequence forward pass and KV-cached inference for classification.
2. Exact mathematical equivalence for regression mode.
3. Input dimension flexibility (2D and 3D tensor shapes).
4. Multi-query batching with single context cache.
5. Disk serialization and deserialization integrity of (kv_cache, n_ctx, stats).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

tabdpt_src = Path(__file__).resolve().parent.parent / "src"
if str(tabdpt_src) not in sys.path:
    sys.path.insert(0, str(tabdpt_src))

from tabdpt import TabDPTClassifier, TabDPTRegressor
from tabdpt.model import TabDPTModel


@pytest.fixture
def tiny_tabdpt_model():
    torch.manual_seed(42)
    model = TabDPTModel(
        dropout=0.0,
        enc_cell_dim=-1,
        n_out=16,
        regression_bin_count=10,
        regression_bin_min=-10,
        regression_bin_max=10,
        nhead=4,
        nhid=64,
        ninp=64,
        nlayers=3,
        num_features=12,
        base_len=16,
        max_len=1024,
        y_encoder_dim=32,
        num_col_attn_layers=1,
        n_thinking_rows=8,
        use_flash=False,
        clip_sigma=8.0,
    ).eval()
    return model


class TestKVCacheEquivalence:
    def test_classification_equivalence(self, tiny_tabdpt_model):
        model = tiny_tabdpt_model
        torch.manual_seed(101)
        B, n_ctx, n_qry, F = 1, 40, 5, 12

        x_ctx = torch.randn(B, n_ctx, F)
        y_ctx = torch.randint(0, 16, (B, n_ctx))
        x_qry = torch.randn(B, n_qry, F)
        x_full = torch.cat([x_ctx, x_qry], dim=1)

        with torch.no_grad():
            full_out = model(x_full, y_ctx, is_cls=True)
            kv_cache, n_ctx_ret, stats = model.encode_context(x_ctx, y_ctx, is_cls=True)
            cache_out = model.predict_with_cache(x_qry, kv_cache, n_ctx_ret, stats)

        assert full_out.shape == cache_out.shape
        max_diff = torch.max(torch.abs(full_out - cache_out)).item()
        assert max_diff < 1e-5, f"Discrepancy detected: {max_diff}"

    def test_regression_equivalence(self, tiny_tabdpt_model):
        model = tiny_tabdpt_model
        torch.manual_seed(202)
        B, n_ctx, n_qry, F = 1, 30, 4, 12

        x_ctx = torch.randn(B, n_ctx, F)
        y_ctx = torch.randn(B, n_ctx)
        x_qry = torch.randn(B, n_qry, F)
        x_full = torch.cat([x_ctx, x_qry], dim=1)

        with torch.no_grad():
            full_out = model(x_full, y_ctx, is_cls=False)
            kv_cache, n_ctx_ret, stats = model.encode_context(
                x_ctx, y_ctx, is_cls=False
            )
            cache_out = model.predict_with_cache(x_qry, kv_cache, n_ctx_ret, stats)

        assert full_out.shape == cache_out.shape
        max_diff = torch.max(torch.abs(full_out - cache_out)).item()
        assert max_diff < 1e-5, f"Discrepancy detected: {max_diff}"

    def test_2d_and_3d_query_input(self, tiny_tabdpt_model):
        model = tiny_tabdpt_model
        torch.manual_seed(303)
        n_ctx, n_qry, F = 35, 3, 12

        x_ctx = torch.randn(n_ctx, F)
        y_ctx = torch.randint(0, 16, (n_ctx,))
        x_qry_2d = torch.randn(n_qry, F)
        x_qry_3d = x_qry_2d.unsqueeze(0)

        with torch.no_grad():
            kv_cache, n_ctx_ret, stats = model.encode_context(x_ctx, y_ctx, is_cls=True)
            out_2d = model.predict_with_cache(x_qry_2d, kv_cache, n_ctx_ret, stats)
            out_3d = model.predict_with_cache(x_qry_3d, kv_cache, n_ctx_ret, stats)

        assert torch.allclose(out_2d, out_3d, atol=1e-6)

    def test_packed_tuple_cache_argument(self, tiny_tabdpt_model):
        model = tiny_tabdpt_model
        torch.manual_seed(404)
        x_ctx = torch.randn(1, 25, 12)
        y_ctx = torch.randint(0, 16, (1, 25))
        x_qry = torch.randn(1, 2, 12)

        with torch.no_grad():
            cached_pack = model.encode_context(x_ctx, y_ctx, is_cls=True)
            out_unpacked = model.predict_with_cache(
                x_qry, cached_pack[0], cached_pack[1], cached_pack[2]
            )
            out_packed = model.predict_with_cache(x_qry, cached_pack)

        assert torch.allclose(out_unpacked, out_packed, atol=1e-6)

    def test_serialization_roundtrip(self, tmp_path, tiny_tabdpt_model):
        model = tiny_tabdpt_model
        torch.manual_seed(505)
        x_ctx = torch.randn(1, 30, 12)
        y_ctx = torch.randint(0, 16, (1, 30))
        x_qry = torch.randn(1, 1, 12)

        cache_file = tmp_path / "test_kv_cache.pt"
        with torch.no_grad():
            cached_pack = model.encode_context(x_ctx, y_ctx, is_cls=True)
            out_before = model.predict_with_cache(x_qry, cached_pack)

        torch.save(cached_pack, cache_file)
        loaded_pack = torch.load(cache_file)

        with torch.no_grad():
            out_after = model.predict_with_cache(x_qry, loaded_pack)

        assert torch.allclose(out_before, out_after, atol=1e-6)

    def test_stats_formats(self, tiny_tabdpt_model):
        model = tiny_tabdpt_model
        torch.manual_seed(606)
        x_ctx = torch.randn(1, 20, 12)
        y_ctx = torch.randint(0, 16, (1, 20))
        x_qry = torch.randn(1, 2, 12)

        with torch.no_grad():
            kv_cache, n_ctx, stats_dict = model.encode_context(
                x_ctx, y_ctx, is_cls=True
            )
            out_dict = model.predict_with_cache(x_qry, kv_cache, n_ctx, stats_dict)

            # Test 6-tuple stats
            stats_tuple_6 = (
                stats_dict["c1_min"],
                stats_dict["c1_max"],
                stats_dict["norm_mean"],
                stats_dict["norm_std"],
                stats_dict["c3_min"],
                stats_dict["c3_max"],
            )
            out_tuple_6 = model.predict_with_cache(
                x_qry, kv_cache, n_ctx, stats_tuple_6
            )
            assert torch.allclose(out_dict, out_tuple_6, atol=1e-6)

            # Test 2-tuple stats (mean, std)
            stats_tuple_2 = (stats_dict["norm_mean"], stats_dict["norm_std"])
            out_tuple_2 = model.predict_with_cache(
                x_qry, kv_cache, n_ctx, stats_tuple_2
            )
            assert out_tuple_2.shape == out_dict.shape

            # Test stats=None
            out_none = model.predict_with_cache(x_qry, kv_cache, n_ctx, None)
            assert out_none.shape == out_dict.shape


class TestEstimatorKVCache:
    def test_classifier_estimator_equivalence(self):
        np.random.seed(42)
        X_train = np.random.randn(40, 6)
        y_train = np.random.randint(0, 3, size=40)
        X_test = np.random.randn(8, 6)

        # 1. Fit estimator with KV caching enabled (default)
        clf_cached = TabDPTClassifier(device="cpu", use_kv_cache=True)
        clf_cached.fit(X_train, y_train)

        # Context KV cache must be precomputed and stored
        assert clf_cached._precomputed_kv is not None

        probs_cached = clf_cached.predict_proba(X_test)
        preds_cached = clf_cached.predict(X_test, n_ensembles=1)

        # 2. Fit estimator with KV caching disabled (baseline full-forward)
        clf_vanilla = TabDPTClassifier(device="cpu", use_kv_cache=False)
        clf_vanilla.fit(X_train, y_train)
        assert clf_vanilla._precomputed_kv is None

        probs_vanilla = clf_vanilla.predict_proba(X_test)
        preds_vanilla = clf_vanilla.predict(X_test, n_ensembles=1)

        # Mathematical equivalence check
        assert np.allclose(probs_cached, probs_vanilla, atol=1e-5)
        assert np.array_equal(preds_cached, preds_vanilla)

    def test_regressor_estimator_equivalence(self):
        np.random.seed(42)
        X_train = np.random.randn(40, 6)
        y_train = np.random.randn(40)
        X_test = np.random.randn(8, 6)

        reg_cached = TabDPTRegressor(device="cpu", use_kv_cache=True)
        reg_cached.fit(X_train, y_train)
        assert reg_cached._precomputed_kv is not None

        pred_cached = reg_cached.predict(X_test, n_ensembles=1)

        reg_vanilla = TabDPTRegressor(device="cpu", use_kv_cache=False)
        reg_vanilla.fit(X_train, y_train)
        assert reg_vanilla._precomputed_kv is None

        pred_vanilla = reg_vanilla.predict(X_test, n_ensembles=1)

        assert np.allclose(pred_cached, pred_vanilla, atol=1e-5)

    def test_classifier_batch_invariance(self):
        np.random.seed(123)
        X_train = np.random.randn(30, 5)
        y_train = np.random.randint(0, 2, size=30)
        X_test = np.random.randn(10, 5)

        clf = TabDPTClassifier(device="cpu", use_kv_cache=True)
        clf.fit(X_train, y_train)

        probs_single_batch = clf.predict_proba(X_test, batch_size=10)
        probs_multi_batch = clf.predict_proba(X_test, batch_size=2)

        assert np.allclose(probs_single_batch, probs_multi_batch, atol=1e-5)
