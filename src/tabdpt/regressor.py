import math
from typing import Literal

import numpy as np
import torch
from sklearn.base import RegressorMixin
from tqdm import tqdm

from .estimator import TabDPTEstimator
from .utils import generate_random_permutation, pad_x, predict_regression_value


class TabDPTRegressor(TabDPTEstimator, RegressorMixin):
    def __init__(
        self,
        inf_batch_size: int = 512,
        normalizer: Literal["standard", "minmax", "robust", "power", "quantile-uniform", "quantile-normal", "log1p"] | None
            = "standard",
        missing_indicators: bool = False,
        clip_sigma: float = 4.,
        feature_reduction: Literal["pca", "subsample"] = "pca",
        faiss_metric: Literal["l2", "ip"] = "l2",
        device: str = None,
        use_flash: bool = True,
        compile: bool = True,
        model_weight_path: str | None = None,
        beta: float = 0.2,
    ):
        super().__init__(
            mode="reg",
            inf_batch_size=inf_batch_size,
            normalizer=normalizer,
            missing_indicators=missing_indicators,
            clip_sigma=clip_sigma,
            feature_reduction=feature_reduction,
            faiss_metric=faiss_metric,
            device=device,
            use_flash=use_flash,
            compile=compile,
            model_weight_path=model_weight_path,
        )
        self.beta = beta

    @torch.no_grad()
    def _predict(self, X: np.ndarray, context_size: int = 2048, seed: int | None = None, beta: float | None = None):
        beta = self.beta if beta is None else beta
        train_x, train_y, test_x = self._prepare_prediction(X, seed=seed)

        if seed is not None:
            feat_perm = generate_random_permutation(train_x.shape[1], seed)
            train_x = train_x[:, feat_perm]
            test_x = test_x[:, feat_perm]

        if context_size >= self.n_instances:
            X_train = pad_x(train_x, self.max_features).to(self.device).unsqueeze(1)  # (T_train, 1, F)
            X_test = pad_x(test_x, self.max_features).to(self.device).unsqueeze(1)    # (T_test, 1, F)
            y_train = train_y.unsqueeze(1).float()                                   # (T_train, 1)
            pred = self.model(
                x_src=torch.cat([X_train, X_test], dim=0),
                y_src=y_train.unsqueeze(-1),
                task=self.mode,
            )
            test_logits = pred[:, 0, :].float()  # (T_test, nbins)
            test_preds = predict_regression_value(test_logits, beta=beta)  # (T_test,)
            return test_preds.detach().cpu().float().numpy()
        else:
            pred_values = []
            for b in range(math.ceil(len(self.X_test) / self.inf_batch_size)):
                start = b * self.inf_batch_size
                end = min(len(self.X_test), (b + 1) * self.inf_batch_size)

                indices_nni = self.faiss_knn.get_knn_indices(self.X_test[start:end], k=context_size)
                idx = torch.as_tensor(indices_nni, device=train_x.device)
                X_nni = train_x[idx]  # (B, ctx, F)
                y_nni = train_y[idx]  # (B, ctx)

                X_nni, y_nni = (
                    pad_x(X_nni, self.max_features).to(self.device).permute(1, 0, 2),  # (ctx, B, F)
                    y_nni.to(self.device).permute(1, 0),                               # (ctx, B)
                )
                X_eval = test_x[start:end]
                X_eval = pad_x(X_eval.unsqueeze(1), self.max_features).to(self.device).permute(1, 0, 2)  # (1, B, F)
                pred = self.model(
                    x_src=torch.cat([X_nni, X_eval], dim=0),
                    y_src=y_nni.unsqueeze(-1),
                    task=self.mode,
                )
                test_logits = pred.squeeze(0).float()  # (B, nbins)
                test_vals = predict_regression_value(test_logits, beta=beta)  # (B,)
                pred_values.append(test_vals)

            return torch.cat(pred_values).detach().cpu().float().numpy()

    def _ensemble_predict(
        self,
        X: np.ndarray,
        n_ensembles: int = 8,
        context_size: int = 2048,
        seed: int | None = None,
        beta: float | None = None,
    ):
        prediction_cumsum = 0
        generator = np.random.SeedSequence(seed)
        for _, inner_seed in tqdm(zip(range(n_ensembles), generator.generate_state(n_ensembles))):
            inner_seed = int(inner_seed)
            prediction_cumsum += self._predict(X, context_size=context_size, seed=inner_seed, beta=beta)
        return prediction_cumsum / n_ensembles

    def predict(
        self,
        X: np.ndarray,
        n_ensembles: int = 8,
        context_size: int = 2048,
        seed: int | None = None,
        beta: float | None = None,
    ):
        if n_ensembles == 1:
            return self._predict(X, context_size=context_size, seed=seed, beta=beta)
        else:
            return self._ensemble_predict(X, n_ensembles=n_ensembles, context_size=context_size, seed=seed, beta=beta)
