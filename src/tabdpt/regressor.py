import math
from typing import Literal

import numpy as np
import torch
from sklearn.base import RegressorMixin

from .estimator import TabDPTEstimator
from .utils import generate_random_permutation, pad_x, normalize_data


class TabDPTRegressor(TabDPTEstimator, RegressorMixin):
    def __init__(
        self,
        normalizer: Literal["standard", "minmax", "robust", "power", "quantile-uniform", "quantile-normal", "log1p"] | None
            = "standard",
        missing_indicators: bool = False,
        clip_sigma: float = 8.,
        feature_reduction: Literal["pca", "subsample"] = "pca",
        context_reduction: Literal["retrieval", "subsample", "subsample-balanced"] = "subsample",
        faiss_metric: Literal["l2", "ip"] = "l2",
        device: str = None,
        use_flash: bool = True,
        compile: bool = True,
        model_weight_path: str | None = None,
        verbose: bool = True,
    ):
        super().__init__(
            mode="reg",
            normalizer=normalizer,
            missing_indicators=missing_indicators,
            clip_sigma=clip_sigma,
            feature_reduction=feature_reduction,
            context_reduction=context_reduction,
            faiss_metric=faiss_metric,
            device=device,
            use_flash=use_flash,
            compile=compile,
            model_weight_path=model_weight_path,
            verbose=verbose,
        )

    def _expectation_from_regression_logits(self, reg_logits: torch.Tensor) -> torch.Tensor:
        edges = torch.linspace(
            self.model.regression_bin_min,
            self.model.regression_bin_max,
            self.model.regression_bin_count + 1,
            device=reg_logits.device,
            dtype=reg_logits.dtype
        )
        bin_centres = 0.5 * (edges[:-1] + edges[1:])
        weights = torch.softmax(reg_logits.float(), dim=-1)
        return (weights * bin_centres).sum(dim=-1)

    @torch.inference_mode()
    def _predict(
        self,
        X: np.ndarray,
        context_size: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None
    ):
        train_x, train_y, test_x = self._prepare_prediction(X, seed=seed)
        num_features = torch.tensor([train_x.shape[-1]], dtype=torch.long, device=train_x.device)

        if context_size is None:
            context_size = np.inf

        if batch_size is None:
            if self.device == "cpu":
                batch_size = 4096
            else:
                batch_size = 128*1024

        if seed is not None:
            feat_perm = generate_random_permutation(train_x.shape[1], seed)
            train_x = train_x[:, feat_perm]
            test_x = test_x[:, feat_perm]

        train_y, mean_y, std_y = normalize_data(train_y, return_mean_std=True)

        pred_list = []
        if self._uses_stacked_context(context_size):
            X_ctx, y_ctx = self._prepare_stacked_context(train_x, train_y, context_size, seed)
            X_test = pad_x(test_x[None, :, :], self.max_features).to(self.device)

            for b in range(math.ceil(len(self.X_test) / batch_size)):
                start = b * batch_size
                end = min(len(self.X_test), (b + 1) * batch_size)

                pred = self.model(
                    x_src=torch.cat([X_ctx, X_test[:, start:end]], dim=1),
                    y_src=y_ctx,
                    num_features=num_features,
                )
                y_hat = self._expectation_from_regression_logits(
                    pred.squeeze(1)[:, self.max_num_classes:].float()
                )
                pred_list.append(y_hat * std_y + mean_y)
        else:
            for b in range(math.ceil(len(self.X_test) / batch_size)):
                start = b * batch_size
                end = min(len(self.X_test), (b + 1) * batch_size)

                indices_nni = self._get_context_indices(
                    self.X_test[start:end], context_size=context_size, seed=seed
                )
                X_nni = train_x[torch.tensor(indices_nni)]
                y_nni = train_y[torch.tensor(indices_nni)]

                X_nni, y_nni = (
                    pad_x(torch.Tensor(X_nni), self.max_features).to(self.device),
                    torch.Tensor(y_nni).to(self.device),
                )
                X_eval = test_x[start:end]
                X_eval = pad_x(X_eval.unsqueeze(1), self.max_features).to(self.device)
                pred = self.model(
                    x_src=torch.cat([X_nni, X_eval], dim=1),
                    y_src=y_nni,
                    num_features=num_features,
                )
                y_hat = self._expectation_from_regression_logits(pred.squeeze(0)[:, self.max_num_classes:].float())
                pred_list.append(y_hat * std_y + mean_y)

        return torch.cat(pred_list).detach().cpu().numpy()

    def _ensemble_predict(
            self,
            X: np.ndarray,
            n_ensembles: int = 8,
            context_size: int | None = None,
            batch_size: int | None = None,
            seed: int | None = None,
        ):
        prediction_cumsum = 0
        for inner_seed in self._get_ensemble_iterator(n_ensembles, seed):
            inner_seed = int(inner_seed)
            prediction_cumsum += self._predict(X, context_size=context_size, batch_size=batch_size, seed=inner_seed)

        return prediction_cumsum / n_ensembles

    def predict(
            self,
            X: np.ndarray,
            n_ensembles: int = 8,
            context_size: int | None = None,
            batch_size: int | None = None,
            seed: int | None = None,
        ) -> np.ndarray:
        """Predict regression output, returning a point prediction

        Args:
            X: Input inference instances, `n_instances` x `n_features`.
            n_ensembles: Number of TabDPT runs to ensemble together.
            context_size: Maximum number of train points in the context. Uses all points if `None`, which can lead to
                GPU OOMs. Otherwise reduces context size based on `context_reduction` setting.
            batch_size: Number of inference points to use in each batch. If `None`, defaults to 4096 on CPU and 128k on
                GPU. Can be increased for faster inference, or decreased to prevent OOMs.
            seed: Seed used for permuting feature order during ensembling. If `n_ensembles` is 1, then feature
                permutation will only be done if this is set to a non-`None` value.

        Returns:
            A point prediction vector of length n_instances.
        """
        if n_ensembles == 1:
            return self._predict(X, context_size=context_size, batch_size=batch_size, seed=seed)
        else:
            return self._ensemble_predict(X, n_ensembles=n_ensembles, context_size=context_size,
                batch_size=batch_size, seed=seed)
