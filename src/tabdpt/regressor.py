import math
from typing import Literal, overload

import numpy as np
import torch
from sklearn.base import RegressorMixin

from .bar_distribution import FullPrediction
from .estimator import TabDPTEstimator
from .utils import generate_random_permutation, pad_x, normalize_data

OutputType = Literal["mean", "full"]


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

    def _borders_from_norm_stats(self, mean_y: torch.Tensor, std_y: torch.Tensor) -> torch.Tensor:
        borders = torch.linspace(
            self.model.regression_bin_min,
            self.model.regression_bin_max,
            self.model.regression_bin_count + 1,
            device=mean_y.device,
            dtype=mean_y.dtype,
        )
        return borders * std_y + mean_y

    @torch.inference_mode()
    def _predict(
        self,
        X: np.ndarray,
        context_size: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        return_logits: bool = False,
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
                logits = pred.squeeze(1)[:, self.max_num_classes:].float()
                if return_logits:
                    pred_list.append(logits)
                else:
                    y_hat = self._expectation_from_regression_logits(logits)
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
                logits = pred.squeeze(0)[:, self.max_num_classes:].float()
                if return_logits:
                    pred_list.append(logits)
                else:
                    y_hat = self._expectation_from_regression_logits(logits)
                    pred_list.append(y_hat * std_y + mean_y)

        preds = torch.cat(pred_list, dim=0)
        if return_logits:
            return preds, mean_y, std_y
        return preds.detach().cpu().numpy()

    def _ensemble_predict(
            self,
            X: np.ndarray,
            n_ensembles: int = 8,
            context_size: int | None = None,
            batch_size: int | None = None,
            seed: int | None = None,
            return_logits: bool = False,
        ):
        prediction_cumsum = 0
        mean_y = std_y = None
        for inner_seed in self._get_ensemble_iterator(n_ensembles, seed):
            inner_seed = int(inner_seed)
            if return_logits:
                logits, mean_y, std_y = self._predict(
                    X, context_size=context_size, batch_size=batch_size, seed=inner_seed, return_logits=True
                )
                prediction_cumsum += logits
            else:
                prediction_cumsum += self._predict(
                    X, context_size=context_size, batch_size=batch_size, seed=inner_seed
                )

        if return_logits:
            return prediction_cumsum / n_ensembles, mean_y, std_y
        return prediction_cumsum / n_ensembles

    @overload
    def predict(
            self,
            X: np.ndarray,
            n_ensembles: int = 8,
            context_size: int | None = None,
            batch_size: int | None = None,
            seed: int | None = None,
            *,
            output_type: Literal["mean"] = "mean",
        ) -> np.ndarray: ...

    @overload
    def predict(
            self,
            X: np.ndarray,
            n_ensembles: int = 8,
            context_size: int | None = None,
            batch_size: int | None = None,
            seed: int | None = None,
            *,
            output_type: Literal["full"],
        ) -> FullPrediction: ...

    def predict(
            self,
            X: np.ndarray,
            n_ensembles: int = 8,
            context_size: int | None = None,
            batch_size: int | None = None,
            seed: int | None = None,
            *,
            output_type: OutputType = "mean",
        ) -> np.ndarray | FullPrediction:
        """Predict regression targets, or the full predictive distribution.

        Default `output_type="mean"` matches the original point-prediction path.
        `output_type="full"` returns ensembled logits and bin borders in raw target space
        (for median/mode/quantiles/samples).
        """
        if output_type not in ("mean", "full"):
            raise ValueError(f"Invalid output type: {output_type}")

        if output_type == "full":
            if n_ensembles == 1:
                logits, mean_y, std_y = self._predict(
                    X, context_size=context_size, batch_size=batch_size, seed=seed, return_logits=True
                )
            else:
                logits, mean_y, std_y = self._ensemble_predict(
                    X,
                    n_ensembles=n_ensembles,
                    context_size=context_size,
                    batch_size=batch_size,
                    seed=seed,
                    return_logits=True,
                )
            return FullPrediction(
                logits=logits,
                borders=self._borders_from_norm_stats(mean_y, std_y),
            )

        if n_ensembles == 1:
            return self._predict(X, context_size=context_size, batch_size=batch_size, seed=seed)
        else:
            return self._ensemble_predict(X, n_ensembles=n_ensembles, context_size=context_size,
                batch_size=batch_size, seed=seed)
