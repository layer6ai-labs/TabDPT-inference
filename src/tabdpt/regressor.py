import math
from typing import Literal, overload

import numpy as np
import torch
from sklearn.base import RegressorMixin

from .bar_distribution import FullPrediction
from .estimator import TabDPTEstimator

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
        prep = self._preprocess(X, context_size=context_size, seed=seed, normalize_y_ctx=True)
        X_ctx, y_ctx, X_query, mean_y, std_y = prep.X_ctx, prep.y_ctx, prep.X_query, prep.mean_y, prep.std_y

        if context_size is None:
            context_size = np.inf
        if batch_size is None:
            batch_size = self._default_batch_size()

        pred_list = []
        for b in range(math.ceil(len(self.X_test) / batch_size)):
            start = b * batch_size
            end = min(len(self.X_test), (b + 1) * batch_size)

            if self._uses_stacked_context(context_size):
                X_query_batch = X_query[:, start:end]
            else:
                X_query_batch = X_query[start:end]

            pred = self.model(
                x_src=torch.cat([X_ctx, X_query_batch], dim=1),
                y_src=y_ctx,
                is_cls=False
            )
            if self._uses_stacked_context(context_size):
                pred = pred.squeeze(1)
            else:
                pred = pred.squeeze(0)
            logits = pred[:, self.max_num_classes:].float()

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

        Args:
            X: Input inference instances, `n_instances` x `n_features`.
            n_ensembles: Number of TabDPT runs to ensemble together.
            context_size: Maximum number of train points in the context. Uses all points if `None`, which can lead to
                GPU OOMs. Otherwise reduces context size based on `context_reduction` setting.
            batch_size: Number of inference points to use in each batch. If `None`, defaults to 4096 on CPU and 128k on
                GPU. Can be increased for faster inference, or decreased to prevent OOMs.
            seed: Seed used for permuting feature order during ensembling. If `n_ensembles` is 1, then feature
                permutation will only be done if this is set to a non-`None` value.
            output_type: `"mean"` returns a point prediction. `"full"` returns a `FullPrediction` with logits and
                bin borders in raw target space.

        Returns:
            If `output_type="mean"`, a point prediction vector of length `n_instances`.
            If `output_type="full"`, a `FullPrediction` dict with `logits` and `borders`.
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
