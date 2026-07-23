import math
from typing import Literal, TypedDict, overload

import numpy as np
import torch
from sklearn.base import RegressorMixin

from .bar_distribution import BarDistribution, distribution_mean
from .estimator import TabDPTEstimator
from .utils import convert_to_torch_tensor, generate_random_permutation, normalize_data, pad_x

REGRESSION_CONSTANT_TARGET_BORDER_EPSILON = 1e-5

OutputType = Literal["mean", "full"]


class FullPrediction(TypedDict):
    logits: torch.Tensor
    criterion: BarDistribution


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

    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)
        self._initialize_regression_state()
        return self

    def _initialize_regression_state(self):
        train_y = convert_to_torch_tensor(self.y_train).float()
        _, mean_y, std_y = normalize_data(train_y, return_mean_std=True)
        self.y_train_mean_ = float(mean_y.item())
        self.y_train_std_ = float(std_y.item())

        self.is_constant_target_ = len(np.unique(self.y_train)) == 1
        self.constant_value_ = float(self.y_train[0]) if self.is_constant_target_ else None

        if self.is_constant_target_:
            border_adjustment = max(
                abs(self.constant_value_ * REGRESSION_CONSTANT_TARGET_BORDER_EPSILON),
                REGRESSION_CONSTANT_TARGET_BORDER_EPSILON,
            )
            raw_borders = torch.tensor(
                [
                    self.constant_value_ - border_adjustment,
                    self.constant_value_ + border_adjustment,
                ],
                dtype=torch.float32,
            )
            self.raw_space_bardist_ = BarDistribution(raw_borders)
            return

        z_borders = torch.linspace(
            self.model.regression_bin_min,
            self.model.regression_bin_max,
            self.model.regression_bin_count + 1,
            dtype=torch.float32,
        )
        self.raw_space_bardist_ = BarDistribution(
            z_borders * self.y_train_std_ + self.y_train_mean_
        )

    @torch.inference_mode()
    def _predict_logits(
        self,
        X: np.ndarray,
        context_size: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        train_x, train_y, test_x = self._prepare_prediction(X, seed=seed)
        num_features = torch.tensor([train_x.shape[-1]], dtype=torch.long, device=train_x.device)

        if context_size is None:
            context_size = np.inf

        if batch_size is None:
            if self.device == "cpu":
                batch_size = 4096
            else:
                batch_size = 128 * 1024

        if seed is not None:
            feat_perm = generate_random_permutation(train_x.shape[1], seed)
            train_x = train_x[:, feat_perm]
            test_x = test_x[:, feat_perm]

        train_y, _, _ = normalize_data(train_y, return_mean_std=True)

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
                pred_list.append(logits)
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
                pred_list.append(logits)

        return torch.cat(pred_list, dim=0)

    def _ensemble_logits(
        self,
        X: np.ndarray,
        n_ensembles: int = 8,
        context_size: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        prediction_cumsum = 0
        for inner_seed in self._get_ensemble_iterator(n_ensembles, seed):
            prediction_cumsum += self._predict_logits(
                X, context_size=context_size, batch_size=batch_size, seed=int(inner_seed)
            )
        return prediction_cumsum / n_ensembles

    def _handle_constant_target(self, n_samples: int, output_type: OutputType) -> np.ndarray | FullPrediction:
        constant_prediction = np.full(n_samples, self.constant_value_)
        if output_type == "mean":
            return constant_prediction
        return FullPrediction(
            logits=torch.zeros((n_samples, 1)),
            criterion=self.raw_space_bardist_,
        )

    @overload
    def predict(
        self,
        X: np.ndarray,
        *,
        output_type: Literal["mean"] = "mean",
        n_ensembles: int = 8,
        context_size: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        X: np.ndarray,
        *,
        output_type: Literal["full"],
        n_ensembles: int = 8,
        context_size: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> FullPrediction: ...

    def predict(
        self,
        X: np.ndarray,
        *,
        output_type: OutputType = "mean",
        n_ensembles: int = 8,
        context_size: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray | FullPrediction:
        """Predict regression targets or the full predictive distribution.

        By default returns the expected value (mean) of the model's binned predictive
        distribution in raw target units. Use ``output_type="full"`` to obtain the
        ensembled logits and the ``BarDistribution`` criterion in raw target space;
        further statistics (median, mode, quantiles, samples) can be derived via
        ``BarDistribution`` methods or the ``distribution_*`` helpers.

        The predictive distribution is a fixed-bin histogram over z-normalized targets
        in ``[regression_bin_min, regression_bin_max]``, mapped to raw units via
        fit-time ``y_train_mean_`` and ``y_train_std_``. All probability mass lies within
        approximately ``[mean - 10*std, mean + 10*std]`` in raw space.
        """
        if output_type not in ("mean", "full"):
            raise ValueError(f"Invalid output type: {output_type}")

        if self.is_constant_target_:
            return self._handle_constant_target(X.shape[0], output_type)

        logits = self._ensemble_logits(
            X,
            n_ensembles=n_ensembles,
            context_size=context_size,
            batch_size=batch_size,
            seed=seed,
        )
        criterion = self.raw_space_bardist_

        if output_type == "full":
            return FullPrediction(logits=logits, criterion=criterion)

        return distribution_mean(logits, criterion)
