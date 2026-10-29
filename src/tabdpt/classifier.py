import math
from typing import Literal

import numpy as np
import torch
from scipy.special import softmax
from sklearn.base import ClassifierMixin

from .estimator import TabDPTEstimator
from .utils import generate_random_permutation, pad_x


class TabDPTClassifier(TabDPTEstimator, ClassifierMixin):
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
            mode="cls",
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
        self.num_classes = len(np.unique(self.y_train))
        assert self.num_classes > 1, "Number of classes must be greater than 1"

    def _predict_large_cls(self, X_context, X_eval, y_context, num_features):
        # X_context/X_eval follow the model's (B, T, F) layout; the model returns (T_eval, B, ...).
        num_digits = math.ceil(math.log(self.num_classes, self.max_num_classes))
        batch_size = X_context.shape[0]
        n_eval = X_eval.shape[1]
        digit_log_probs = []
        for digit_index in range(num_digits):
            y_digit = (y_context // (self.max_num_classes**digit_index)) % self.max_num_classes
            forward_out = self.model(
                x_src=torch.cat([X_context, X_eval], dim=1),
                y_src=y_digit,
                is_cls=True
            )
            chunk = forward_out[-n_eval:, :, : self.max_num_classes].float().detach()
            digit_log_probs.append(torch.nn.functional.log_softmax(chunk, dim=-1))
            del forward_out
        log_prob_per_class = torch.zeros((n_eval, batch_size, self.num_classes), device=X_context.device)
        for class_index in range(self.num_classes):
            acc = torch.zeros((n_eval, batch_size), device=X_eval.device)
            for digit_index, digit_lp in enumerate(digit_log_probs):
                digit_value = (class_index // (self.max_num_classes**digit_index)) % self.max_num_classes
                acc = acc + digit_lp[:, :, digit_value]
            log_prob_per_class[:, :, class_index] = acc
        return log_prob_per_class

    @torch.inference_mode()
    def predict_proba(
        self,
        X: np.ndarray,
        temperature: float = 1.,
        context_size: int | None = None,
        batch_size: int | None = None,
        return_logits: bool = False,
        seed: int | None = None,
        class_perm: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict classification output without ensembling, returning class probabilities

        Args:
            X: Input inference instances, `n_instances` x `n_features`	.
            temperature: Temperature scaling applied before softmax, can be used to calibrate probability outputs.
            context_size: Maximum number of train points in the context. Uses all points if `None`, which can lead to
                GPU OOMs. Otherwise reduces context size based on `context_reduction` setting.
            batch_size: Number of inference points to use in each batch. If `None`, defaults to 4096 on CPU and 128k on
                GPU. Can be increased for faster inference, or decreased to prevent OOMs.
            return_logits: If `True`, returns logits without temperature scaling or softmax.
            seed: Seed used for permuting feature/class order during ensembling. If `n_ensembles` is 1, then feature
                permutation will only be done if this is set to a non-`None` value.
            class_perm: An optional permutation of classes to apply before prediction, mainly for ensembling.

        Returns:
            A class probability array of size `n_instances` x `n_classes`.
        """
        prep = self._preprocess(X, context_size=context_size, seed=seed, class_perm=class_perm)
        X_ctx, y_ctx, X_query = prep.X_ctx, prep.y_ctx, prep.X_query
        num_features = torch.tensor([X_ctx.shape[-1]], dtype=torch.long, device=X_ctx.device)

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
                X_ctx_batch = X_ctx
                y_ctx_batch = y_ctx
            else:
                X_query_batch = X_query[start:end]
                X_ctx_batch = X_ctx[prep.indices_nni[start:end]]
                y_ctx_batch = y_ctx[prep.indices_nni[start:end]]

            if self.num_classes <= self.max_num_classes:
                pred = self.model(
                    x_src=torch.cat([X_ctx_batch, X_query_batch], dim=1),
                    y_src=y_ctx_batch,
                    is_cls=True
                )
                pred = torch.nn.functional.log_softmax(pred[..., : self.num_classes].float(), dim=-1)
            else:
                pred = self._predict_large_cls(X_ctx_batch, X_query_batch, y_ctx_batch, num_features)

            if not return_logits:
                pred = pred[..., :self.num_classes] / temperature
                pred = torch.nn.functional.softmax(pred.float(), dim=-1)

            if self._uses_stacked_context(context_size):
                pred_list.append(pred.squeeze(1))
            else:
                if not return_logits:
                    # numerical stability - not sure if needed now, but from legacy code
                    pred /= pred.sum(axis=-1, keepdims=True)
                pred_list.append(pred.squeeze(0))

        pred_val = torch.cat(pred_list, dim=0).detach().cpu().float().numpy()

        # Ensure pred_val is always (n_samples, n_classes) even for a single sample.
        if pred_val.ndim == 1:
            pred_val = pred_val[None, :]  # (n_classes,) -> (1, n_classes)
        return pred_val

    def ensemble_predict_proba(
        self,
        X,
        n_ensembles: int = 8,
        temperature: float = 1.,
        context_size: int | None = None,
        batch_size: int | None = None,
        permute_classes: bool = True,
        seed: int | None = None,
    ) -> np.ndarray:
        """Predict classification output using an ensemble of models, returning class probabilities

        Args:
            X: Input inference instances, `n_instances` x `n_features`.
            n_ensembles: Number of TabDPT runs to ensemble together.
            temperature: Temperature scaling applied before softmax, can be used to calibrate probability outputs.
            context_size: Maximum number of train points in the context. Uses all points if `None`, which can lead to
                GPU OOMs. Otherwise reduces context size based on `context_reduction` setting.
            batch_size: Number of inference points to use in each batch. If `None`, defaults to 4096 on CPU and 128k on
                GPU. Can be increased for faster inference, or decreased to prevent OOMs.
            permute_classes: Whether to permute classes during ensembled prediction.
            seed: Seed used for permuting feature/class order during ensembling.

        Returns:
            A class probability array of size `n_instances` x `n_classes`.
        """
        if n_ensembles == 1:
            return self.predict_proba(
                X,
                temperature=temperature,
                context_size=context_size,
                batch_size=batch_size,
                seed=seed
            )

        logit_cumsum = None
        for inner_seed in self._get_ensemble_iterator(n_ensembles, seed):
            inner_seed = int(inner_seed)
            perm = torch.arange(self.num_classes)
            if permute_classes:
                perm = generate_random_permutation(self.num_classes, inner_seed)
            inv_perm = np.argsort(perm)

            logits = self.predict_proba(
                X,
                context_size=context_size,
                batch_size=batch_size,
                return_logits=True,
                seed=inner_seed,
                class_perm=perm,
            )
            logits = logits[..., inv_perm]
            if logit_cumsum is None:
                logit_cumsum = np.zeros_like(logits)
            logit_cumsum += logits

        logits = (logit_cumsum / n_ensembles)[..., :self.num_classes] / temperature
        pred = softmax(logits, axis=-1)
        pred /= pred.sum(axis=-1, keepdims=True)
        return pred

    def predict(
        self,
        X,
        n_ensembles: int = 8,
        temperature: float = 1.,
        context_size: int | None = None,
        batch_size: int | None = None,
        permute_classes: bool = True,
        seed: int | None = None,
    ) -> np.ndarray:
        """Predict classification output, returning class indices

        Args:
            X: Input inference instances, `n_instances` x `n_features`.
            n_ensembles: Number of TabDPT runs to ensemble together.
            temperature: Temperature scaling applied before softmax, can be used to calibrate probability outputs. Does
                not affect class index predictions.
            context_size: Maximum number of train points in the context. Uses all points if `None`, which can lead to
                GPU OOMs. Otherwise reduces context size based on `context_reduction` setting.
            batch_size: Number of inference points to use in each batch. If `None`, defaults to 4096 on CPU and 128k on
                GPU. Can be increased for faster inference, or decreased to prevent OOMs.
            permute_classes: Whether to permute classes during ensembled prediction.
            seed: Seed used for permuting feature/class order during ensembling. If `n_ensembles` is 1, then feature
                permutation will only be done if this is set to a non-`None` value.

        Returns:
            A class index vector of length `n_instances`.
        """
        if n_ensembles == 1:
            return self.predict_proba(
                X,
                temperature=temperature,
                context_size=context_size,
                batch_size=batch_size,
                seed=seed
            ).argmax(axis=-1)
        else:
            return self.ensemble_predict_proba(
                X,
                n_ensembles=n_ensembles,
                temperature=temperature,
                context_size=context_size,
                batch_size=batch_size,
                permute_classes=permute_classes,
                seed=seed,
            ).argmax(axis=-1)
