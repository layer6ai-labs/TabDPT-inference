import warnings
from typing import Literal

import numpy as np

from .classifier import TabDPTClassifier

UNLABELLED = -1


class TabDPTSelfTrainingClassifier(TabDPTClassifier):
    """Few-shot TabDPT classifier using self-training, following Section 4.6 of the TabDPT paper.

    `fit` takes `y` with unlabelled rows marked `-1`. It pseudo-labels them using the labelled rows as
    context, keeps the `n_pseudo` most confident, and refits on both. Prediction is unchanged.

    Attributes:
        pseudo_label_indices_: Indices into `X` of the pseudo-labelled rows, most confident first.
        pseudo_labels_: Labels assigned to those rows.
    """

    def __init__(
        self,
        n_pseudo: int = 1000,
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
        """Initializes the self-training TabDPT classifier

        Args:
            n_pseudo: Maximum number of pseudo-labelled rows to add to the context. Clamped to the
                number of unlabelled rows available.

        All remaining arguments are passed through to `TabDPTClassifier` unchanged.
        """
        # Validate before loading the checkpoint, so a bad argument fails immediately.
        assert n_pseudo > 0, "n_pseudo must be positive"
        super().__init__(
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
        self.n_pseudo = n_pseudo

    def _select_pseudo_labelled(self, proba: np.ndarray) -> np.ndarray:
        """Choose which pseudo-labelled rows to trust, most confident first.

        Ranks all unlabelled rows together by confidence, rather than taking a quota per class.

        Args:
            proba: Predicted class probabilities for the unlabelled rows, `n_unlabelled` x
                `n_classes`.

        Returns:
            Indices into `proba`, ordered from most to least confident, at most `n_pseudo` of them.
        """
        return np.argsort(-proba.max(axis=1))[: self.n_pseudo]

    def fit(self, X: np.ndarray, y: np.ndarray, n_ensembles: int = 8, seed: int | None = None):
        """Fit on a partially labelled training set, expanding the context with pseudo-labels

        Args:
            X: Feature matrix, `n_instances` x `n_features`.
            y: Labels encoded as integers in `[0, n_classes)`, with `-1` marking unlabelled rows.
            n_ensembles: Number of TabDPT runs to ensemble when predicting the pseudo-labels. Does
                not affect `predict`, which takes its own `n_ensembles`.
            seed: Seed used for the pseudo-labelling pass.
        """
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert y.ndim == 1, "y must be a 1D array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert n_ensembles > 0, "n_ensembles must be positive"

        is_labelled = y != UNLABELLED
        classes = np.unique(y[is_labelled])
        if len(classes) < 2:
            raise ValueError("y must contain at least two labelled classes")
        # `TabDPTClassifier.predict` returns argmax indices, so it assumes labels are 0..n_classes-1.
        # That assumption is silent in the parent, but here it would corrupt the context: the
        # pseudo-labels below are argmax indices concatenated back onto `y`.
        if not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError(
                f"labelled targets must be integers in [0, n_classes), got {classes.tolist()}. "
                "Label-encode y before fitting, leaving unlabelled rows as -1."
            )

        labelled, unlabelled = np.flatnonzero(is_labelled), np.flatnonzero(~is_labelled)

        # Stage 1: fit on the labelled rows alone.
        super().fit(X[labelled], y[labelled])
        if len(unlabelled) == 0:
            warnings.warn(
                "y contains no unlabelled rows (-1), so there is nothing to pseudo-label; "
                "falling back to a standard TabDPTClassifier fit.",
                stacklevel=2,
            )
            self.pseudo_label_indices_, self.pseudo_labels_ = unlabelled, y[unlabelled]
            return

        # Pseudo-label the unlabelled pool and keep the rows the model is most confident about.
        proba = self.ensemble_predict_proba(X[unlabelled], n_ensembles=n_ensembles, seed=seed)
        kept = self._select_pseudo_labelled(proba)
        pseudo_label_indices = unlabelled[kept]
        pseudo_labels = proba[kept].argmax(axis=1).astype(y.dtype)

        # Stage 2: refit on the original labelled rows plus the rows we chose to trust. The attributes
        # are set only once this succeeds, so they always describe the context actually in use.
        super().fit(
            X[np.concatenate([labelled, pseudo_label_indices])],
            np.concatenate([y[labelled], pseudo_labels]),
        )
        self.pseudo_label_indices_, self.pseudo_labels_ = pseudo_label_indices, pseudo_labels
