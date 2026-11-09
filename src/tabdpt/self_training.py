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
        pseudo_label_indices_: Indices into `X` of the pseudo-labelled rows.
        pseudo_labels_: Labels assigned to those rows.
    """

    def __init__(self, n_pseudo: int = 1000, selection: Literal["global", "per-class"] = "global", **kwargs):
        """Initializes the self-training TabDPT classifier

        Args:
            n_pseudo: Maximum number of pseudo-labelled rows to add to the context. Clamped to the
                number of unlabelled rows available.
            selection: `"global"` keeps the most confident rows overall, as in the paper. `"per-class"`
                ranks rows within each predicted class and keeps `n_pseudo // n_classes` (at least one) from
                each, so a class the model is over-confident about cannot take most of the slots.
            **kwargs: Passed through to `TabDPTClassifier`.
        """
        # Validate before loading the checkpoint, so a bad argument fails immediately.
        assert n_pseudo > 0, "n_pseudo must be positive"
        assert selection in ("global", "per-class"), f"selection must be 'global' or 'per-class', got {selection!r}"
        super().__init__(**kwargs)
        self.n_pseudo = n_pseudo
        self.selection = selection

    @classmethod
    def _get_param_names(cls):
        # sklearn's get_params() and clone() read parameter names from the __init__ signature, which
        # would miss everything passed through **kwargs.
        return sorted(set(TabDPTClassifier._get_param_names()) | set(super()._get_param_names()))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_ensembles: int = 8,
        context_size: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ):
        """Fit on a partially labelled training set, expanding the context with pseudo-labels

        Args:
            X: Feature matrix, `n_instances` x `n_features`.
            y: Labels encoded as integers in `[0, n_classes)`, with `-1` marking unlabelled rows.
            n_ensembles: Number of TabDPT runs to ensemble when predicting the pseudo-labels. Does
                not affect `predict`, which takes its own `n_ensembles`.
            context_size: Maximum number of labelled rows in the context when predicting the pseudo-labels.
            batch_size: Number of unlabelled rows to predict in each batch. Can be decreased to prevent OOMs.
            seed: Seed used for the pseudo-labelling pass.
        """
        is_labelled = y != UNLABELLED
        classes = np.unique(y[is_labelled])
        if len(classes) < 2:
            raise ValueError("y must contain at least two labelled classes")
        # `TabDPTClassifier.predict` returns argmax indices, so it assumes labels are 0..n_classes-1.
        # Here that would corrupt the context, since the pseudo-labels below are argmax indices.
        if not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError(
                f"labelled targets must be integers in [0, n_classes), got {classes.tolist()}. "
                "Label-encode y before fitting, leaving unlabelled rows as -1."
            )
        unlabelled = np.nonzero(~is_labelled)[0]

        # Stage 1: fit on the labelled rows alone.
        super().fit(X[is_labelled], y[is_labelled])
        if len(unlabelled) == 0:
            warnings.warn(
                "y contains no unlabelled rows (-1), so there is nothing to pseudo-label; "
                "falling back to a standard TabDPTClassifier fit.",
                stacklevel=2,
            )
            self.pseudo_label_indices_, self.pseudo_labels_ = unlabelled, y[unlabelled]
            return

        # Pseudo-label the unlabelled rows and keep the ones the model is most confident about.
        proba = self.ensemble_predict_proba(
            X[unlabelled], n_ensembles=n_ensembles, context_size=context_size, batch_size=batch_size, seed=seed
        )
        predicted, confidence = proba.argmax(axis=1), proba.max(axis=1)
        if self.selection == "global":
            kept = np.argsort(-confidence)[: self.n_pseudo]
        else:
            per_class = max(1, self.n_pseudo // len(classes))
            kept = np.concatenate([
                rows[np.argsort(-confidence[rows])[:per_class]]
                for rows in (np.nonzero(predicted == c)[0] for c in range(len(classes)))
            ])
        pseudo_label_indices, pseudo_labels = unlabelled[kept], predicted[kept].astype(y.dtype)

        # Stage 2: refit on the labelled rows plus the pseudo-labelled ones. The attributes are set only
        # once this succeeds.
        super().fit(
            np.concatenate([X[is_labelled], X[pseudo_label_indices]]),
            np.concatenate([y[is_labelled], pseudo_labels]),
        )
        self.pseudo_label_indices_, self.pseudo_labels_ = pseudo_label_indices, pseudo_labels
