import math
import warnings
from captum.attr import IntegratedGradients
import numpy as np
import torch

from .classifier import TabDPTClassifier
from .regressor import TabDPTRegressor
from .utils import normalize_data

def integrated_gradients(
    fitted_model: TabDPTRegressor | TabDPTClassifier,
    X_test: np.ndarray,
    class_index: int = 0,
    batch_size: int | None = None,
    n_steps: int = 50,
    context_size: int | None = None,
    seed: int | None = None,
    class_perm: np.ndarray | None = None
) -> np.ndarray:
    """
    Runs integrated gradients for one output and returns local attributions.

    Attributions for classification are relative to logit space (normalized by log_softmax), and attributions for
    regression are relative to output values after normalization by context target mean/std.

    Options for baselines are not included because by default, mean/std scaling is applied, meaning that a zero baseline
    is reasonable, and using the model without similar scaling would be an uncommon use case.

    Retrieval is not supported since the context will no longer be relevant when testing points along the integration
    path. Support for high numbers of classes is not yet implemented, although integrated gradients does not scale well
    to high numbers of classes anyway.

    Args:
        fitted_model: TabDPT model after running .fit
        X_test
        class_index: For classifiers, the index of the class to produce attributions for. Logits are normalized using
            log_softmax, so for the binary classification case, only explaining either class index 0 or 1 should suffice,
            but multiple runs could be required for >2 classes.
        batch_size: Internal test batch size for forward and backward passes. The context is shared and batching
            across the sequence dimension is used. Note that IG requires `n_steps*X_test.shape[0]` actual test points, so
            larger batch sizes than the size of `X_test` can be used if memory permits.
        n_steps: Number of steps to integrate over
        context_size, seed, class_perm: Same meaning as in predict functions

    Returns:
        Local attributions of size `n_test` x `n_features`
    """

    if context_size is not None and not fitted_model._uses_stacked_context(context_size):
        raise NotImplementedError(
            "Integrated gradients only supports shared contexts for test points. Note that IG with retrieval is not "
            "well-defined since the context depends on the test point values."
        )

    is_cls = type(fitted_model) is TabDPTClassifier
    if is_cls and fitted_model.num_classes > fitted_model.max_num_classes:
        raise NotImplementedError(
            "Integrated gradients not implemented for high numbers of classes (requiring multiple forward passes)"
        )

    if fitted_model.scaler is None:
        warnings.warn(
            "Using integrated gradients with no scaler will likely produce invalid results since we assume a zero baseline"
        )

    if batch_size is None:
        batch_size = fitted_model._default_batch_size()

    prep = fitted_model._preprocess(
        X_test,
        context_size=context_size,
        seed=seed,
        class_perm=class_perm
    )
    X_ctx, y_ctx, X_query, feat_perm = prep.X_ctx, prep.y_ctx, prep.X_query, prep.feat_perm
    feat_perm = feat_perm.cpu().numpy()

    def cls_wrapped_forward(X_query):
        x_src = torch.cat([X_ctx, X_query.unsqueeze(0)], dim=1)
        pred = fitted_model.model(x_src=x_src, y_src=y_ctx, is_cls=True)
        return torch.nn.functional.log_softmax(pred[..., :fitted_model.num_classes], dim=-1).squeeze(1)

    def reg_wrapped_forward(X_query):
        y_ctx_norm, mean_y, std_y = normalize_data(y_ctx, dim=1, return_mean_std=True)
        x_src = torch.cat([X_ctx, X_query.unsqueeze(0)], dim=1)
        pred = fitted_model.model(x_src=x_src, y_src=y_ctx_norm, is_cls=False)
        logits = pred.squeeze(1)[:, fitted_model.max_num_classes:].float()
        ret = fitted_model._expectation_from_regression_logits(logits)
        return ret

    forward = cls_wrapped_forward if is_cls else reg_wrapped_forward
    ig = IntegratedGradients(forward)

    attrs = []
    for b in range(math.ceil(X_query.shape[1] / batch_size)):
        attrs.append(ig.attribute(
            X_query.squeeze(0)[b*batch_size : (b+1)*batch_size],
            internal_batch_size=batch_size,
            n_steps=n_steps,
            target=class_index if is_cls else None
        ).cpu().numpy())
    attrs = np.concatenate(attrs)

    if fitted_model.n_features > fitted_model.max_features and fitted_model.feature_reduction == "pca":
        # In the case of PCA projection, IG is done with respect to the principal components: u = x @ V, so divide by u
        # to get the integral, multiply by V.T to make the inner derivative in the IG formula be with respect to x
        # (chain rule), then multiply by x.
        pca_X = X_query.squeeze(0).cpu().numpy()
        integral = np.divide(attrs, pca_X, out=np.zeros_like(attrs), where=pca_X!=0) # Avoid divide by zero
        inv_perm = np.argsort(feat_perm)
        integral = integral[:,inv_perm]
        proj = fitted_model.projection.cpu().numpy()
        # Padding to max_features in _preprocess can lead to X_query/integral being larger than projection matrix
        attrs = integral[:, :proj.shape[1]] @ proj.T * fitted_model.X_test
    else:
        inv_perm = np.argsort(feat_perm)
        # If feature subsampling is used, feat_perm will be longer than attrs, so we need to add zero attributions
        if attrs.shape[1] < len(inv_perm):
            attrs = np.pad(attrs, ((0, 0), (0, len(inv_perm) - attrs.shape[1])))
        attrs = attrs[:,inv_perm]

    if fitted_model.missing_indicators and any(fitted_model.has_missing_indicator):
        n_missing_indicators = np.sum(fitted_model.has_missing_indicator)
        feat_attrs = attrs[:, :-n_missing_indicators]
        feat_attrs[:, fitted_model.has_missing_indicator] += attrs[:, -n_missing_indicators:]
        return feat_attrs

    return attrs
