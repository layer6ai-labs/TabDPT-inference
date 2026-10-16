from __future__ import annotations
import json
from typing import Literal
from dataclasses import dataclass

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors import safe_open
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from .model import TabDPTModel
from .utils import convert_to_torch_tensor, Log1pScaler, generate_random_permutation, pad_x, normalize_data

# Constants for model caching and download
_VERSION = "1_3"
_MODEL_NAME = f"tabdpt{_VERSION}.safetensors"
_HF_REPO_ID = "Layer6/TabDPT"


class TabDPTEstimator(BaseEstimator):
    @staticmethod
    def download_weights() -> str:
        path = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=_MODEL_NAME,
        )
        return path

    def __init__(
        self,
        mode: Literal["cls", "reg"],
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
        """
        Initializes the TabDPT Estimator

        Args:
            mode: Defines what mode the estimator is
                "cls" is classification, "reg" is regression
            normalizer: Specifies normalization used for preprocessing before retrieval. Note that
                the model performs additional normalization in its forward function. By default the
                scikit-learn `StandardScaler` is used, which matches model training. Other options are:
                - "minmax": scikit-learn `MinMaxScaler(feature_range=(-1,1))`
                - "robust": scikit-learn `RobustScaler()`
                - "power": scikit-learn `PowerTransformer()`
                - "quantile-uniform": scikit-learn `QuantileTransformer(output_distribution="uniform")`, rescaled to (-1,1)
                - "quantile-normal": scikit-learn `QuantileTransformer(output_distribution="normal")`
                - "log1p": `sign(X) * log(1 + abs(X))`
                - None: no normalization
            missing_indicators: If True, adds an additional binary column for each feature with
                missing values indicating their position.
            clip_sigma: n*sigma used for outlier clipping
            feature_reduction: Method used to reduce the number of features when over the model's
                limit, either "pca" or "subsample"
            context_reduction: Method used to reduce training rows when train size exceeds
                context_size at predict time:
                - "subsample": random subset of training rows (shared across test points);
                  stacked on the sequence axis with eval rows, same as full-context inference
                - "subsample-balanced": class-balanced random subset (classification only);
                  same stacked layout as subsample
                - "retrieval": per test point, nearest training rows in feature space (FAISS);
                  batched with one eval row per test point. When using retrieval, memory usage
                  scales much faster with the number of inference instances, so a lower
                  batch_size might have to be used.
            faiss_metric: Distance used for kNN context reduction, either "l2" or "ip"
            device: Specifies the computational device (e.g., CPU, GPU)
                Identical to https://docs.pytorch.org/docs/stable/generated/torch.cuda.device.html
            use_flash: Specifies whether to use flash attention or not
            compile: Specifies whether to compile the model with torch before inference
            model_weight_path: path on file system specifying the model weights
                If no path is specified, then the model weights are downloaded from HuggingFace
            verbose: Specifies whether to add tqdm looping to ensemble estimator
        """
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_flash = use_flash and self.device == "cuda"
        self.missing_indicators = missing_indicators
        self.clip_sigma = clip_sigma
        self.model_weight_path = model_weight_path

        self._load_model()

        self.max_features = self.model.num_features
        self.max_num_classes = self.model.n_out
        self.compile = compile and self.device == "cuda"
        self.feature_reduction = feature_reduction
        self.context_reduction = context_reduction
        self.faiss_metric = faiss_metric
        self.faiss_knn = None

        assert self.mode in ["cls", "reg"], "mode must be 'cls' or 'reg'"
        assert self.feature_reduction in ["pca", "subsample"], \
                "feature_reduction must be 'pca' or 'subsample'"
        assert self.context_reduction in ["retrieval", "subsample", "subsample-balanced"], \
                "context_reduction must be 'retrieval', 'subsample', or 'subsample-balanced'"
        if self.mode == "reg" and self.context_reduction == "subsample-balanced":
            raise ValueError("context_reduction='subsample-balanced' is only supported for classification")
        assert self.faiss_metric in ["l2", "ip"], 'faiss_metric must be "l2" or "ip"'

        self.verbose = verbose

        self.normalizer = normalizer
        match normalizer:
            case "standard":
                self.scaler = StandardScaler()
            case "minmax":
                self.scaler = MinMaxScaler(feature_range=(-1,1))
            case "robust":
                self.scaler = RobustScaler()
            case "power":
                self.scaler = PowerTransformer()
            case "quantile-uniform":
                self.scaler = QuantileTransformer(output_distribution="uniform")
            case "quantile-normal":
                self.scaler = QuantileTransformer(output_distribution="normal")
            case "log1p":
                self.scaler = Log1pScaler()
            case None:
                self.scaler = None
            case _:
                raise ValueError(
                    'normalizer must be one of '
                    '["standard", "minmax", "robust", "power", "quantile-uniform", "quantile-normal", "log1p", None]'
                )

        self.projection = None

    def _load_model(self) -> None:
        """Read the checkpoint and build ``self.model``; the constructor calls this once.

        Separated out so a caller can share one network across several estimators built with the
        same ``model_weight_path`` (or the same default download), ``use_flash`` and ``clip_sigma``:
        build one estimator, then for the next skip its own load by overriding or monkeypatching
        this method to hand back the cached module instead of rebuilding it. Downloads from
        Hugging Face when ``model_weight_path`` is unset.
        """
        if self.model_weight_path:
            self.path = self.model_weight_path
        else:
            self.path = self.download_weights()

        with safe_open(self.path, framework="pt", device=self.device) as f:
            meta = f.metadata()
            cfg_dict = json.loads(meta["cfg"])
            cfg = OmegaConf.create(cfg_dict)
            model_state = {k: f.get_tensor(k) for k in f.keys()}

        cfg.env.device = self.device
        self.model = TabDPTModel.load(
            model_state=model_state, config=cfg, use_flash=self.use_flash, clip_sigma=self.clip_sigma
        )
        self.model.eval()

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"

        if self.missing_indicators:
            inds = np.isnan(X)
            self.has_missing_indicator = inds.any(axis=0)
            inds = inds[:, self.has_missing_indicator].astype(float)
            X = np.hstack((X, inds))

        self.imputer = SimpleImputer(strategy="mean")
        X = self.imputer.fit_transform(X)
        if self.scaler:
            X = self.scaler.fit_transform(X)
        if self.normalizer == 'quantile-uniform':
            X = 2*X - 1

        self.n_instances, self.n_features = X.shape
        self.X_train = X
        self.y_train = y
        if self.n_features > self.max_features and self.feature_reduction == "pca":
            train_x = convert_to_torch_tensor(self.X_train).to(self.device).float()
            _, _, self.projection = torch.pca_lowrank(train_x, q=min(train_x.shape[0], self.max_features))

        # Reset in case the model was previously fit, this will be lazily initialized
        self.faiss_knn = None

        self.is_fitted_ = True
        if self.compile:
            self.model.compile()

    def to(self, device: str):
        self.device = device
        self.model.to(device)

        if self.projection is not None:
            self.projection = self.projection.to(device)

    def _get_faiss_knn_indices(self, X_test: np.ndarray, context_size: int, seed: int | None = None):
        if self.faiss_knn is None:  # Lazily perform initialization
            from .utils import FAISS
            self.faiss_knn = FAISS(self.X_train, metric=self.faiss_metric)
            if seed is not None:
                self.faiss_knn.index.seed = seed

        return self.faiss_knn.get_knn_indices(X_test, k=context_size)

    def _get_subsample_indices(self, context_size: int, seed: int | None = None) -> np.ndarray:
        perm = generate_random_permutation(self.n_instances, seed)
        return perm[:context_size].cpu().numpy()

    def _get_balanced_subsample_indices(self, context_size: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # Identify unique classes and indices for each class in y_train
        classes = np.unique(self.y_train)
        class_indices = {c: np.where(self.y_train == c)[0] for c in classes}

        per_class = context_size // len(classes)  # Minimum samples per class
        remainder = context_size % len(classes)   # How many will need an extra sample to distribute exactly

        selected_parts = []
        for i, c in enumerate(classes):
            # Distribute extra samples among the first 'remainder' classes
            quota = per_class + (1 if i < remainder else 0)
            idx = class_indices[c]
            # Take as many as possible, up to the quota, with no replacement
            n_take = min(quota, len(idx))
            selected_parts.append(rng.choice(idx, size=n_take, replace=False))

        # Combine all sampled indices
        selected = np.concatenate(selected_parts)
        # If not enough samples (some classes had too few members), fill randomly from remaining indices
        if len(selected) < context_size:
            remaining = np.setdiff1d(np.arange(self.n_instances), selected)
            n_need = context_size - len(selected)
            if len(remaining) > 0:
                n_take = min(n_need, len(remaining))
                selected = np.concatenate([selected, rng.choice(remaining, size=n_take, replace=False)])

        return selected

    def _get_context_indices(
        self, X_test: np.ndarray, context_size: int, seed: int | None = None
    ) -> np.ndarray:
        if self.context_reduction == "retrieval":
            return self._get_faiss_knn_indices(X_test, context_size=context_size, seed=seed)
        if self.context_reduction == "subsample":
            return self._get_subsample_indices(context_size, seed)
        if self.context_reduction == "subsample-balanced":
            return self._get_balanced_subsample_indices(context_size, seed)
        raise ValueError(f"Invalid context reduction mode: {self.context_reduction}")

    def _uses_stacked_context(self, context_size: float) -> bool:
        if context_size >= self.n_instances:
            return True
        return self.context_reduction in ("subsample", "subsample-balanced")

    def _prepare_stacked_context(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        context_size: float,
        seed: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context_size >= self.n_instances:
            ctx_x, ctx_y = train_x, train_y
        else:
            idx = self._get_context_indices(self.X_test, int(context_size), seed=seed)
            ctx_x = train_x[idx]
            ctx_y = train_y[idx]
        X_ctx = pad_x(ctx_x[None, :, :], self.max_features).to(self.device)
        y_ctx = ctx_y[None, :].float()
        return X_ctx, y_ctx

    def _get_ensemble_iterator(self, n_ensembles: int, seed: int | None = None):
        generator = np.random.SeedSequence(seed)
        ensemble_iterator = generator.generate_state(n_ensembles)
        if self.verbose:
            from tqdm import tqdm
            ensemble_iterator = tqdm(ensemble_iterator, desc="ensembles")
        return ensemble_iterator

    @dataclass
    class PreprocessResult:
        X_ctx: torch.Tensor
        y_ctx: torch.Tensor
        X_query: torch.Tensor
        feat_perm: torch.Tensor
        mean_y: torch.Tensor | None
        std_y: torch.Tensor | None

    def _preprocess(
        self,
        X: np.ndarray,
        context_size: int | None = None,
        seed: int | None = None,
        class_perm: np.ndarray | None = None,
        normalize_y_ctx: bool = False
    ) -> TabDPTEstimator.PreprocessResult:
        """
        Do all operations requred to go from a predict function input to tensors for the model
        """

        check_is_fitted(self)
        if context_size is None:
            context_size = np.inf

        if self.missing_indicators:
            inds = np.isnan(X)[:, self.has_missing_indicator].astype(float)
            X = np.hstack((X, inds))
        self.X_test = self.imputer.transform(X)
        if self.scaler:
            self.X_test = self.scaler.transform(self.X_test)
            if self.normalizer == 'quantile-uniform':
                self.X_test = 2*self.X_test - 1

        X_ctx, y_ctx, X_query = (
            convert_to_torch_tensor(self.X_train).to(self.device).float(),
            convert_to_torch_tensor(self.y_train).to(self.device).float(),
            convert_to_torch_tensor(self.X_test).to(self.device).float(),
        )

        # Apply PCA/subsampling to reduce the number of features if necessary
        feat_perm = None
        if self.n_features > self.max_features:
            if self.feature_reduction == "pca":
                X_ctx = X_ctx @ self.projection
                X_query = X_query @ self.projection
            elif self.feature_reduction == "subsample":
                feat_perm = generate_random_permutation(X_ctx.shape[1], seed)
                X_ctx = X_ctx[:, feat_perm][:, :self.max_features]
                X_query = X_query[:, feat_perm][:, :self.max_features]

        if class_perm is not None:
            assert self.mode == "cls", "class_perm only makes sense for classification"
            inv_perm = np.argsort(class_perm)
            y_ctx = y_ctx.to(torch.long)
            inv_perm = torch.as_tensor(inv_perm, device=y_ctx.device)
            y_ctx = inv_perm[y_ctx].to(torch.float)

        if feat_perm is None:
            if seed is not None:
                feat_perm = generate_random_permutation(X_ctx.shape[1], seed)
                X_ctx = X_ctx[:, feat_perm]
                X_query = X_query[:, feat_perm]
            else:
                feat_perm = torch.arange(X_ctx.shape[1])

        if normalize_y_ctx:
            y_ctx, mean_y, std_y = normalize_data(y_ctx, return_mean_std=True)
        else:
            mean_y, std_y = None, None

        if not self._uses_stacked_context(context_size):
            indices_nni = self._get_context_indices(
                self.X_test, context_size=context_size, seed=seed
            )
            X_nni = X_ctx[torch.tensor(indices_nni)]
            y_nni = y_ctx[torch.tensor(indices_nni)]
            X_nni = pad_x(X_nni, self.max_features)
            X_query = pad_x(X_query.unsqueeze(1), self.max_features).to(self.device)
            return self.PreprocessResult(X_nni, y_nni, X_query, feat_perm, mean_y, std_y)

        X_ctx, y_ctx = self._prepare_stacked_context(X_ctx, y_ctx, context_size, seed)
        X_query = pad_x(X_query[None, :, :], self.max_features).to(self.device)
        return self.PreprocessResult(X_ctx, y_ctx, X_query, feat_perm, mean_y, std_y)

    def _default_batch_size(self):
        return 4096 if self.device == "cpu" else 128*1024
