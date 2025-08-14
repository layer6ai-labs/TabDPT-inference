import json
# from enum import StrEnum, auto

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .model import TabDPTModel
from .utils import FAISS, convert_to_torch_tensor, generate_random_permutation

# Constants for model caching and download
_VERSION = "1_1"
_MODEL_NAME = f"tabdpt{_VERSION}.safetensors"
CPU_INF_BATCH = 16


# class LongSchemaStrategy(StrEnum):
#     PCA = auto()
#     SUBSAMPLE = auto()


class TabDPTEstimator(BaseEstimator):
    def __init__(
        self,
        mode: str,
        inf_batch_size: int = 512,
        device: str = None,
        use_flash: bool = True,
        compile: bool = True,
        # long_schema_strategy: LongSchemaStrategy = LongSchemaStrategy.PCA,
    ):
        self.mode = mode
        # self.long_schema_strategy = long_schema_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inf_batch_size = (
            inf_batch_size if self.device == "cuda" else min(inf_batch_size, CPU_INF_BATCH)
        )
        self.use_flash = use_flash and self.device == "cuda"

        self.path = hf_hub_download(
            repo_id="Layer6/TabDPT",
            filename=_MODEL_NAME,
        )

        with safe_open(self.path, framework="pt", device=self.device) as f:
            meta = f.metadata()
            cfg_dict = json.loads(meta["cfg"])
            cfg = OmegaConf.create(cfg_dict)
            model_state = {k: f.get_tensor(k) for k in f.keys()}

        cfg.env.device = self.device
        self.model = TabDPTModel.load(model_state=model_state, config=cfg, use_flash=self.use_flash)
        self.model.eval()

        self.max_features = self.model.num_features
        self.max_num_classes = self.model.n_out
        self.compile = compile and self.device == "cuda"
        assert self.mode in ["cls", "reg"], "mode must be 'cls' or 'reg'"

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"

        self.imputer = SimpleImputer(strategy="mean")
        X = self.imputer.fit_transform(X)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.faiss_knn = FAISS(X)
        self.n_instances, self.n_features = X.shape
        self.X_train = X
        self.y_train = y
        if (
            self.n_features > self.max_features
            # and self.long_schema_strategy == LongSchemaStrategy.PCA
        ):
            train_x = convert_to_torch_tensor(self.X_train).to(self.device).float()
            _, _, self.V = torch.pca_lowrank(train_x, q=min(train_x.shape[0], self.max_features))

        self.is_fitted_ = True
        if self.compile:
            self.model = torch.compile(self.model)

    def _prepare_prediction(
        self, X: np.ndarray, seed: int | None = None, class_perm: np.ndarray | None = None
    ):
        check_is_fitted(self)
        self.X_test = self.imputer.transform(X)
        self.X_test = self.scaler.transform(self.X_test)
        train_x, train_y, test_x = (
            convert_to_torch_tensor(self.X_train).to(self.device).float(),
            convert_to_torch_tensor(self.y_train).to(self.device).float(),
            convert_to_torch_tensor(self.X_test).to(self.device).float(),
        )

        # Apply PCA or subsampling optionally to reduce the number of features
        if self.n_features > self.max_features:
            # if self.long_schema_strategy == LongSchemaStrategy.PCA:
            train_x = train_x @ self.V
            test_x = test_x @ self.V
            # elif self.long_schema_strategy == LongSchemaStrategy.SUBSAMPLE:
            #     feat_perm = generate_random_permutation(train_x.shape[1], seed)
            #     train_x = train_x[:, feat_perm][:, : self.model.encoder.num_features]
            #     test_x = test_x[:, feat_perm][:, : self.model.encoder.num_features]
            # else:
            #     raise Exception("Undefined method for handling long context.")

        if class_perm is not None:
            inv_perm = np.argsort(class_perm)
            train_y = train_y.to(torch.long)
            inv_perm = torch.as_tensor(inv_perm, device=train_y.device)
            train_y = inv_perm[train_y].to(torch.float)

        return train_x, train_y, test_x
