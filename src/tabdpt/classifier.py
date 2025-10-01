import math

import numpy as np
import torch
from scipy.special import softmax
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .estimator import TabDPTEstimator
from .utils import generate_random_permutation, pad_x


class TabDPTClassifier(TabDPTEstimator, ClassifierMixin):
    def __init__(
        self,
        inf_batch_size: int = 512,
        device: str = None,
        use_flash: bool = True,
        compile: bool = True,
        model_weight_path: str | None = None,
    ):
        super().__init__(
            mode="cls",
            inf_batch_size=inf_batch_size,
            device=device,
            use_flash=use_flash,
            compile=compile,
            model_weight_path=model_weight_path,
        )

    @torch.no_grad()
    def get_optimum_temperature_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        temp_grid: list | None = None,
        autotune_folds: int = 5,
        max_autotune_context: int = 2048,
    ):
        default_grid = [x / 10 for x in range(1, 13)]
        if temp_grid is None:
            temp_grid = default_grid
        skf = StratifiedKFold(n_splits=autotune_folds, shuffle=True, random_state=42)
        all_logits = []
        all_targets = []

        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]
            super().fit(X_tr, y_tr)
            self.num_classes = len(np.unique(self.y_train))
            ctx = min(max_autotune_context, len(X_tr))
            logits = self.predict_proba(X_va, return_logits=True, context_size=ctx)
            logits = np.asarray(logits)
            all_logits.append(logits)
            all_targets.append(y_va)

        logits_cv = np.concatenate(all_logits, axis=0)
        y_cv = np.concatenate(all_targets, axis=0)
        C = logits_cv.shape[-1]
        losses = []
        for t in temp_grid:
            probs = softmax(logits_cv / float(t), axis=-1)
            losses.append(log_loss(y_cv, probs, labels=np.arange(C)))

        return temp_grid[int(np.argmin(losses))]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        super().fit(X, y)
        self.num_classes = len(np.unique(self.y_train))
        assert self.num_classes > 1, "Number of classes must be greater than 1"

    def _predict_large_cls(self, X_train, X_test, y_train):
        num_digits = math.ceil(math.log(self.num_classes, self.max_num_classes))

        digit_preds = []
        for i in range(num_digits):
            y_train_digit = (y_train // (self.max_num_classes**i)) % self.max_num_classes
            pred = self.model(
                x_src=torch.cat([X_train, X_test], dim=1),
                y_src=y_train_digit.unsqueeze(-1),
                task=self.mode,
            )
            digit_preds.append(pred.float())

        full_pred = torch.zeros((X_test.shape[0], X_test.shape[1], self.num_classes), device=X_train.device)
        for class_idx in range(self.num_classes):
            class_pred = torch.zeros_like(digit_preds[0][:, :, 0])
            for digit_idx, digit_pred in enumerate(digit_preds):
                digit_value = (class_idx // (self.max_num_classes**digit_idx)) % self.max_num_classes
                class_pred += digit_pred[:, :, digit_value]
            full_pred[:, :, class_idx] = class_pred.transpose(0, 1)

        return full_pred

    @torch.no_grad()
    def predict_proba(
        self,
        X: np.ndarray,
        temperature: float = 0.8,
        context_size: int = 2048,
        return_logits: bool = False,
        seed: int | None = None,
        class_perm: np.ndarray | None = None,
    ):
        train_x, train_y, test_x = self._prepare_prediction(X, class_perm=class_perm)

        if seed is not None:
            self.faiss_knn.index.seed = seed
            feat_perm = generate_random_permutation(train_x.shape[1], seed)
            train_x = train_x[:, feat_perm]
            test_x = test_x[:, feat_perm]

        if context_size >= self.n_instances:
            X_train = pad_x(train_x[None, :, :], self.max_features).to(self.device)
            X_test = pad_x(test_x[None, :, :], self.max_features).to(self.device)
            y_train = train_y[None, :].float()

            if self.num_classes <= self.max_num_classes:
                pred = self.model(
                    x_src=torch.cat([X_train, X_test], dim=1),
                    y_src=y_train.unsqueeze(-1),
                    task=self.mode,
                )
            else:
                pred = self._predict_large_cls(X_train, X_test, y_train)

            if not return_logits:
                pred = pred[..., :self.num_classes] / temperature
                pred = torch.nn.functional.softmax(pred.float(), dim=-1)
            pred_val = pred.float().squeeze().detach().cpu().numpy()
        else:
            pred_list = []
            for b in range(math.ceil(len(self.X_test) / self.inf_batch_size)):
                start = b * self.inf_batch_size
                end = min(len(self.X_test), (b + 1) * self.inf_batch_size)

                indices_nni = self.faiss_knn.get_knn_indices(self.X_test[start:end], k=context_size)
                X_nni = train_x[torch.tensor(indices_nni)]
                y_nni = train_y[torch.tensor(indices_nni)]

                X_nni, y_nni = (
                    pad_x(torch.Tensor(X_nni), self.max_features).to(self.device),
                    torch.Tensor(y_nni).to(self.device),
                )
                X_eval = test_x[start:end]
                X_eval = pad_x(X_eval.unsqueeze(1), self.max_features).to(self.device)

                if self.num_classes <= self.max_num_classes:
                    pred = self.model(
                        x_src=torch.cat([X_nni, X_eval], dim=1),
                        y_src=y_nni.unsqueeze(-1),
                        task=self.mode,
                    )
                else:
                    pred = self._predict_large_cls(X_nni, X_eval, y_nni)

                pred = pred.float()
                if not return_logits:
                    pred = pred[..., :self.num_classes] / temperature
                    pred = torch.nn.functional.softmax(pred, dim=-1)

                pred_list.append(pred.squeeze(dim=0))
            pred_val = torch.cat(pred_list, dim=0).squeeze().detach().cpu().float().numpy()

        return pred_val

    def ensemble_predict_proba(
        self,
        X,
        n_ensembles: int = 8,
        temperature: float = 0.8,
        context_size: int = 2048,
        permute_classes: bool = True,
        seed: int | None = None,
    ):
        root_ss = np.random.SeedSequence(seed)
        inner_seeds = root_ss.generate_state(n_ensembles)
        logit_cumsum = None

        for inner_seed in tqdm(inner_seeds, desc="ensembles"):
            inner_seed = int(inner_seed)
            perm = torch.arange(self.num_classes)
            if permute_classes:
                perm = generate_random_permutation(self.num_classes, inner_seed)
            inv_perm = np.argsort(perm)

            logits = self.predict_proba(
                X,
                context_size=context_size,
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
        temperature: float = 0.8,
        context_size: int = 2048,
        permute_classes: bool = True,
        return_probs: bool = False,
        seed: int | None = None,
    ):
        if n_ensembles == 1:
            out = self.predict_proba(X, temperature=temperature, context_size=context_size, seed=seed)
        else:
            out = self.ensemble_predict_proba(
                X,
                n_ensembles=n_ensembles,
                temperature=temperature,
                context_size=context_size,
                permute_classes=permute_classes,
                seed=seed,
            )
        if not return_probs:
            out = out.argmax(axis=-1)
        return out
