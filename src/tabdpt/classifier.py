import math
from typing import Literal

import numpy as np
import torch
from scipy.special import softmax
from sklearn.base import ClassifierMixin
from tqdm import tqdm

from .estimator import TabDPTEstimator
from .utils import generate_random_permutation, pad_x


class TabDPTClassifier(TabDPTEstimator, ClassifierMixin):
    def __init__(
        self,
        inf_batch_size: int = 512,
        normalizer: Literal["standard", "minmax", "robust", "power", "quantile-uniform", "quantile-normal", "log1p"] | None
            = "standard",
        missing_indicators: bool = False,
        clip_sigma: float = 4.,
        feature_reduction: Literal["pca", "subsample"] = "pca",
        faiss_metric: Literal["l2", "ip"] = "l2",
        device: str = None,
        use_flash: bool = True,
        compile: bool = True,
        model_weight_path: str | None = None,
    ):
        super().__init__(
            mode="cls",
            inf_batch_size=inf_batch_size,
            normalizer=normalizer,
            missing_indicators=missing_indicators,
            clip_sigma=clip_sigma,
            feature_reduction=feature_reduction,
            faiss_metric=faiss_metric,
            device=device,
            use_flash=use_flash,
            compile=compile,
            model_weight_path=model_weight_path,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)
        self.num_classes = len(np.unique(self.y_train))
        assert self.num_classes > 1, "Number of classes must be greater than 1"

    @torch.no_grad()
    def _predict_large_cls(self, X_train, X_test, y_train):
        num_digits = math.ceil(math.log(self.num_classes, self.max_num_classes))

        digit_preds = []
        for i in range(num_digits):
            y_train_digit = (y_train // (self.max_num_classes**i)) % self.max_num_classes
            pred = self._run_model(
                x_src=torch.cat([X_train, X_test], dim=0),
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
            full_pred[:, :, class_idx] = class_pred

        return full_pred

    @torch.no_grad()
    def _predict_proba_batched(
        self,
        X: np.ndarray,
        n_ensembles: int,
        temperature: float,
        context_size: int,
        permute_classes: bool,
        seed: int | None,
        return_logits: bool,
    ):
        # Only used when context_size >= n_instances (no retrieval).
        train_x, train_y, test_x = self._prepare_prediction(X, class_perm=None, seed=None)

        seeds = np.random.SeedSequence(seed).generate_state(n_ensembles)
        train_x_list, test_x_list, y_train_list, inv_perms = [], [], [], []

        for s in seeds:
            s = int(s)
            feat_perm = generate_random_permutation(train_x.shape[1], s)
            tx = train_x[:, feat_perm]
            te = test_x[:, feat_perm]
            ty = train_y.clone()

            inv_perm = None
            if permute_classes:
                perm = generate_random_permutation(self.num_classes, s)
                inv_perm = torch.as_tensor(np.argsort(perm), device=ty.device)
                ty = inv_perm[ty.long()].float()  # remap labels into permuted space

            train_x_list.append(tx)
            test_x_list.append(te)
            y_train_list.append(ty)
            inv_perms.append(inv_perm)

        X_train = pad_x(torch.stack(train_x_list, dim=1), self.max_features).to(self.device)
        X_test = pad_x(torch.stack(test_x_list, dim=1), self.max_features).to(self.device)
        y_train = torch.stack(y_train_list, dim=1).float()

        pred = self.model(
            x_src=torch.cat([X_train, X_test], dim=0),
            y_src=y_train.unsqueeze(-1),
            task=self.mode,
        )  # (T_test, B, num_classes)

        if permute_classes:
            reordered = []
            for b, inv_perm in enumerate(inv_perms):
                if inv_perm is None:
                    reordered.append(pred[:, b, :])
                else:
                    reordered.append(pred[:, b, :][:, inv_perm])
            pred = torch.stack(reordered, dim=1)

        pred = pred.float()
        logit_mean = pred.mean(dim=1)  # average across ensemble dimension

        if return_logits:
            return logit_mean.detach().cpu().numpy()

        logit_mean = logit_mean / temperature
        prob = torch.nn.functional.softmax(logit_mean, dim=-1)
        prob = prob.detach().cpu().numpy()
        return prob

    def predict_proba(
        self,
        X: np.ndarray,
        temperature: float = 0.8,
        context_size: int = 2048,
        return_logits: bool = False,
        seed: int | None = None,
        class_perm: np.ndarray | None = None,
    ):
        train_x, train_y, test_x = self._prepare_prediction(X, class_perm=class_perm, seed=seed)

        if seed is not None:
            feat_perm = generate_random_permutation(train_x.shape[1], seed)
            train_x = train_x[:, feat_perm]
            test_x = test_x[:, feat_perm]

        if context_size >= self.n_instances:
            X_train = pad_x(train_x, self.max_features).to(self.device).unsqueeze(1)  # (T_train, 1, F)
            X_test = pad_x(test_x, self.max_features).to(self.device).unsqueeze(1)    # (T_test, 1, F)
            y_train = train_y.unsqueeze(1).float()                                   # (T_train, 1)

            if self.num_classes <= self.max_num_classes:
                pred = self._run_model(
                    x_src=torch.cat([X_train, X_test], dim=0),
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

                if self.faiss_knn is None:
                    # fallback: use all train points if index is absent
                    indices_nni = [list(range(self.n_instances))] * (end - start)
                else:
                    indices_nni = self.faiss_knn.get_knn_indices(self.X_test[start:end], k=context_size)
                idx = torch.as_tensor(indices_nni, device=train_x.device)
                X_nni = train_x[idx]  # (B, ctx, F)
                y_nni = train_y[idx]  # (B, ctx)

                X_nni, y_nni = (
                    pad_x(X_nni, self.max_features).to(self.device).permute(1, 0, 2),  # (ctx, B, F)
                    y_nni.to(self.device).permute(1, 0),                               # (ctx, B)
                )
                X_eval = test_x[start:end]
                X_eval = pad_x(X_eval.unsqueeze(1), self.max_features).to(self.device).permute(1, 0, 2)  # (1, B, F)

                if self.num_classes <= self.max_num_classes:
                    pred = self._run_model(
                        x_src=torch.cat([X_nni, X_eval], dim=0),
                        y_src=y_nni.unsqueeze(-1),
                        task=self.mode,
                    )
                else:
                    pred = self._predict_large_cls(X_nni, X_eval, y_nni)

                pred = pred.float()
                if not return_logits:
                    pred = pred[..., :self.num_classes] / temperature
                    pred = torch.nn.functional.softmax(pred, dim=-1)
                    pred /= pred.sum(axis=-1, keepdims=True)  # numerical stability

                pred_list.append(pred.squeeze(dim=0))
            pred_val = torch.cat(pred_list, dim=0).squeeze().detach().cpu().float().numpy()

        return pred_val

    @torch.no_grad()
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

    @torch.no_grad()
    def predict(
        self,
        X,
        n_ensembles: int = 8,
        temperature: float = 0.8,
        context_size: int = 2048,
        permute_classes: bool = True,
        seed: int | None = None,
    ):
        if n_ensembles == 1:
            return self.predict_proba(X, temperature=temperature, context_size=context_size, seed=seed).argmax(axis=-1)
        else:
            return self.ensemble_predict_proba(
                X,
                n_ensembles=n_ensembles,
                temperature=temperature,
                context_size=context_size,
                permute_classes=permute_classes,
                seed=seed,
            ).argmax(axis=-1)
