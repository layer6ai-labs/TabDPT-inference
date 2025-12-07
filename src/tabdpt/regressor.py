import math
from typing import Literal

import numpy as np
import torch
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .estimator import TabDPTEstimator
from .utils import generate_random_permutation, pad_x, predict_regression_value


class TabDPTRegressor(TabDPTEstimator, RegressorMixin):
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
        beta: float = 0.2,
    ):
        super().__init__(
            mode="reg",
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
        self.beta = beta

    @torch.no_grad()
    def _predict(
        self,
        X: np.ndarray,
        context_size: int = 2048,
        seed: int | None = None,
        beta: float | None = None,
        return_logits: bool = False,
    ):
        beta = self.beta if beta is None else beta
        train_x, train_y, test_x = self._prepare_prediction(X, seed=seed)

        if seed is not None:
            feat_perm = generate_random_permutation(train_x.shape[1], seed)
            train_x = train_x[:, feat_perm]
            test_x = test_x[:, feat_perm]

        if context_size >= self.n_instances:
            X_train = pad_x(train_x, self.max_features).to(self.device).unsqueeze(1)  # (T_train, 1, F)
            X_test = pad_x(test_x, self.max_features).to(self.device).unsqueeze(1)    # (T_test, 1, F)
            y_train = train_y.unsqueeze(1).float()                                   # (T_train, 1)
            pred = self._run_model(
                x_src=torch.cat([X_train, X_test], dim=0),
                y_src=y_train.unsqueeze(-1),
                task=self.mode,
            )
            test_logits = pred[:, 0, :].float()  # (T_test, nbins)
            if return_logits:
                return {"logits": test_logits.detach().cpu(), "mean": None, "std": None}
            test_preds = predict_regression_value(test_logits, beta=beta)  # (T_test,)
            return test_preds.detach().cpu().float().numpy()
        else:
            pred_values = []
            logits_batches = []
            mean_batches = []
            std_batches = []
            for b in range(math.ceil(len(self.X_test) / self.inf_batch_size)):
                start = b * self.inf_batch_size
                end = min(len(self.X_test), (b + 1) * self.inf_batch_size)

                if self.faiss_knn is None:
                    indices_nni = [list(range(self.n_instances))] * (end - start)
                else:
                    indices_nni = self.faiss_knn.get_knn_indices(self.X_test[start:end], k=context_size)
                idx = torch.as_tensor(indices_nni, device=train_x.device)
                X_nni = train_x[idx]  # (B, ctx, F)
                y_nni = train_y[idx]  # (B, ctx)

                # Scale y_nni per batch element (B, ctx)
                mean = y_nni.mean(dim=1, keepdim=True)
                std = y_nni.std(dim=1, keepdim=True) + 1e-6
                y_nni_scaled = (y_nni - mean) / std 
                y_nni_scaled = torch.clamp(y_nni_scaled, -10.0, 10.0)

                X_nni, y_nni = (
                    pad_x(X_nni, self.max_features).to(self.device).permute(1, 0, 2),  # (ctx, B, F)
                    y_nni_scaled.to(self.device).permute(1, 0),                        # (ctx, B)
                )
                X_eval = test_x[start:end]
                X_eval = pad_x(X_eval.unsqueeze(1), self.max_features).to(self.device).permute(1, 0, 2)  # (1, B, F)
                pred = self._run_model(
                    x_src=torch.cat([X_nni, X_eval], dim=0),
                    y_src=y_nni.unsqueeze(-1),
                    task=self.mode,
                )
                test_logits = pred.squeeze(0).float()  # (B, nbins)
                if return_logits:
                    logits_batches.append(test_logits.detach().cpu())
                    mean_batches.append(mean.detach().cpu())
                    std_batches.append(std.detach().cpu())
                else:
                    test_vals = predict_regression_value(test_logits, beta=beta)  # (B,)
                    # unscale per batch
                    test_vals = test_vals * std.squeeze(1) + mean.squeeze(1)
                    pred_values.append(test_vals.detach().cpu())

            if return_logits:
                logits = torch.cat(logits_batches)
                mean_all = torch.cat(mean_batches).squeeze(1)
                std_all = torch.cat(std_batches).squeeze(1)
                return {"logits": logits, "mean": mean_all, "std": std_all}

            return torch.cat(pred_values).float().numpy()

    @torch.no_grad()
    def _ensemble_predict(
        self,
        X: np.ndarray,
        n_ensembles: int = 8,
        context_size: int = 2048,
        seed: int | None = None,
        beta: float | None = None,
        avg_logits: bool = False,
    ):
        if avg_logits:
            logits_cumsum = None
            base_mean = None
            base_std = None
        else:
            prediction_cumsum = 0

        generator = np.random.SeedSequence(seed)
        for _, inner_seed in tqdm(zip(range(n_ensembles), generator.generate_state(n_ensembles))):
            inner_seed = int(inner_seed)
            if avg_logits:
                outputs = self._predict(
                    X, context_size=context_size, seed=inner_seed, beta=beta, return_logits=True
                )
                logits = outputs["logits"]
                if logits_cumsum is None:
                    logits_cumsum = logits
                    base_mean = outputs["mean"]
                    base_std = outputs["std"]
                else:
                    logits_cumsum += logits
            else:
                prediction_cumsum += self._predict(X, context_size=context_size, seed=inner_seed, beta=beta)

        if avg_logits:
            avg_logits_tensor = logits_cumsum / n_ensembles
            preds_scaled = predict_regression_value(avg_logits_tensor, beta=self.beta if beta is None else beta)
            if base_mean is not None and base_std is not None:
                preds = preds_scaled * base_std + base_mean
            else:
                preds = preds_scaled
            return preds.detach().cpu().float().numpy()
        else:
            return prediction_cumsum / n_ensembles

    def predict(
        self,
        X: np.ndarray,
        n_ensembles: int = 8,
        context_size: int = 2048,
        seed: int | None = None,
        beta: float | None = None,
        avg_logits: bool = False,
    ):
        if n_ensembles == 1:
            return self._predict(X, context_size=context_size, seed=seed, beta=beta)
        else:
            return self._ensemble_predict(
                X,
                n_ensembles=n_ensembles,
                context_size=context_size,
                seed=seed,
                beta=beta,
                avg_logits=avg_logits,
            )

    # @torch.no_grad()
    # def predict_sorted_refine(
    #     self,
    #     X: np.ndarray,
    #     context_size: int = 2048,
    #     batch_size: int = 512,
    #     seed: int | None = None,
    #     beta: float | None = None,
    # ):
    #     """
    #     Refines predictions by aligning sorted test batches with sorted training windows.
        
    #     1. Runs a coarse prediction.
    #     2. Sorts X_test by coarse predictions.
    #     3. Sorts X_train by true y_train.
    #     4. For each batch of sorted test data, selects the window of X_train 
    #        centered around the batch's mean predicted value.
    #     5. Locally normalizes the window to maximize model resolution.
    #     """
    #     beta = self.beta if beta is None else beta
    #     device = self.device
        
    #     # --- 1. Coarse Step ---
    #     # We use a large context (or full context) to get the initial "location"
    #     # If dataset is huge, you might want to limit context_size here for speed
    #     print("Running coarse estimation...")
    #     coarse_preds = self._predict(X, context_size=context_size, seed=seed, beta=beta)

    #     # --- 2. Prepare & Sort Data ---
    #     train_x, train_y, test_x = self._prepare_prediction(X, seed=seed)
        
    #     # Move to CPU numpy for sorting logic to save VRAM
    #     y_train_np = train_y.cpu().numpy()
    #     x_train_np = train_x.cpu().numpy()
    #     x_test_np = test_x.cpu().numpy()
        
    #     # Sort Training Data by Target Y
    #     train_sort_idx = np.argsort(y_train_np)
    #     x_train_sorted = x_train_np[train_sort_idx]
    #     y_train_sorted = y_train_np[train_sort_idx]
        
    #     # Sort Test Data by Coarse Prediction
    #     test_sort_idx = np.argsort(coarse_preds)
    #     x_test_sorted = x_test_np[test_sort_idx]
    #     coarse_sorted = coarse_preds[test_sort_idx] # We need this to find insertion points

    #     # We will store results here and unsort them at the end
    #     refined_preds_sorted = []
        
    #     # --- 3. Sliding Window Inference ---
    #     n_test = len(x_test_np)
    #     n_train = len(y_train_np)
        
    #     # Ensure context size isn't larger than available training data
    #     real_ctx_size = min(n_train, context_size)
    #     half_ctx = real_ctx_size // 2

    #     print(f"Refining with sliding window (Context: {real_ctx_size})...")
        
    #     for b_start in tqdm(range(0, n_test, batch_size)):
    #         b_end = min(b_start + batch_size, n_test)
            
    #         # Identify the "Center of Mass" for this batch
    #         # We look at the median coarse prediction for this batch
    #         batch_coarse_vals = coarse_sorted[b_start:b_end]
    #         batch_median = np.median(batch_coarse_vals)
            
    #         # Find where this median fits in the sorted training distribution
    #         center_idx = np.searchsorted(y_train_sorted, batch_median)
            
    #         # Determine Window Indices (Clamp to array bounds)
    #         # We try to center the window around the insertion index
    #         w_start = center_idx - half_ctx
    #         w_end = center_idx + half_ctx
            
    #         # Adjust if window falls out of bounds (shift the window, don't shrink it)
    #         if w_start < 0:
    #             w_end -= w_start # add the deficit to the end
    #             w_start = 0
    #         if w_end > n_train:
    #             w_start -= (w_end - n_train) # subtract overflow from start
    #             w_end = n_train
                
    #         # Final clamp (in case n_train < context_size)
    #         w_start = max(0, w_start)
    #         w_end = min(n_train, w_end)
            
    #         # --- Extract Context ---
    #         # Now we have a dense cluster of Y values similar to our test batch predictions
    #         ctx_x_batch = x_train_sorted[w_start:w_end]
    #         ctx_y_batch = y_train_sorted[w_start:w_end]
            
    #         # --- Local Normalization (Crucial Step) ---
    #         # We normalize this specific window to be N(0,1)
    #         # This allows the model to differentiate small differences within this cluster
    #         mu_local = ctx_y_batch.mean()
    #         std_local = ctx_y_batch.std() + 1e-6
            
    #         # Scale and Clip (as per training requirements)
    #         ctx_y_scaled = (ctx_y_batch - mu_local) / std_local
    #         ctx_y_scaled = np.clip(ctx_y_scaled, -10.0, 10.0)
            
    #         # Prepare Tensors
    #         X_ctx_t = pad_x(torch.from_numpy(ctx_x_batch).float().unsqueeze(1).to(device), self.max_features)
    #         y_ctx_t = torch.from_numpy(ctx_y_scaled).float().unsqueeze(1).to(device)
            
    #         # Prepare Test Batch
    #         X_test_batch = x_test_sorted[b_start:b_end]
    #         X_test_t = pad_x(torch.from_numpy(X_test_batch).float().unsqueeze(1).to(device), self.max_features)
            
    #         # --- Inference ---
    #         # Shape: X_src = (n_ctx + n_test_batch, 1, F)
    #         pred = self._run_model(
    #             x_src=torch.cat([X_ctx_t, X_test_t], dim=0),
    #             y_src=y_ctx_t.unsqueeze(-1),
    #             task=self.mode,
    #         )
            
    #         # Extract logits for test part only
    #         # The test samples are at the end of the sequence
    #         test_logits = pred[-len(X_test_batch):, 0, :].float()
            
    #         # Predict values in the SCALED space
    #         batch_preds_scaled = predict_regression_value(test_logits, beta=beta)
    #         batch_preds_scaled = batch_preds_scaled.detach().cpu().numpy()
            
    #         # --- Denormalize ---
    #         # Map back to real space using the local window stats
    #         batch_preds = batch_preds_scaled * std_local + mu_local
    #         refined_preds_sorted.append(batch_preds)

    #     # --- 4. Reconstruct Order ---
    #     all_refined_sorted = np.concatenate(refined_preds_sorted)
        
    #     # We must unsort to match the original X order
    #     # We create an empty array and scatter the results back using the sort indices
    #     final_predictions = np.zeros_like(all_refined_sorted)
    #     final_predictions[test_sort_idx] = all_refined_sorted
        
    #     return final_predictions

    @torch.no_grad()
    def _predict_sorted_refine_single(
        self,
        X: np.ndarray,
        context_size: int = 2048,
        batch_size: int = 512,
        seed: int | None = None,
        beta: float | None = None,
        disable_tqdm: bool = False,
    ):
        """
        Internal method: Runs a single pass of the Sort & Refine strategy.
        """
        beta = self.beta if beta is None else beta
        device = self.device
        
        # --- 1. Coarse Step ---
        # Get a rough estimate to organize the data
        coarse_preds = self._predict(X, context_size=context_size, seed=seed, beta=beta)

        # --- 2. Prepare & Sort Data ---
        train_x, train_y, test_x = self._prepare_prediction(X, seed=seed)
        
        # Feature permutation is handled inside _prepare_prediction via seed,
        # but if you need explicit control here:
        if seed is not None:
            feat_perm = generate_random_permutation(train_x.shape[1], seed)
            train_x = train_x[:, feat_perm]
            test_x = test_x[:, feat_perm]
        
        # Move to CPU numpy for sorting logic
        y_train_np = train_y.cpu().numpy()
        x_train_np = train_x.cpu().numpy()
        x_test_np = test_x.cpu().numpy()
        
        # Sort Training Data by Target Y
        train_sort_idx = np.argsort(y_train_np)
        x_train_sorted = x_train_np[train_sort_idx]
        y_train_sorted = y_train_np[train_sort_idx]
        
        # Sort Test Data by Coarse Prediction
        test_sort_idx = np.argsort(coarse_preds)
        x_test_sorted = x_test_np[test_sort_idx]
        coarse_sorted = coarse_preds[test_sort_idx]

        refined_preds_sorted = []
        
        # --- 3. Sliding Window Inference ---
        n_test = len(x_test_np)
        n_train = len(y_train_np)
        real_ctx_size = min(n_train, context_size)
        half_ctx = real_ctx_size // 2
        
        # Only show tqdm if this is not a sub-process of a larger ensemble loop
        # or if the user explicitly wants it
        iterator = range(0, n_test, batch_size)
        if not disable_tqdm:
            iterator = tqdm(iterator, desc="Refining batches")

        for b_start in iterator:
            b_end = min(b_start + batch_size, n_test)
            
            # Identify the median "location" for this batch
            batch_coarse_vals = coarse_sorted[b_start:b_end]
            batch_median = np.median(batch_coarse_vals)
            
            # Find insertion point in sorted training data
            center_idx = np.searchsorted(y_train_sorted, batch_median)
            
            # Determine Window Indices
            w_start = center_idx - half_ctx
            w_end = center_idx + half_ctx
            
            # Boundary checks (shift window, don't shrink)
            if w_start < 0:
                w_end -= w_start
                w_start = 0
            if w_end > n_train:
                w_start -= (w_end - n_train)
                w_end = n_train
            w_start = max(0, w_start)
            w_end = min(n_train, w_end)
            
            # --- Extract Context ---
            ctx_x_batch = x_train_sorted[w_start:w_end]
            ctx_y_batch = y_train_sorted[w_start:w_end]
            
            # --- Local Normalization ---
            mu_local = ctx_y_batch.mean()
            std_local = ctx_y_batch.std() + 1e-6
            
            ctx_y_scaled = (ctx_y_batch - mu_local) / std_local
            ctx_y_scaled = np.clip(ctx_y_scaled, -10.0, 10.0)
            
            # To Tensors
            X_ctx_t = pad_x(torch.from_numpy(ctx_x_batch).float().unsqueeze(1).to(device), self.max_features)
            y_ctx_t = torch.from_numpy(ctx_y_scaled).float().unsqueeze(1).to(device)
            
            X_test_batch = x_test_sorted[b_start:b_end]
            X_test_t = pad_x(torch.from_numpy(X_test_batch).float().unsqueeze(1).to(device), self.max_features)
            
            # Inference
            pred = self._run_model(
                x_src=torch.cat([X_ctx_t, X_test_t], dim=0),
                y_src=y_ctx_t.unsqueeze(-1),
                task=self.mode,
            )
            
            test_logits = pred[-len(X_test_batch):, 0, :].float()
            
            # Predict & Denormalize
            batch_preds_scaled = predict_regression_value(test_logits, beta=beta).detach().cpu().numpy()
            batch_preds = batch_preds_scaled * std_local + mu_local
            refined_preds_sorted.append(batch_preds)

        # --- 4. Reconstruct Order ---
        all_refined_sorted = np.concatenate(refined_preds_sorted)
        final_predictions = np.zeros_like(all_refined_sorted)
        # Unsort to match original X order
        final_predictions[test_sort_idx] = all_refined_sorted
        
        return final_predictions

    def predict_sorted_refine(
        self,
        X: np.ndarray,
        n_ensembles: int = 1,
        context_size: int = 2048,
        batch_size: int = 512,
        beta: float = 0.25,
        seed: int | None = None,
    ):
        """
        Refines predictions using Dynamic Sorted Contexts.
        Supports ensembling: effectively averages over different feature permutations 
        and different 'coarse' starting points.
        """
        if n_ensembles <= 1:
            return self._predict_sorted_refine_single(
                X, context_size=context_size, batch_size=batch_size, seed=seed, beta=beta
            )
        
        prediction_cumsum = 0
        generator = np.random.SeedSequence(seed)
        
        print(f"Running Refined Ensemble ({n_ensembles} estimators)...")
        # We iterate through seeds
        for i, inner_seed in tqdm(enumerate(generator.generate_state(n_ensembles)), total=n_ensembles):
            inner_seed = int(inner_seed)
            # We disable the inner tqdm to prevent log spamming, only showing the ensemble progress
            pred = self._predict_sorted_refine_single(
                X, 
                context_size=context_size, 
                batch_size=batch_size, 
                seed=inner_seed, 
                beta=beta, 
                disable_tqdm=True
            )
            prediction_cumsum += pred
            
        return prediction_cumsum / n_ensembles
