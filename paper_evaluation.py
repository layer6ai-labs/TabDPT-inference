import argparse
import hashlib
import itertools
import os
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import accuracy_score, f1_score, log_loss, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from tqdm import tqdm

from tabdpt import TabDPTClassifier, TabDPTRegressor
from tabdpt_datasets.openml import OpenMLDataset, TabZillaDataset

CLS_DATASET_PATH = "tabdpt_datasets/data_splits/cls_datasets.csv"
REG_DATASET_PATH = "tabdpt_datasets/data_splits/reg_datasets.csv"
FEATURE_PRUNE_THRESHOLD = 100

def prune_features_if_needed(X_train: np.ndarray, X_test: np.ndarray):
    if X_train.shape[1] <= FEATURE_PRUNE_THRESHOLD:
        return X_train, X_test, 0, 0

    const_mask = np.isclose(np.nanstd(X_train, axis=0), 0)
    removed_constant = int(const_mask.sum())
    if removed_constant:
        keep_mask = ~const_mask
        X_train = X_train[:, keep_mask]
        X_test = X_test[:, keep_mask]

    duplicate_mask = np.zeros(X_train.shape[1], dtype=bool)
    seen = {}
    for idx in range(X_train.shape[1]):
        col = np.ascontiguousarray(X_train[:, idx])
        key = hashlib.sha1(col.tobytes()).hexdigest()
        if key in seen:
            ref_idx = seen[key]
            if np.array_equal(col, X_train[:, ref_idx], equal_nan=True):
                duplicate_mask[idx] = True
        else:
            seen[key] = idx

    removed_duplicate = int(duplicate_mask.sum())
    if removed_duplicate:
        keep_mask = ~duplicate_mask
        X_train = X_train[:, keep_mask]
        X_test = X_test[:, keep_mask]

    return X_train, X_test, removed_constant, removed_duplicate


def main():
    parser = argparse.ArgumentParser(description="Run TabDPT evaluation")
    parser.add_argument("--context_size", type=int, default=100000, help="Context size for the model")
    parser.add_argument("--fold", type=int, default=0, help="Fold number to use for evaluation")
    parser.add_argument("--n-ensembles", type=int, default=1, help="Number of ensembles to use for evaluation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for classification")
    parser.add_argument("--beta", type=float, default=0.225, help="Softmax temperature for regression bin decoding")
    parser.add_argument("--seed", type=int, default=0, help="Model evaluation seed")
    parser.add_argument("--inf-batch-size", type=int, default=100000, help="Batch size for inference")
    parser.add_argument("--use-cpu", action="store_true", help="If true, use CPU for evalutation")
    parser.add_argument("--gpu-to-use", type=int, default=0, help="Which GPU to use")
    parser.add_argument("--results-folder", type=str, default="eval_precision", help="Parent results directory")
    parser.add_argument("--precision-tag", type=str, default="bf16", help="Tag to identify precision mode in filenames")
    parser.add_argument(
        "--avg-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, average regression logits across ensembles before softmax decoding (default: True).",
    )
    args = parser.parse_args()

    if args.use_cpu:
        device = "cpu"
    else:
        device = "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_to_use)

    df_eval_cls = pd.read_csv(CLS_DATASET_PATH)
    cc18_test_df = df_eval_cls[df_eval_cls["test"] == True]
    cc18_dids = cc18_test_df["did"]

    df_eval_reg = pd.read_csv(REG_DATASET_PATH)
    ctr23_test_df = df_eval_reg[df_eval_reg["test"] == True]
    ctr23_dids = ctr23_test_df["did"]

    did_tid_mapping = dict(zip(cc18_dids, cc18_test_df["tid"]))
    did_tid_mapping.update(dict(zip(ctr23_dids, ctr23_test_df["tid"])))

    results = {
        "name": [],
        "acc": [],
        "f1": [],
        "auc": [],
        "log_loss": [],
        "mse": [],
        "corr": [],
        "r2": [],
        "train_time": [],
        "inference_time": [],
        "beta": [],
    }

    model_path = "16k_last_380epoch.ckpt"
    normalizer = 'standard'
    model_cls = TabDPTClassifier(
        inf_batch_size=args.inf_batch_size, device=device, model_weight_path=model_path, compile=False,
        feature_reduction='subsample',
        normalizer=normalizer,
        
    )
    model_reg = TabDPTRegressor(
        inf_batch_size=args.inf_batch_size,
        device=device,
        model_weight_path=model_path,
        compile=False,
        beta=args.beta,
        feature_reduction='subsample',
        normalizer=normalizer,
    )

    pbar = tqdm(
        itertools.chain(itertools.product(["cls"], cc18_dids), itertools.product(["reg"], ctr23_dids)),
        total=(len(cc18_dids) + len(ctr23_dids)),
    )
    for mode, did in pbar:
        if mode == "cls":
            continue
            tid = did_tid_mapping[did]
            dataset = TabZillaDataset(task_id=tid, fold=args.fold)
            dataset.prepare_data(".cache")
            dataset_name = dataset.name
        else:
            dataset = OpenMLDataset("openml_dataset", task_id=int(did_tid_mapping[did]), fold=args.fold)
            dataset.prepare_data(".cache")
            dataset_name = dataset.openml_dataset.name

        pbar.set_description(f"Running {dataset_name}")

        X_train, y_train = dataset.train_instances()
        X_val, y_val = dataset.val_instances()
        X_train = np.concatenate([X_train, X_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)
        X_test, y_test = dataset.test_instances()

        X_train, X_test, removed_constant, removed_duplicate = prune_features_if_needed(X_train, X_test)

        if mode == "cls":
            t0 = time()
            model_cls.fit(X_train, y_train)
            train_time = time() - t0

            t1 = time()
            pred_val = model_cls.ensemble_predict_proba(
                X_test,
                temperature=args.temperature,
                context_size=args.context_size,
                n_ensembles=args.n_ensembles,
                seed=args.seed,
            )
            inference_time = time() - t1

            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, pred_val[:, 1])
            else:
                auc = roc_auc_score(y_test, pred_val, multi_class="ovo")

            f1 = f1_score(y_test, np.argmax(pred_val, axis=1), average="weighted")
            acc = accuracy_score(y_test, np.argmax(pred_val, axis=1))
            ce_loss = log_loss(y_test, pred_val)
            mse, corr, r2 = None, None, None

            print(
                f"[CLS] {dataset_name}: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}, "
                f"loss={ce_loss:.4f}, train={train_time:.2f}s, test={inference_time:.2f}s"
            )
        else:
            # y_train = np.sign(y_train) * np.log1p(np.abs(y_train))
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

            t0 = time()
            model_reg.fit(X_train, y_train_scaled)
            train_time = time() - t0

            t1 = time()
            # if args.reg_use_quantile_refine:
            # pred_val_scaled = model_reg.predict_sorted_refine(
            #     X_test,
            #     context_size=args.context_size,
            #     seed=args.seed,
            #     beta=args.beta,
            #     n_ensembles=args.n_ensembles,
            #     )
            # else:
            pred_val_scaled = model_reg.predict(
                X_test,
                context_size=args.context_size,
                n_ensembles=args.n_ensembles,
                seed=args.seed,
                beta=args.beta,
                avg_logits=args.avg_logits,
            )
            inference_time = time() - t1
            # pred_val_scaled = (pred_val_scaled - np.mean(pred_val_scaled)) / (np.std(pred_val_scaled) + 1e-6)  # normalize
            pred_val = scaler.inverse_transform(pred_val_scaled.reshape(-1, 1)).ravel()
            # undo the signed log1p now
            # pred_val = np.sign(pred_val) * (np.expm1(np.abs(pred_val)))

            mse = np.mean((y_test - pred_val) ** 2)
            corr = scipy.stats.pearsonr(y_test, pred_val.flatten())[0]
            r2 = r2_score(y_test, pred_val)
            f1, acc, auc, ce_loss = None, None, None, None

            print(
                f"[REG] {dataset_name}: mse={mse:.4f}, corr={corr:.4f}, r2={r2:.4f}, "
                f"train={train_time:.2f}s, test={inference_time:.2f}s"
            )

        if removed_constant or removed_duplicate:
            print(
                f"  Pruned features (>{FEATURE_PRUNE_THRESHOLD}): "
                f"removed {removed_constant} constant, {removed_duplicate} duplicate; "
                f"{X_train.shape[1]} features remain"
            )

        results["name"].append(dataset_name)
        results["acc"].append(acc)
        results["f1"].append(f1)
        results["auc"].append(auc)
        results["log_loss"].append(ce_loss)
        results["mse"].append(mse)
        results["corr"].append(corr)
        results["r2"].append(r2)
        results["beta"].append(args.beta if mode == "reg" else np.nan)
        results["train_time"].append(train_time)
        results["inference_time"].append(inference_time)

    df = pd.DataFrame(results)

    datetime_string = datetime.now().isoformat(timespec="seconds").replace("T", "_").replace(":", "-")
    csv_name = (
        f"results_{args.precision_tag}_context={args.context_size}_"
        f"fold={args.fold}_N={args.n_ensembles}_seed={args.seed}_beta={args.beta}_.csv"
    )

    os.makedirs(args.results_folder, exist_ok=True)
    df.to_csv(os.path.join(args.results_folder, csv_name), index=False)
    print(f"Saved results to {os.path.join(args.results_folder, csv_name)}")

    print(f"IQM for Fold {args.fold}, N={args.n_ensembles}, T={args.temperature}:")
    metric_cols = ["acc", "auc", "corr", "r2"]
    metrics_df = df[metric_cols].apply(pd.to_numeric, errors="coerce")
    print(metrics_df.apply(np.nanmean))


if __name__ == "__main__":
    main()
