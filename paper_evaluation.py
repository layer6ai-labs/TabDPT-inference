import os
import argparse
import itertools
from time import time
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler
from rliable import metrics

from tabdpt import TabDPTClassifier
from tabdpt import TabDPTRegressor
from tabdpt_datasets.openml import OpenMLDataset, TabZillaDataset


CLS_DATASET_PATH = "tabdpt_datasets/data_splits/cls_datasets.csv"
REG_DATASET_PATH = "tabdpt_datasets/data_splits/reg_datasets.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TabDPT evaluation")
    parser.add_argument("--context_size", type=int, default=2048, help="Context size for the model")
    parser.add_argument("--fold", type=int, default=0, help="Fold number to use for evaluation")
    parser.add_argument("--n-ensembles", type=int, default=8, help="Number of ensembles to use for evaluation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for classification")
    parser.add_argument("--seed", type=int, default=0, help="Model evaluation seed")
    parser.add_argument("--inf-batch-size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--use-cpu", action="store_true", help="If true, use CPU for evalutation")
    parser.add_argument("--gpu-to-use", type=int, default=0, help="Which GPU to use")
    parser.add_argument("--results-folder", type=str, default="eval_output", help="Parent directory to store results")
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
    }
    model_cls = TabDPTClassifier(inf_batch_size=args.inf_batch_size, device=device)
    model_reg = TabDPTRegressor(inf_batch_size=args.inf_batch_size, device=device)

    pbar = tqdm(
        itertools.chain(itertools.product(["cls"], cc18_dids), itertools.product(["reg"], ctr23_dids)),
        total=(len(cc18_dids)+len(ctr23_dids))
    )
    for mode, did in pbar:
        if mode == "cls":
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

        if mode == "cls":
            model = model_cls

            t0 = time()
            model.fit(X_train, y_train)
            train_time = time() - t0

            t1 = time()
            pred_val = model.ensemble_predict_proba(
                X_test,
                temperature=args.temperature,
                context_size=args.context_size,
                n_ensembles=args.n_ensembles,
                seed=args.seed
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

        else:
            model = model_reg

            t0 = time()
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).ravel()
            model.fit(X_train, y_train_scaled)
            train_time = time() - t0

            t1 = time()
            pred_val_scaled = model.predict(
                X_test,
                context_size=args.context_size,
                n_ensembles=args.n_ensembles,
                seed=args.seed
            )
            inference_time = time() - t1
            pred_val = scaler.inverse_transform(pred_val_scaled.reshape(-1, 1)).ravel()

            mse = np.mean((y_test - pred_val) ** 2)
            corr = scipy.stats.pearsonr(y_test, pred_val.flatten())[0]
            r2 = r2_score(y_test, pred_val)
            f1, acc, auc, ce_loss = None, None, None, None

        results["name"].append(dataset_name)
        results["acc"].append(acc)
        results["f1"].append(f1)
        results["auc"].append(auc)
        results["log_loss"].append(ce_loss)
        results["mse"].append(mse)
        results["corr"].append(corr)
        results["r2"].append(r2)
        results["train_time"].append(train_time)
        results["inference_time"].append(inference_time)

        print(f"\nDataset: {dataset_name}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"train_time: {train_time:.4f}s, inference_time: {inference_time:.4f}s")

        if mode == "cls":
            print(f"acc {acc}, f1 {f1}, auc {auc}, loss {ce_loss}")
        else:
            print(f"mse {mse}, corr {corr}, r2 {r2}")

    df = pd.DataFrame(results)

    datetime_string = datetime.now().isoformat(timespec="seconds")
    datetime_string = datetime_string.replace("T", "_").replace(":", "-")
    csv_name = (
        f"results_{datetime_string}_context={args.context_size}_" \
        f"fold={args.fold}_N={args.n_ensembles}_seed={args.seed}.csv"
    )

    os.makedirs(args.results_folder, exist_ok=True)
    df.to_csv(os.path.join(args.results_folder, csv_name), index=False)

    def robust_iqm(x):
        try:
            x = x[~np.isnan(x)]
            return metrics.aggregate_iqm(x)
        except TypeError:
            return None

    print(f"IQM for Fold {args.fold}, N={args.n_ensembles}, T={args.temperature}:")
    print(df[["acc", "auc", "corr", "r2"]].apply(lambda x: robust_iqm(x)))
