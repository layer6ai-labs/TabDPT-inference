import argparse
import itertools
import os
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
import scipy
from rliable import metrics
from sklearn.metrics import accuracy_score, f1_score, log_loss, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from tabdpt import TabDPTClassifier, TabDPTRegressor
from tabdpt_datasets.openml import OpenMLDataset, TabZillaDataset

CLS_DATASET_PATH = "tabdpt_datasets/data_splits/cls_datasets.csv"
REG_DATASET_PATH = "tabdpt_datasets/data_splits/reg_datasets.csv"

# fmt: off
TABARENA_IDS = [
    363621, 363629, 363614, 363698, 363626, 363685, 363625, 363696, 363675, 363707, 363671, 363612,
    363615, 363711, 363682, 363684, 363674, 363700, 363702, 363620, 363677, 363704, 363623, 363697,
    363694, 363708, 363706, 363689, 363624, 363619, 363676, 363712, 363632, 363691, 363681, 363686,
    363679, 363678, 363705, 363627, 363613, 363618, 363672, 363693, 363683, 363631, 363630, 363616,
    363699, 363628, 363673
]
# fmt: on


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TabDPT evaluation")
    parser.add_argument("--context_size", type=int, help="Context size for the model")
    parser.add_argument("--fold", type=int, default=0, help="Fold number to use for evaluation")
    parser.add_argument("--n-ensembles", type=int, default=8, help="Number of ensembles to use for evaluation")
    parser.add_argument("--temperature", type=float, default=1., help="Temperature for classification")
    parser.add_argument("--seed", type=int, default=0, help="Model evaluation seed")
    parser.add_argument("--batch-size", type=int, help="Batch size for inference")
    parser.add_argument("--use-cpu", action="store_true", help="If true, use CPU for evalutation")
    parser.add_argument("--gpu-to-use", type=int, default=0, help="Which GPU to use")
    parser.add_argument("--results-folder", type=str, default="eval_output", help="Parent results directory")
    parser.add_argument(
        "--suites", nargs="+", default=["cc18", "ctr23", "tabarena"], choices=["cc18", "ctr23", "tabarena"],
        help="Which suites to evaluate. Use to resume only the missing suites without redoing others.",
    )
    parser.add_argument(
        "--model-weight-path", type=str, default=None,
        help="Path to model weights. Defaults to the estimator's default (v1.2).",
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
        "suite": [],
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
    model_cls = TabDPTClassifier(device=device, model_weight_path=args.model_weight_path)
    model_reg = TabDPTRegressor(device=device, model_weight_path=args.model_weight_path)

    suites = set(args.suites)
    tasks = []
    if "cc18" in suites:
        tasks += list(zip(itertools.repeat("cc18"), cc18_dids))
    if "ctr23" in suites:
        tasks += list(zip(itertools.repeat("ctr23"), ctr23_dids))
    if "tabarena" in suites:
        tasks += list(zip(itertools.repeat("tabarena"), TABARENA_IDS))

    print(f"""Total of {len(tasks)} tasks are loaded from {', '.join(suites)}.""")

    os.makedirs(args.results_folder, exist_ok=True)
    datetime_string = datetime.now().isoformat(timespec="seconds").replace("T", "_").replace(":", "-")
    csv_name = (
        f"results_{datetime_string}_context={args.context_size}_"
        f"fold={args.fold}_N={args.n_ensembles}_seed={args.seed}.csv"
    )
    csv_path = os.path.join(args.results_folder, csv_name)

    pbar = tqdm(tasks, total=len(tasks))
    for suite, ident in pbar:
        if suite == "cc18":
            dataset = TabZillaDataset(task_id=did_tid_mapping[ident], fold=args.fold)
            dataset.prepare_data(".cache")
            dataset_name = dataset.name
            mode = "cls"
        elif suite == "ctr23":
            dataset = OpenMLDataset("openml_dataset", task_id=int(did_tid_mapping[ident]), fold=args.fold)
            dataset.prepare_data(".cache")
            dataset_name = dataset.openml_dataset.name
            mode = "reg"
        else:
            dataset = OpenMLDataset("openml_dataset", task_id=int(ident), fold=args.fold)
            dataset.prepare_data(".cache")
            dataset_name = dataset.openml_dataset.name
            mode = "cls" if dataset.metadata["target_type"] == "classification" else "reg"

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
                batch_size=args.batch_size,
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
                batch_size=args.batch_size,
                n_ensembles=args.n_ensembles,
                seed=args.seed
            )
            inference_time = time() - t1
            pred_val = scaler.inverse_transform(pred_val_scaled.reshape(-1, 1)).ravel()

            mse = np.mean((y_test - pred_val) ** 2)
            corr = scipy.stats.pearsonr(y_test, pred_val.flatten())[0]
            r2 = r2_score(y_test, pred_val)
            f1, acc, auc, ce_loss = None, None, None, None

        results["suite"].append(suite)
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

        pd.DataFrame(results).to_csv(csv_path, index=False)

        print(f"\nDataset: {dataset_name}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"train_time: {train_time:.4f}s, inference_time: {inference_time:.4f}s")

        if mode == "cls":
            print(f"acc {acc}, f1 {f1}, auc {auc}, loss {ce_loss}")
        else:
            print(f"mse {mse}, corr {corr}, r2 {r2}")

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    def robust_iqm(x):
        try:
            x = np.asarray(x, dtype=float)
            x = x[~np.isnan(x)]
            return metrics.aggregate_iqm(x) if len(x) else None
        except TypeError:
            return None

    weights_used = args.model_weight_path or "<estimator default (v1.2)>"
    lines = [
        f"TabDPT v1.2 paper evaluation",
        f"weights: {weights_used}",
        f"fold={args.fold}  N={args.n_ensembles}  T={args.temperature}  context={args.context_size}  seed={args.seed}",
        f"csv: {csv_name}",
        "",
    ]
    for suite in ["cc18", "ctr23", "tabarena"]:
        sub = df[df["suite"] == suite]
        if sub.empty:
            continue
        lines.append(f"[{suite}] datasets={len(sub)}  IQM:")
        for metric in ["acc", "auc", "corr", "r2"]:
            val = robust_iqm(sub[metric])
            if val is not None:
                lines.append(f"    {metric}: {val:.4f}")
    lines.append("")
    lines.append("[all] IQM:")
    for metric in ["acc", "auc", "corr", "r2"]:
        val = robust_iqm(df[metric])
        if val is not None:
            lines.append(f"    {metric}: {val:.4f}")

    summary = "\n".join(lines)
    print(summary)
    txt_name = csv_name.replace(".csv", ".txt")
    with open(os.path.join(args.results_folder, txt_name), "w") as f:
        f.write(summary + "\n")
