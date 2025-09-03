import gzip
import json
import os
import zipfile
import warnings
from pathlib import Path

import gdown
import numpy as np
import openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .dataset import Dataset


class OpenMLDataset(Dataset):
    """
    Generic class for loading any OpenML dataset
    """

    # OpenML typically only uses train/test splits, so get a 15% val from the train set
    _VALIDATION_FRACTION = 0.15

    # Although in some cases it does not even provide train/test splits
    _TEST_FRACTION = 0.15

    assert _VALIDATION_FRACTION + _TEST_FRACTION < 1

    @staticmethod
    def all_names():
        return None

    def __init__(
        self,
        name,
        dataset_id: int | None = None,
        task_id: int | None = None,
        fold: int = 0,
        split_seed: int = 0,
    ):
        super().__init__(name)
        if (dataset_id is None) == (task_id is None):
            raise ValueError("Must specify exactly one of dataset_id or task_id")
        self.did = dataset_id
        self.tid = task_id
        self.rng = np.random.default_rng(split_seed)
        self.split_seed = split_seed
        self.fold = fold

    @property
    def unique_name(self) -> str:
        """Returns a unique name for the dataset that can be used as an identifier"""
        if self.did:
            return f"OpenML-did={self.did}-{self.name}-seed={self.split_seed}"
        else:
            return f"OpenML-tid={self.tid}-{self.name}-fold={self.fold}-seed={self.split_seed}"

    @property
    def openml_dataset(self):
        if not hasattr(self, "_openml_dataset"):
            raise ValueError("Data not loaded yet")
        return self._openml_dataset

    @retry(
        stop=stop_after_attempt(3),
        # Wait strategy: Wait exponentially, starting at 0.5s, with a max of 10s
        wait=wait_exponential(multiplier=0.5, max=30),
        retry=retry_if_exception_type(openml.exceptions.OpenMLServerError),
    )
    def prepare_data(self, download_dir):

        openml.config.set_root_cache_directory(os.path.join(download_dir, "openml_cache"))

        if self.tid:
            task = openml.tasks.get_task(
                self.tid,
                download_splits=True,
                download_data=True,
                download_qualities=True,
                download_features_meta_data=True,
            )
            dataset = openml.datasets.get_dataset(
                task.dataset_id,
                download_data=True,
                download_qualities=True,
                download_features_meta_data=True,
            )
            X, y, _, self.column_names = dataset.get_data(target=dataset.default_target_attribute)
            n = len(X)

            # Some tasks have multiple folds
            split = task.get_train_test_split_indices(fold=self.fold)
            perm = self.rng.permutation(len(split.train))

            val_split_point = int((len(split.train) + len(split.test)) * self._VALIDATION_FRACTION)
            self._train_inds = split.train[perm[val_split_point:]]
            self._val_inds = split.train[perm[:val_split_point]]
            self._test_inds = split.test

            self.metadata["openml_task_id"] = self.tid
            self.metadata["openml_dataset_id"] = dataset.dataset_id

        else:
            dataset = openml.datasets.get_dataset(
                self.did,
                download_data=True,
                download_qualities=True,
                download_features_meta_data=True,
            )
            X, y, _, self.column_names = dataset.get_data(target=dataset.default_target_attribute)
            n = len(X)

            perm = self.rng.permutation(n)

            train_end_ind = int(n * (1 - self._VALIDATION_FRACTION - self._TEST_FRACTION))
            test_start_ind = int(n * (1 - self._TEST_FRACTION))
            self._train_inds = perm[:train_end_ind]
            self._val_inds = perm[train_end_ind:test_start_ind]
            self._test_inds = perm[test_start_ind:]

            self.metadata["openml_dataset_id"] = self.did

        if dataset.default_target_attribute and "," in dataset.default_target_attribute:
            y = None
            warnings.warn(
                f"Dataset {self.metadata['openml_dataset_id']} has multiple targets, which is "
                "not supported. Omitting targets."
            )

        self._openml_dataset = dataset

        categorical_inds = []
        for i, col in enumerate(X.columns):
            if X[col].dtype == "object" or pd.api.types.is_categorical_dtype(X[col]):
                enc = OrdinalEncoder()
                X[[col]] = enc.fit_transform(X[[col]])
                categorical_inds.append(i)
        self.metadata["categorical_feature_inds"] = categorical_inds
        self.X = X.to_numpy().astype(np.float32)

        if y is None:
            self.y = None
            self.metadata["target_type"] = "none"
            return
        target_feature = [f for f in dataset.features.values() if f.name == dataset.default_target_attribute][0]
        if target_feature.data_type == "nominal" or y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
            enc = LabelEncoder()
            self.y = enc.fit_transform(y)
            self.metadata["target_type"] = "classification"
        else:
            self.y = y.to_numpy().astype(np.float32)
            self.metadata["target_type"] = "regression"

    def all_instances(self):
        return self.X, self.y

    def train_inds(self):
        return self._train_inds

    def val_inds(self):
        return self._val_inds

    def test_inds(self):
        return self._test_inds


class TabZillaDataset(Dataset):
    @staticmethod
    def all_names():
       return TABZILLA_NAME_LIST

    def __init__(
            self,
            name: str | None = None,
            task_id: int | None = None,
            fold: int = 0
        ):
        super().__init__(name)

        # check at least one of name or task_id is provided
        if name is None and task_id is None:
            raise ValueError("Must specify at least one of name or task_id")

        name_match = None
        task_id_match = None
        for i, n in enumerate(self.all_names()):
            if name == n or (name is not None and n.split("__")[1] == name):
                name_match = i
            if task_id is not None and int(n.split("__")[2]) == task_id:
                task_id_match = i

        if name_match is not None and task_id_match is not None and name_match != task_id_match:
            raise ValueError("Both name and task_id provided, but they don't match")
        if name_match is None and task_id_match is None:
            raise ValueError("No matching dataset found")

        idx = name_match if name_match is not None else task_id_match
        self.tid = int(self.all_names()[idx].split("__")[2])
        self.name = self.all_names()[idx].split("__")[1]
        self.fold = fold

    def prepare_data(self, download_dir):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        subdir = Path(download_dir)
        if not (subdir / "tabzilla" / "DONE").exists():
            zip_path = os.path.join(download_dir, "tabzilla.zip")
            gdown.download(id="1Zzhekd0auCGdDjWKsthaxMiOAlAzPuoR", output=zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(subdir)
            os.remove(zip_path)
            # indicate that the download is done
            with open(subdir / "tabzilla" / "DONE", "w") as f:
                pass

        task_dir = subdir / "tabzilla" / f"openml__{self.name}__{self.tid}"

        with gzip.GzipFile(task_dir / "X.npy.gz", "rb") as f:
            self.X = np.load(f, allow_pickle=True).astype(np.float32)
        with gzip.GzipFile(task_dir / "y.npy.gz", "rb") as f:
            self.y = np.load(f, allow_pickle=True)

        # load json metadata
        with open(subdir / "tabzilla" / f"openml__{self.name}__{self.tid}" / "metadata.json", "r") as f:
            self.metadata["tabzilla"] = json.load(f)
        self.metadata["target_type"] = (
            "classification" if self.metadata["tabzilla"]["target_type"] != "regression" else "regression"
        )
        self.metadata["categorical_feature_inds"] = self.metadata["tabzilla"]["cat_idx"]
        if self.metadata["target_type"] == "classification":
            self.y = self.y.astype(np.int32)
        else:
            self.y = self.y.astype(np.float32)

        # load the splits
        with gzip.GzipFile(
            os.path.join(download_dir, "tabzilla", f"openml__{self.name}__{self.tid}", "split_indeces.npy.gz"),
            "rb",
        ) as f:
            self.folds = np.load(f, allow_pickle=True)

    @property
    def unique_name(self):
        return f"TabZilla-OpenML-{self.name}-tid={self.tid}-fold={self.fold}"

    def all_instances(self):
        return self.X, self.y

    def train_inds(self):
        return self.folds[self.fold]["train"]

    def val_inds(self):
        return self.folds[self.fold]["val"]

    def test_inds(self):
        return self.folds[self.fold]["test"]


TABZILLA_NAME_LIST = [
    "openml__acute-inflammations__10089",
    "openml__ada_agnostic__3896",
    "openml__adult__7592",
    "openml__adult-census__3953",
    "openml__airlines__189354",
    "openml__albert__189356",
    "openml__aloi__12732",
    "openml__Amazon_employee_access__34539",
    "openml__analcatdata_authorship__3549",
    "openml__analcatdata_boxing1__3540",
    "openml__analcatdata_chlamydia__3739",
    "openml__analcatdata_dmft__3560",
    "openml__anneal__2867",
    "openml__APSFailure__168868",
    "openml__arrhythmia__5",
    "openml__artificial-characters__14964",
    "openml__audiology__7",
    "openml__Australian__146818",
    "openml__autos__9",
    "openml__balance-scale__11",
    "openml__bank-marketing__14965",
    "openml__banknote-authentication__10093",
    "openml__Bank-Note-Authentication-UCI__361002",
    "openml__Bioresponse__9910",
    "openml__blood-transfusion-service-center__10101",
    "openml__BNG(credit-a)__146047",
    "openml__bodyfat__5514",
    "openml__breast-cancer__145799",
    "openml__breast-w__15",
    "openml__california__361089",
    "openml__car__146821",
    "openml__cardiotocography__9979",
    "openml__car-evaluation__146192",
    "openml__Census-Income__168340",
    "openml__chess__3952",
    "openml__christine__168908",
    "openml__chscase_foot__5012",
    "openml__churn__167141",
    "openml__CIFAR_10__167124",
    "openml__cjs__14967",
    "openml__cleveland__2285",
    "openml__Click_prediction_small__190408",
    "openml__climate-model-simulation-crashes__146819",
    "openml__cmc__23",
    "openml__cnae-9__9981",
    "openml__colic__25",
    "openml__colic__27",
    "openml__colleges__359942",
    "openml__collins__3567",
    "openml__connect-4__146195",
    "openml__covertype__7593",
    "openml__cpu_small__4883",
    "openml__credit-approval__29",
    "openml__credit-g__31",
    "openml__cylinder-bands__14954",
    "openml__dataset_sales__190418",
    "openml__dermatology__35",
    "openml__Devnagari-Script__167121",
    "openml__diabetes__37",
    "openml__dilbert__168909",
    "openml__dionis__189355",
    "openml__dna__167140",
    "openml__dresses-sales__125920",
    "openml__ecoli__145977",
    "openml__eeg-eye-state__14951",
    "openml__EgyptianSkulls__5040",
    "openml__electricity__219",
    "openml__elevators__3711",
    "openml__eucalyptus__2079",
    "openml__eye_movements__3897",
    "openml__fabert__168910",
    "openml__Fashion-MNIST__146825",
    "openml__fertility__9984",
    "openml__first-order-theorem-proving__9985",
    "openml__fl2000__3566",
    "openml__fri_c0_100_5__3620",
    "openml__fri_c3_100_5__3779",
    "openml__gas-drift__9986",
    "openml__gas-drift-different-concentrations__9987",
    "openml__GesturePhaseSegmentationProcessed__14969",
    "openml__gina_agnostic__3891",
    "openml__glass__40",
    "openml__guillermo__168337",
    "openml__haberman__42",
    "openml__har__14970",
    "openml__hayes-roth__146063",
    "openml__heart-c__48",
    "openml__heart-h__50",
    "openml__helena__168329",
    "openml__hepatitis__54",
    "openml__higgs__146606",
    "openml__hill-valley__145847",
    "openml__house_16H__3686",
    "openml__ilpd__9971",
    "openml__Internet-Advertisements__167125",
    "openml__ionosphere__145984",
    "openml__iris__59",
    "openml__irish__3543",
    "openml__isolet__3481",
    "openml__jannis__168330",
    "openml__JapaneseVowels__3510",
    "openml__jasmine__168911",
    "openml__jm1__3904",
    "openml__jungle_chess_2pcs_raw_endgame_complete__167119",
    "openml__kc1__3917",
    "openml__kc2__3913",
    "openml__kin8nm__2280",
    "openml__kropt__2076",
    "openml__kr-vs-kp__3",
    "openml__labor__4",
    "openml__ldpa__9974",
    "openml__LED-display-domain-7digit__125921",
    "openml__letter__6",
    "openml__libras__360948",
    "openml__liver-disorders__52948",
    "openml__lung-cancer__146024",
    "openml__lymph__10",
    "openml__madelon__9976",
    "openml__magic__146206",
    "openml__MagicTelescope__3954",
    "openml__meta__4729",
    "openml__mfeat-factors__12",
    "openml__mfeat-fourier__14",
    "openml__mfeat-karhunen__16",
    "openml__mfeat-morphological__18",
    "openml__mfeat-pixel__146824",
    "openml__mfeat-zernike__22",
    "openml__MiceProtein__146800",
    "openml__MiniBooNE__168335",
    "openml__mnist_784__3573",
    "openml__monks-problems-2__146065",
    "openml__mushroom__24",
    "openml__musk__3950",
    "openml__mv__4774",
    "openml__nomao__9977",
    "openml__numerai28.6__167120",
    "openml__nursery__9892",
    "openml__one-hundred-plants-texture__9956",
    "openml__optdigits__28",
    "openml__ozone-level-8hr__9978",
    "openml__page-blocks__30",
    "openml__pbc__4850",
    "openml__pc1__3918",
    "openml__pc3__3903",
    "openml__pc4__3902",
    "openml__pendigits__32",
    "openml__philippine__190410",
    "openml__PhishingWebsites__14952",
    "openml__phoneme__9952",
    "openml__poker-hand__9890",
    "openml__pollen__3735",
    "openml__postoperative-patient-data__146210",
    "openml__primary-tumor__146032",
    "openml__profb__3561",
    "openml__qsar-biodeg__9957",
    "openml__rabe_266__3647",
    "openml__riccardo__168338",
    "openml__robert__168332",
    "openml__Satellite__167211",
    "openml__satimage__2074",
    "openml__scene__3485",
    "openml__segment__146822",
    "openml__semeion__9964",
    "openml__shuttle__146212",
    "openml__sick__3021",
    "openml__skin-segmentation__9965",
    "openml__socmob__3797",
    "openml__solar-flare__2068",
    "openml__sonar__39",
    "openml__soybean__41",
    "openml__spambase__43",
    "openml__SpeedDating__146607",
    "openml__splice__45",
    "openml__steel-plates-fault__146817",
    "openml__sulfur__360966",
    "openml__sylva_agnostic__3889",
    "openml__sylvine__168912",
    "openml__synthetic_control__3512",
    "openml__tae__47",
    "openml__texture__125922",
    "openml__tic-tac-toe__49",
    "openml__transplant__3748",
    "openml__vehicle__53",
    "openml__veteran__4828",
    "openml__visualizing_environmental__3602",
    "openml__visualizing_livestock__3731",
    "openml__volkert__168331",
    "openml__vowel__3022",
    "openml__walking-activity__9945",
    "openml__wall-robot-navigation__9960",
    "openml__wdbc__9946",
    "openml__wilt__146820",
    "openml__Wine__190420",
    "openml__Wisconsin-breast-cancer-cytology-features__361003",
    "openml__yeast__145793",
]
