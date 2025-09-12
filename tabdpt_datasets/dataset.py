from abc import ABC, abstractmethod

import numpy as np


class Dataset(ABC):
    """
    Base class for dataset classes.

    Each dataset class can be used for multiple datasets, distinguished by names.

    To support evaluation over a range of methods, the current code assumes that the entire dataset
    can be loaded into memory and accessed as numpy arrays using all_instances, which also supports
    fast indexing.  Fixed train/val/test splits are used, stored as indices to avoid duplication in
    memory. Synthetic datasets with a fixed set of instances can either generate their data at
    initialization or lazily.

    We will probably want to add separate methods for synthetic datasets that generate an indefinite
    number of instances on the fly, since these are less appropriate for evaluation and might not be
    usable for all baseline models anyway (e.g., decision trees that expect all instances to be
    available at once).
    """

    def __init__(self, name):
        """
        Params:
        name - the name of the dataset to load
        """
        self.name = name
        self.metadata = {"class": self.__class__.__name__}
        self.column_names = None

    @staticmethod
    @abstractmethod
    def all_names() -> list[str] | None:
        """
        Return the names of all datasets provided by this class, or None if it does not provide a
        fixed set of datasets
        """
        pass

    @abstractmethod
    def prepare_data(self, download_dir: str):
        """
        Download data if needed and do any CPU-side preprocessing, splitting, etc as needed.
        """
        pass

    def __len__(self) -> int:
        xs, _ = self.all_instances()
        return xs.shape[0]

    def __getitem__(self, i) -> tuple[np.ndarray, np.ndarray | int | float | None]:
        xs, ys = self.all_instances()
        if ys is not None:
            return xs[i], ys[i]
        return xs[i], None

    @abstractmethod
    def all_instances(self) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Return all instances as a feature matrix and target vector.
        """
        pass

    @abstractmethod
    def train_inds(self) -> list[int] | np.ndarray | range:
        pass

    @abstractmethod
    def val_inds(self) -> list[int] | np.ndarray | range:
        pass

    @abstractmethod
    def test_inds(self) -> list[int] | np.ndarray | range:
        pass

    def train_instances(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = self.all_instances()
        return X[self.train_inds()], y[self.train_inds()]

    def val_instances(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = self.all_instances()
        return X[self.val_inds()], y[self.val_inds()]

    def test_instances(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = self.all_instances()
        return X[self.test_inds()], y[self.test_inds()]

    def auto_populate_metadata(self):
        X, y = self.all_instances()
        self.metadata["size"] = X.shape[0]
        self.metadata["n_features"] = X.shape[1]
        if y is None:
            self.metadata["target_type"] = "none"
        else:
            self.metadata["y_mean"] = np.mean(y)
            self.metadata["y_var"] = np.var(y)
            if "target_type" not in self.metadata:
                self.metadata["target_type"] = "unknown"
        self.metadata["n_train"] = len(self.train_inds())
        self.metadata["n_val"] = len(self.val_inds())
        self.metadata["n_test"] = len(self.test_inds())

        lin_coeffs = []
        for i in range(X.shape[1]):
            col = np.nan_to_num(X[:, i])
            if np.all(np.isclose(col, col[0])):
                lin_coeffs.append(None)
                continue
            try:
                res = np.linalg.lstsq(np.stack((col, np.ones_like(col)), axis=1), y, rcond=None)
                lin_coeffs.append(res[0].tolist())
            except np.linalg.LinAlgError:
                lin_coeffs.append(None)
                continue
        self.metadata["column_lin_coeffs"] = lin_coeffs

        self.metadata["column_means"] = X.mean(axis=0).tolist()
        self.metadata["column_vars"] = X.var(axis=0).tolist()
