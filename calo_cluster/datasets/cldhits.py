from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import awkward as ak
from calo_cluster.datasets.base import BaseDataset, BaseDataModule

from pathlib import Path
from typing import List

from calo_cluster.datasets.mixins.offset import (
    OffsetDataModuleMixin,
    OffsetDatasetMixin,
)
from calo_cluster.datasets.mixins.scaled import (
    ScaledDataModuleMixin,
    ScaledDatasetMixin,
)
from calo_cluster.datasets.mixins.simple import SimpleDataModuleMixin
from calo_cluster.datasets.mixins.sparse import (
    SparseDataModuleMixin,
    SparseDatasetMixin,
)


def get_hit_labels(hit_idx, gen_idx, weights):
    """
    Assign labels to hits based on the genparticle index with the highest weight.

    Parameters:
        hit_idx (np.ndarray): Array of hit indices.
        gen_idx (np.ndarray): Array of genparticle indices corresponding to each hit.
        weights (np.ndarray): Array of weights corresponding to each hit.

    Returns:
        np.ndarray: Array of labels for each hit, where each label corresponds to the genparticle index.
    """
    # Initialize an array to store labels for each hit
    hit_labels = np.full(
        np.max(hit_idx) + 1, -1, dtype=int
    )  # Default label is -1 (unclassified)
    hit_label_weights = dict()  # To keep track of the highest weight for each hit

    # Iterate through the sparse COO matrix data
    for h_idx, g_idx, weight in zip(hit_idx, gen_idx, weights):
        if hit_labels[h_idx] == -1 or weight > hit_label_weights[h_idx]:
            hit_labels[h_idx] = g_idx
            hit_label_weights[h_idx] = weight

    # hit_labels now contains the genparticle index for each hit

    return hit_labels


def standardize_calo_hit_features(calo_hit_features):
    calo_hit_features[..., 0] = calo_hit_features[..., 0] / 1e4  # position x
    calo_hit_features[..., 1] = calo_hit_features[..., 1] / 1e4  # position y
    calo_hit_features[..., 2] = calo_hit_features[..., 2] / 1e4  # position z
    calo_hit_features[..., 3] = np.log(calo_hit_features[..., 3] * 1e2) / 10  # energy
    return calo_hit_features


def inverse_standardize_calo_hit_features(calo_hit_features):
    calo_hit_features[..., 0] = calo_hit_features[..., 0] * 1e4  # position x
    calo_hit_features[..., 1] = calo_hit_features[..., 1] * 1e4  # position y
    calo_hit_features[..., 2] = calo_hit_features[..., 2] * 1e4  # position z
    calo_hit_features[..., 3] = np.exp(calo_hit_features[..., 3] * 10) / 1e2  # energy
    return calo_hit_features


@dataclass
class CLDHitsDataset(SparseDatasetMixin, ScaledDatasetMixin, BaseDataset):
    # feats: List[str]
    # coords: List[str]
    # weight: str
    # sparse: bool
    # voxel_size: float

    def _get_numpy(self, index: int):
        file = self.files[index]
        data = ak.from_parquet(file)
        genparticle_to_calo_hit_matrix = data["genparticle_to_calo_hit_matrix"]
        all_calo_hit_features = data["calo_hit_features"]

        gen_idx = genparticle_to_calo_hit_matrix["gen_idx"].to_numpy()
        hit_idx = genparticle_to_calo_hit_matrix["hit_idx"].to_numpy()
        weights = genparticle_to_calo_hit_matrix["weight"].to_numpy()

        calo_hit_features = np.column_stack(
            (
                all_calo_hit_features["type"].to_numpy(),
                all_calo_hit_features["subdetector"].to_numpy(),
                all_calo_hit_features["energy"].to_numpy(),
            )
        )
        calo_hit_coordinates = np.column_stack(
            (
                all_calo_hit_features["position.x"].to_numpy(),
                all_calo_hit_features["position.y"].to_numpy(),
                all_calo_hit_features["position.z"].to_numpy(),
            )
        )

        hit_labels = get_hit_labels(
            hit_idx, gen_idx, weights
        )  # This could be moved to the pre-processing step if needed

        _, contiguous_labels = np.unique(hit_labels, return_inverse=True)
        return {
            "features": calo_hit_features,
            "coordinates": calo_hit_coordinates,
            "weights": weights,
            "instance_labels": contiguous_labels,
        }


@dataclass
class CLDHitsDataModule(SparseDataModuleMixin, ScaledDataModuleMixin, BaseDataModule):
    sparse: bool
    voxel_size: float

    def make_dataset(self, files: List[Path], split: str) -> CLDHitsDataset:
        kwargs = self.make_dataset_kwargs()
        kwargs["sparse"] = self.sparse
        kwargs["voxel_size"] = self.voxel_size
        return CLDHitsDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append("dataset=cldhits")
        return overrides
