import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from calo_cluster.evaluation import method_evaluator as me


def get_hit_labels(hit_idx, gen_idx, weights, max_hits=None):
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
    if not max_hits:
        max_hits = np.max(hit_idx) + 1
    hit_labels = np.full(max_hits, -1, dtype=int)  # Default label is -1 (unclassified)
    hit_label_weights = dict()  # To keep track of the highest weight for each hit

    # Iterate through the sparse COO matrix data
    for h_idx, g_idx, weight in zip(hit_idx, gen_idx, weights):
        if hit_labels[h_idx] == -1 or weight > hit_label_weights[h_idx]:
            hit_labels[h_idx] = g_idx
            hit_label_weights[h_idx] = weight

    # hit_labels now contains the genparticle index for each hit

    return hit_labels


class PandoraCluster:
    def __init__(
        self,
        data_dir: str = "../data/p8_ee_tt_ecm365/",
        input_file: str = "reco_p8_ee_tt_ecm365_60000",
    ):
        self.data_dir = data_dir
        self.input_file = input_file
        self.data = self._load_data()

    def _load_data(self):
        file_path = os.path.join(self.data_dir, "parquet", f"{self.input_file}.parquet")
        data = ak.from_parquet(file_path)
        return data

    def get_event_info(self, event_idx: int):
        calo_hit_features = self.data["calo_hit_features"][event_idx]

        genparticle_to_calo_hit_matrix = self.data["genparticle_to_calo_hit_matrix"][
            event_idx
        ]

        hit_idx = genparticle_to_calo_hit_matrix["hit_idx"].to_numpy()
        gen_clusters = get_hit_labels(
            hit_idx=genparticle_to_calo_hit_matrix["hit_idx"].to_numpy(),
            gen_idx=genparticle_to_calo_hit_matrix["gen_idx"].to_numpy(),
            weights=genparticle_to_calo_hit_matrix["weight"].to_numpy(),
        )

        cluster_to_cluster_hit_matrix = self.data["cluster_to_cluster_hit_matrix"][
            event_idx
        ]
        pandora_clusters = get_hit_labels(
            cluster_to_cluster_hit_matrix["hit_idx"],
            cluster_to_cluster_hit_matrix["cluster_idx"],
            cluster_to_cluster_hit_matrix["weight"],
            max_hits=np.max(hit_idx) + 1,
        )

        hit_coordinates = np.column_stack(
            (
                calo_hit_features["position.x"].to_numpy(),
                calo_hit_features["position.y"].to_numpy(),
                calo_hit_features["position.z"].to_numpy(),
            )
        )
        _, gen_contiguous_labels = np.unique(gen_clusters, return_inverse=True)
        _, pandora_contiguous_labels = np.unique(pandora_clusters, return_inverse=True)
        return {
            "gen_clusters": gen_contiguous_labels,
            "pandora_clusters": pandora_contiguous_labels,
            "calo_hit_features": calo_hit_features,
            "hit_coordinates": hit_coordinates,
        }

    def visualize_pred_vs_true(
        self, num_events: int = 8, savefig: bool = False, output_dir: str = ""
    ):
        fig, axs = plt.subplots(
            num_events,
            2,
            figsize=(20, num_events * 10),
            subplot_kw={"projection": "3d"},
        )
        axs = axs.flatten()
        for idx in range(num_events):
            event_info = self.get_event_info(idx)
            event_reco_labels = event_info["pandora_clusters"]
            event_hit_coordinates = event_info["hit_coordinates"]
            unique_true_labels = np.unique(event_info["gen_clusters"])
            unique_reco_labels = np.unique(event_reco_labels)

            # for evaluation check how many "wrong" cluster label hits in the given cluster
            cmap = plt.get_cmap("jet")
            distinct_reco_colors = cmap(np.linspace(0, 1, len(unique_reco_labels)))
            distinct_true_colors = cmap(np.linspace(0, 1, len(unique_true_labels)))

            axs[idx * 2 + 1].set_box_aspect([1, 1, 1])
            axs[idx * 2 + 1].view_init(elev=20, azim=45)
            axs[idx * 2 + 1].scatter(
                event_hit_coordinates[:, 0],
                event_hit_coordinates[:, 1],
                event_hit_coordinates[:, 2],
                # s=np.clip(100*event_hit_coordinates[:, 3], 0.1, 10),
                c=distinct_true_colors[event_info["gen_clusters"]],
            )

            axs[idx * 2].set_box_aspect([1, 1, 1])
            axs[idx * 2].view_init(elev=20, azim=45)
            axs[idx * 2].scatter(
                event_hit_coordinates[:, 0][event_reco_labels != -1],
                event_hit_coordinates[:, 1][event_reco_labels != -1],
                event_hit_coordinates[:, 2][event_reco_labels != -1],
                # s=np.clip(100*event_hit_coordinates[:, 3], 0.1, 10),  # Here should be the energy
                c=distinct_reco_colors[event_reco_labels][event_reco_labels != -1],
            )
            axs[idx * 2].scatter(
                event_hit_coordinates[:, 0][event_reco_labels == -1],
                event_hit_coordinates[:, 1][event_reco_labels == -1],
                event_hit_coordinates[:, 2][event_reco_labels == -1],
                # s=np.clip(100*event_hit_coordinates[:, 3], 0.1, 10),  # Here should be the energy
                c="k",
            )

    def evaluate(self, num_evaluation_events: int):
        purities = []
        ious = []
        for idx in range(num_evaluation_events):
            event_info = self.get_event_info(idx)
            true_labels = event_info["gen_clusters"]
            pred_labels = event_info["pandora_clusters"]
            purity = me.ClusterPurity(
                true_labels=true_labels, pred_labels=pred_labels
            ).score
            iou = me.IntersectionOverUnion(
                true_labels=true_labels, pred_labels=pred_labels
            ).score
            purities.append(purity)
            ious.append(iou)
        print("Mean purity: ", np.mean(purities))
        print("Mean IoU: ", np.mean(ious))
