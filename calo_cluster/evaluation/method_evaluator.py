import numpy as np
import awkward as ak
from collections import Counter
from scipy.optimize import linear_sum_assignment
from calo_cluster.evaluation.cldhits_eval import DistanceFitter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D


class MethodBase:
    def __init__(self, method_name: str):
        self.method_name = method_name

    def get_predictions(self):
        raise ValueError("Please implement this function to your subclass")

    def standardize_input(self, inputs: ak.Array):
        all_inputs = ak.flatten(inputs)
        print("... Scaling inputs")
        scaler = StandardScaler()
        scaler.fit(all_inputs)
        scaled_inputs = [scaler.transform(input_) for input_ in inputs]
        return scaled_inputs


class SPVCNNPlusDBScan(MethodBase):
    def __init__(self, embeddings, eps: float = 2.12, estimate_eps: bool = True):
        self.embeddings = embeddings
        self.eps = eps
        self.estimate_eps = estimate_eps
        super().__init__(method_name="CoordinateBasedDBSCan")
        self.scaled_inputs = self.standardize_input(embeddings)

    def get_predictions(self):
        print("... Clustering using DBSCAN")
        if self.estimate_eps:
            print("    (Estimating the `eps` parameter for DBSCAN using kNN)")
            self.eps = DistanceFitter(self.scaled_inputs).eps
        clusters = []
        for encoding in self.scaled_inputs:
            db = DBSCAN(eps=self.eps, min_samples=5).fit(encoding)
            cluster_ids = db.labels_  # -1 = noise
            clusters.append(cluster_ids)
        return clusters


class CoordinateBasedDBScan(MethodBase):
    def __init__(self, coordinates, eps: float = 2.12, estimate_eps: bool = True):
        self.coordinates = coordinates
        self.eps = eps
        self.estimate_eps = estimate_eps
        super().__init__(method_name="CoordinateBasedDBSCan")
        self.scaled_inputs = self.standardize_input(coordinates)
        self.predicted_clusters = self.get_predictions()

    def get_predictions(self):
        print("... Clustering using DBSCAN")
        if self.estimate_eps:
            print("    (Estimating the `eps` parameter for DBSCAN using kNN)")
            self.eps = DistanceFitter(self.scaled_inputs).eps
        clusters = []
        for encoding in self.scaled_inputs:
            db = DBSCAN(eps=self.eps, min_samples=5).fit(encoding)
            cluster_ids = db.labels_  # -1 = noise
            clusters.append(cluster_ids)
        return clusters


class MethodEvaluator:
    def __init__(self, method, batch, model, n_evaluation_events: int = 8):
        self.batch = batch
        self.model = model
        self.n_evaluation_events = n_evaluation_events
        self.hit_coordinates, self.encodings, self.true_labels = (
            self._get_event_info_from_batch()
        )
        # self.hit_coordinates, self.encodings, self.true_labels = 0, 0, 0  # TODO: Temporary
        self.method = method
        self.predicted_clusters = None

    def evaluate(self):
        self.predicted_clusters = self.method.get_predictions()
        purities = []
        ious = []
        for true_labels, pred_labels in zip(self.true_labels, self.predicted_clusters):
            purity = ClusterPurity(
                true_labels=true_labels, pred_labels=pred_labels
            ).score
            iou = IntersectionOverUnion(
                true_labels=true_labels, pred_labels=pred_labels
            ).score
            purities.append(purity)
            ious.append(iou)
        print("Mean purity: ", np.mean(purities))
        print("Mean IoU: ", np.mean(ious))

    def _get_event_info_from_batch(self):
        print("... Splitting batch to events")
        truth = self.batch["instance_labels"].F
        event_indices = self.batch["coordinates"].C[:, 3]
        outputs = self.model(self.batch["features"])
        unique_events = np.unique(event_indices.numpy())
        true_labels = []
        hit_coordinates = []
        encodings = []
        n_eval = max([len(unique_events), self.n_evaluation_events])
        for event_idx in n_eval:
            event_mask = event_indices == event_idx
            event_hits = self.batch["coordinates"].F[:, :3][event_mask].detach().numpy()
            event_hits_encoding = outputs[event_mask].detach().numpy()
            event_hits_true_labels = truth[event_mask].detach().numpy()
            encodings.append(event_hits_encoding)
            true_labels.append(event_hits_true_labels)
            hit_coordinates.append(event_hits)
        hit_coordinates = ak.Array(hit_coordinates)
        encodings = ak.Array(encodings)
        true_labels = ak.Array(true_labels)
        return hit_coordinates, encodings, true_labels

    def plot_predicted_clusters(
        self, num_events: int = 8, savefig: bool = False, output_dir: str = ""
    ):
        fig, axs = plt.subplots(
            num_events,
            2,
            figsize=(20, num_events * 10),
            subplot_kw={"projection": "3d"},
        )
        axs = axs.flatten()
        num_plots = min(len(axs), len(self.true_labels))
        for idx in range(num_plots):
            event_reco_labels = self.predicted_clusters[idx]
            event_hit_coordinates = self.hit_coordinates[idx]
            unique_true_labels = np.unique(self.true_labels[idx])
            unique_reco_labels = np.unique(event_reco_labels)

            # for evaluation check how many "wrong" cluster label hits in the given cluster
            cmap = plt.get_cmap("jet")
            distinct_reco_colors = cmap(np.linspace(0, 1, len(unique_reco_labels) - 1))
            distinct_true_colors = cmap(np.linspace(0, 1, len(unique_true_labels)))

            axs[idx * 2 + 1].set_box_aspect([1, 1, 1])
            axs[idx * 2 + 1].view_init(elev=20, azim=45)
            axs[idx * 2 + 1].scatter(
                event_hit_coordinates[:, 0],
                event_hit_coordinates[:, 1],
                event_hit_coordinates[:, 2],
                # s=np.clip(100*event_hit_coordinates[:, 3], 0.1, 10),
                c=distinct_true_colors[self.true_labels[idx]],
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
        if savefig:
            output_path = os.path.join(
                output_dir, f"{self.method.method_name}_clusters.png"
            )
            fig.savefig(output_path)
        else:
            plt.show()
        plt.close("all")


class ClusterPurity:
    def __init__(self, true_labels: np.array, pred_labels: np.array):
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.purity, self.all_purities = self.cluster_purity()

    @property
    def score(self):
        return self.purity

    def cluster_purity(self):
        pred_clusters = np.unique(self.pred_labels)
        purities = []
        for p in pred_clusters:
            pred_hits = np.where(self.pred_labels == p)[0]
            # true labels within this predicted cluster
            true_ids = self.true_labels[pred_hits]
            # count how many hits belong to each true cluster
            counts = Counter(true_ids)
            largest_overlap = max(counts.values())
            purity = largest_overlap / len(pred_hits)
            purities.append(purity)
        # return average purity across predicted clusters
        return np.mean(purities), purities


class IntersectionOverUnion:
    def __init__(self, true_labels: np.array, pred_labels: np.array):
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.iou_score = self.cluster_iou_score()

    @property
    def score(self):
        return self.iou_score

    def compute_iou_matrix(self):
        """
        Compute IoU between all pairs of true and predicted clusters.
        """
        true_clusters = np.unique(self.true_labels)
        pred_clusters = np.unique(self.pred_labels)

        iou_matrix = np.zeros((len(true_clusters), len(pred_clusters)))

        for i, t in enumerate(true_clusters):
            true_set = set(np.where(self.true_labels == t)[0])
            for j, p in enumerate(pred_clusters):
                pred_set = set(np.where(self.pred_labels == p)[0])
                intersection = len(true_set & pred_set)
                union = len(true_set | pred_set)
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0
        return iou_matrix

    def cluster_iou_score(self):
        """
        Compute a global IoU score for all clusters using optimal matching.
        """
        iou_matrix = self.compute_iou_matrix()
        # Hungarian algorithm finds optimal matching that maximizes total IoU
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # maximize
        matched_iou = iou_matrix[row_ind, col_ind]
        # Average IoU over matched clusters
        score = matched_iou.mean()
        return score
