import torch
import numpy as np
import awkward as ak
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ElbowFinder:
    def __init__(self, encoding: np.array, k: int = 4):
        self.encoding = encoding
        self.k = k
        self.k_dist, self.elbow_y, self.elbow_idx = self._fit()

    def _find_elbow_chord(self, k_dist):
        x = np.arange(len(k_dist))
        y = k_dist
        # Line between endpoints
        x1, y1 = x[0], y[0]
        x2, y2 = x[-1], y[-1]
        # Distance of each point from the line
        num = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        dist = num / den
        elbow_idx = np.argmax(dist)
        elbow_val = y[elbow_idx]
        return elbow_val, elbow_idx

    def _fit(self):
        nn = NearestNeighbors(n_neighbors=self.k)
        nn.fit(self.encoding)
        distances, _ = nn.kneighbors(self.encoding)
        # take distance to the k-th nearest neighbor
        k_dist = np.sort(distances[:, -1])
        elbow_y, elbow_idx = self._find_elbow_chord(k_dist)
        return k_dist, elbow_y, elbow_idx

    def plot(self, ax):
        ax.scatter(self.elbow_idx, self.elbow_y, color="r")
        ax.plot(self.k_dist)
        ax.set_xlabel("Points sorted")
        ax.set_ylabel("k-distance")


class DistanceFitter:
    def __init__(self, encodings: list, k: int = 4):
        self.encodings = encodings
        self.k = k
        self.fitters = self._instantiate_fitters()
        self.eps = np.mean([fitter.elbow_y for fitter in self.fitters])

    def _instantiate_fitters(self):
        fitters = []
        for encoding in self.encodings:
            fitters.append(ElbowFinder(encoding=encoding, k=self.k))
        return fitters

    def plot(self, output_path: str = "", ncols: int = 3):
        nrows = int(np.ceil(len(self.fitters) / ncols))
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(ncols * 10, nrows * 10), sharey=True
        )
        axs = axs.flatten()
        for idx, fitter in enumerate(self.fitters):
            ax = axs[idx]
            fitter.plot(ax)
        if output_path != "":
            fig.savefig(output_path)
        else:
            plt.show()


class EncodingEvaluator:
    def __init__(
        self,
        model,
        test_loader,
        estimate_eps: bool = True,
        mpl_cmap: str = "jet",
        autosave_plots: bool = True,
        output_dir: str = "",
    ):
        self.model = model
        self.test_loader = test_loader
        self.estimate_eps = estimate_eps
        self.batch = self._load_single_batch()
        self.cmap = plt.get_cmap(mpl_cmap)
        self.hit_coordinates, self.encodings, self.true_labels = (
            self._get_event_data_from_batch()
        )
        self.all_encodings = ak.flatten(self.encodings)
        self.scaled_encodings = self._standardize_encodings()
        self.DBSCAN_clusters = self._get_DBSCAN_clusters()
        self.tSNE_encodings = self._get_tSNE_encodings()
        self.PCA_encodings = self._get_PCA_encodings()
        self.output_dir = output_dir
        if autosave_plots:
            self._autosave()

    def _autosave(self):
        if self.output_dir == "":
            print("No output_dir specified. Saving to CWD")
        self.plot_encoding_distribution(savefig=True)
        self.plot_SPVCNN_encoding(savefig=True)
        self.plot_tsne(savefig=True)
        self.plot_DBSCAN_results(savefig=True)

    def _load_single_batch(self):
        return next(iter(self.test_loader))

    def _get_event_data_from_batch(self):
        print("... Splitting batch to events")
        truth = self.batch["instance_labels"].F
        event_indices = self.batch["coordinates"].C[:, 3]
        outputs = self.model(self.batch["features"])
        unique_events = np.unique(event_indices.numpy())
        true_labels = []
        hit_coordinates = []
        encodings = []
        for event_idx in unique_events:
            event_mask = event_indices == event_idx
            event_hits = self.batch["coordinates"].F[:, :3][event_mask].detach().numpy()
            event_hits_encoding = outputs[event_mask].detach().numpy()
            event_hits_true_labels = truth[event_mask].detach().numpy()
            encodings.append(event_hits_encoding)
            true_labels.append(event_hits_true_labels)
            hit_coordinates.append(event_hits)
        return hit_coordinates, encodings, true_labels

    def _standardize_encodings(self):
        print("... Scaling encodings")
        scaler = StandardScaler()
        scaler.fit(self.all_encodings)
        scaled_encodings = [scaler.transform(encoding) for encoding in self.encodings]
        return scaled_encodings

    def _get_DBSCAN_clusters(self, eps: float = 2.12):
        print("... Clustering using DBSCAN")
        if self.estimate_eps:
            print("    (Estimating the `eps` parameter for DBSCAN using kNN)")
            eps = DistanceFitter(self.scaled_encodings).eps
        clusters = []
        for encoding in self.scaled_encodings:
            db = DBSCAN(eps=eps, min_samples=5).fit(encoding)
            cluster_ids = db.labels_  # -1 = noise
            clusters.append(cluster_ids)
        return clusters

    def _get_tSNE_encodings(self):
        print("... Evaluating using tSNE")
        tsne_encodings = []
        for encoding in self.scaled_encodings:
            tsne_encodings.append(TSNE(n_components=2).fit_transform(encoding))
        return tsne_encodings

    def _get_PCA_encodings(self):
        print("... Evaluating using PCA")
        pca_encodings = []
        for encoding in self.scaled_encodings:
            pca_encodings.append(PCA(n_components=2).fit_transform(encoding))
        return pca_encodings

    def plot_encoding_distribution(
        self,
        num_columns: int = 3,
        min_bin: float = 1e-5,
        max_bin: float = 50,
        savefig: bool = False,
        filename: str = "SPVCNN_encoding_distribution.png",
    ):
        num_rows = int(np.ceil(8 / num_columns))
        fig, axs = plt.subplots(
            num_rows, num_columns, figsize=(num_columns * 10, num_rows * 10)
        )
        axs = axs.flatten()
        bins = np.logspace(np.log10(min_bin), np.log10(max_bin), num=101)
        for encoding_dim, _ in enumerate(tqdm(self.encodings[0][0])):
            axs[encoding_dim].hist(
                np.abs(self.all_encodings[:, encoding_dim]), bins=bins
            )
            axs[encoding_dim].set_yscale("log")
            axs[encoding_dim].set_xscale("log")
        if savefig:
            output_path = os.path.join(self.output_dir, filename)
            fig.savefig(output_path)
        else:
            plt.show()
        plt.close("all")

    def plot_tsne(
        self,
        num_columns: int = 3,
        savefig: bool = False,
        filename: str = "tSNE_encoding.png",
    ):
        num_rows = int(np.ceil(8 / num_columns))
        fig, axs = plt.subplots(
            num_rows, num_columns, figsize=(num_columns * 10, num_rows * 10)
        )
        axs = axs.flatten()
        num_plots = min(len(axs), len(self.true_labels))
        for idx, ax in enumerate(axs):
            if (idx + 1) > num_plots:
                continue
            color_map = self.cmap(
                np.linspace(0, 1, len(np.unique(self.true_labels[idx])))
            )
            colors = color_map[self.true_labels[idx]]
            ax.scatter(
                self.tSNE_encodings[idx][:, 0], self.tSNE_encodings[idx][:, 1], c=colors
            )
        if savefig:
            output_path = os.path.join(self.output_dir, filename)
            fig.savefig(output_path)
        else:
            plt.show()
        plt.close("all")

    def plot_pca(
        self,
        num_columns: int = 3,
        savefig: bool = False,
        filename: str = "PCA_encoding.png",
    ):
        num_rows = int(np.ceil(8 / num_columns))
        fig, axs = plt.subplots(
            num_rows, num_columns, figsize=(num_columns * 10, num_rows * 10)
        )
        axs = axs.flatten()
        num_plots = min(len(axs), len(self.true_labels))
        for idx, ax in enumerate(axs):
            if (idx + 1) > num_plots:
                continue
            color_map = self.cmap(
                np.linspace(0, 1, len(np.unique(self.true_labels[idx])))
            )
            colors = color_map[self.true_labels[idx]]
            ax.scatter(
                self.PCA_encodings[idx][:, 0], self.PCA_encodings[idx][:, 1], c=colors
            )
        if savefig:
            output_path = os.path.join(self.output_dir, filename)
            fig.savefig(output_path)
        else:
            plt.show()
        plt.close("all")

    def plot_SPVCNN_encoding(
        self,
        num_columns: int = 3,
        encoding_pair: tuple = (0, 1),
        savefig: bool = False,
        filename: str = "SPVCNN_encoding.png",
    ):
        num_rows = int(np.ceil(8 / num_columns))
        fig, axs = plt.subplots(
            num_rows, num_columns, figsize=(num_columns * 10, num_rows * 10)
        )
        axs = axs.flatten()
        num_plots = min(len(axs), len(self.true_labels))
        for idx, ax in enumerate(axs):
            if (idx + 1) > num_plots:
                continue
            encoding = self.encodings[idx]
            color_map = self.cmap(
                np.linspace(0, 1, len(np.unique(self.true_labels[idx])))
            )
            colors = color_map[self.true_labels[idx]]
            ax.scatter(
                encoding[:, encoding_pair[0]], encoding[:, encoding_pair[1]], c=colors
            )
        if savefig:
            output_path = os.path.join(self.output_dir, filename)
            fig.savefig(output_path)
        else:
            plt.show()
        plt.close("all")

    def plot_DBSCAN_results(
        self,
        num_events: int = 8,
        savefig: bool = False,
        filename: str = "DBSCAN_clustering.png",
    ):
        fig, axs = plt.subplots(
            num_events,
            2,
            figsize=(20, num_events * 10),
            subplot_kw={"projection": "3d"},
        )
        axs = axs.flatten()
        num_plots = min(len(axs), len(self.true_labels))
        for idx, ax in enumerate(axs):
            if (idx + 1) > num_plots:
                continue
            event_reco_labels = self.DBSCAN_clusters[idx]
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
                c="r",
            )
        if savefig:
            output_path = os.path.join(self.output_dir, filename)
            fig.savefig(output_path)
        else:
            plt.show()
        plt.close("all")


@hydra.main(config_path="../../train_configs", config_name="config")
def main(cfg: DictConfig) -> None:

    cfg.dataset.batch_size = 8

    model = instantiate(cfg.model.target, cfg)
    ckpt = torch.load(cfg.ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("Model instantiated and state dict loaded")

    datamodule = instantiate(cfg.dataset)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    print("Datamodule instantiated and test_loader set up.")

    # with torch.no_grad():
    #     evaluate_training(test_loader, model)


if __name__ == "__main__":
    main()
