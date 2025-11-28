import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


MODEL_PATH = "/home/laurits/MLPF_clustering/hydra/2025-11-26/12-14-23/lightning_logs/version_58928844/checkpoints/14-4694.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def estimate_eps_from_single_knn(event_hits_encoding, k: int = 4):
    # Find the optimal distance parameter
    # y-value at the "elbow=point" is a good value for the epsilon value in DBSCAN.
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(event_hits_encoding)
    distances, _ = nn.kneighbors(event_hits_encoding)
    # take distance to the k-th nearest neighbor
    k_dist = np.sort(distances[:, -1])
    elbow_y, elbow_idx = find_elbow_chord(k_dist)
    return k_dist, elbow_y, elbow_idx


def estimate_mean_eps_from_knn(encodings: list, visualize: bool = False, k: int = 4):
    k_dists = []
    elbow_ys = []
    elbow_idxs = []
    for event_hits_encoding in encodings:
        event_hits_encoding = event_hits_encoding.detach().numpy()
        k_dist, elbow_y, elbow_idx = estimate_eps_from_single_knn(
            event_hits_encoding=event_hits_encoding, k=k
        )
        k_dists.append(k_dist)
        elbow_ys.append(elbow_y)
        elbow_idxs.append(elbow_idx)
    if visualize:
        visualize_knn_result(k_dists, elbow_ys, elbow_idxs)
    mean_y = np.mean(elbow_ys)
    print(f"Mean value for eta: {mean_y}")
    return mean_y


def split_batch_to_events(batch, model, estimate_eps: bool = False, eps: float = 2.12):
    print("Splitting batch into events")
    inputs = batch["features"]
    truth = batch["instance_labels"].F
    event_indices = batch["coordinates"].C[:, 3]
    outputs = model(inputs)
    unique_events = np.unique(event_indices.numpy())
    true_labels = []
    hit_coordinates = []
    reco_cluster_ids = []
    encodings = []
    for event_idx in unique_events:
        event_mask = event_indices == event_idx
        event_hits = batch["coordinates"].F[:, :3][event_mask].detach().numpy()
        event_hits_encoding = outputs[event_mask].detach().numpy()
        event_hits_true_labels = truth[event_mask].detach().numpy()
        encodings.append(event_hits_encoding)
        true_labels.append(event_hits_true_labels)
        hit_coordinates.append(event_hits)
    if estimate_eps:
        print("Estimating the `eps` parameter for DBSCAN using kNN")
        eps = estimate_mean_eps_from_knn(encodings, visualize=True)
    # Now cluster the embeddings
    for encoding in encodings:
        db = DBSCAN(eps=eps, min_samples=5).fit(encoding)
        cluster_ids = db.labels_  # -1 = noise
        reco_cluster_ids.append(cluster_ids)
    return hit_coordinates, reco_cluster_ids, true_labels


def visualize_knn_result(k_dists, elbow_ys, elbow_idxs):
    _, axs = plt.subplots(2, 3, figsize=(10, 10), sharey=True)
    axs = axs.flatten()
    for idx, k_dist in enumerate(k_dists):
        ax = axs[idx]
        elbow_idx = elbow_idxs[idx]
        y = elbow_ys[idx]
        k_dist = k_dists[idx]
        ax.scatter(elbow_idx, y, color="r")
        ax.plot(k_dist)
    plt.title("k-distance graph")
    plt.xlabel("Points sorted")
    plt.ylabel("k-distance")
    plt.show()


def find_elbow_chord(k_dist):
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


def visualize_batch_prediction_and_truth(
    hit_coordinates: np.array, reco_cluster_ids: np.array, true_labels: np.array
):
    fig, axs = plt.subplots(
        len(true_labels),
        2,
        figsize=(20, len(true_labels) * 10),
        subplot_kw={"projection": "3d"},
    )
    axs = axs.flatten()
    for event_idx, event_true_labels in enumerate(true_labels):
        event_reco_labels = reco_cluster_ids[event_idx]
        event_hit_coordinates = hit_coordinates[event_idx]
        unique_true_labels = np.unique(event_true_labels)
        unique_reco_labels = np.unique(event_reco_labels)

        # for evaluation check how many "wrong" cluster label hits in the given cluster
        cmap = plt.get_cmap("jet")
        distinct_reco_colors = cmap(np.linspace(0, 1, len(unique_reco_labels) - 1))
        distinct_true_colors = cmap(np.linspace(0, 1, len(unique_true_labels)))

        axs[event_idx * 2 + 1].set_box_aspect([1, 1, 1])
        axs[event_idx * 2 + 1].view_init(elev=20, azim=45)
        axs[event_idx * 2 + 1].scatter(
            event_hit_coordinates[:, 0],
            event_hit_coordinates[:, 1],
            event_hit_coordinates[:, 2],
            # s=np.clip(100*event_hit_coordinates[:, 3], 0.1, 10),
            c=distinct_true_colors[event_true_labels],
        )

        axs[event_idx * 2].set_box_aspect([1, 1, 1])
        axs[event_idx * 2].view_init(elev=20, azim=45)
        axs[event_idx * 2].scatter(
            event_hit_coordinates[:, 0][event_reco_labels != -1],
            event_hit_coordinates[:, 1][event_reco_labels != -1],
            event_hit_coordinates[:, 2][event_reco_labels != -1],
            # s=np.clip(100*event_hit_coordinates[:, 3], 0.1, 10),  # Here should be the energy
            c=distinct_reco_colors[event_reco_labels][event_reco_labels != -1],
        )
        axs[event_idx * 2].scatter(
            event_hit_coordinates[:, 0][event_reco_labels == -1],
            event_hit_coordinates[:, 1][event_reco_labels == -1],
            event_hit_coordinates[:, 2][event_reco_labels == -1],
            # s=np.clip(100*event_hit_coordinates[:, 3], 0.1, 10),  # Here should be the energy
            c="r",
        )
    fig.savefig("event_clusters.png")


def evaluate_training(test_loader, model):
    print("Starting evaluation")
    all_outputs = []
    truth = []
    batch_indices = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):  # or test_loader
            inputs = batch["features"]
            truth.append(batch["instance_labels"].F)
            bidx = batch["coordinates"].C[:, 3]

            if i == 0:
                hit_coordinates, reco_cluster_ids, true_labels = split_batch_to_events(
                    batch, model
                )
                visualize_batch_prediction_and_truth(
                    hit_coordinates, reco_cluster_ids, true_labels
                )

            if i == 1:
                break

            outputs = model(inputs)
            all_outputs.append(outputs)
            batch_indices.append(bidx)


@hydra.main(config_path="../../train_configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model = instantiate(cfg.model.target, cfg)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("Model instantiated and state dict loaded")

    datamodule = instantiate(cfg.dataset)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    print("Datamodule instantiated and test_loader set up.")

    evaluate_training(test_loader, model)


if __name__ == "__main__":
    main()
