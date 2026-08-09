"""MNIST dataset loading and federated partitioning across clients.

This module is the data layer for all three phases of the project. It does two
jobs:

1. Downloads MNIST (once) via torchvision and yields PyTorch DataLoaders for
   training and evaluation.
2. Splits the training set across N simulated clients to mimic a federated
   setting, supporting both IID and non-IID partitions.

Why this separation matters
---------------------------
In real federated learning, each device sees only its own slice of data and
the server never gathers raw data centrally. We *simulate* that on a single
machine by physically partitioning the dataset and giving each client only
its own subset. The model code never knows whether it is seeing the full
dataset or a single client's slice — the partition is enforced at the
DataLoader level.

McMahan et al. (2017) — the FedAvg paper — distinguish between IID and
non-IID partitions. IID is a sanity-check baseline; non-IID is the realistic
setting where each client has a skewed label distribution (e.g. one user's
phone keyboard mostly produces digits they personally type often). We
support both so we can compare and discuss this trade-off in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# We keep MNIST in a local `data/` directory at the project root. torchvision
# will download it the first time and reuse the cache afterwards.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# MNIST is grayscale 28x28. The standard normalization values (mean=0.1307,
# std=0.3081) come from the MNIST training set statistics. Using them here
# makes our results comparable to the broader literature, which uses the same
# constants.
_MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),                          # PIL -> [0,1] tensor
        transforms.Normalize((0.1307,), (0.3081,)),     # standardize
    ]
)


@dataclass
class FederatedSplit:
    """A federated partition of MNIST into client-local DataLoaders.

    Attributes
    ----------
    client_loaders : list of DataLoader
        client_loaders[i] is the training DataLoader for client i.
    test_loader : DataLoader
        Shared test set used by the server to evaluate the global model.
    num_clients : int
        Number of simulated clients.
    iid : bool
        Whether the split was IID (True) or non-IID (False).
    """

    client_loaders: List[DataLoader]
    test_loader: DataLoader
    num_clients: int
    iid: bool


def load_mnist_federated(
    num_clients: int = 5,
    batch_size: int = 32,
    iid: bool = True,
    seed: int = 42,
) -> FederatedSplit:
    """Load MNIST and split the training set across `num_clients` clients.

    Parameters
    ----------
    num_clients : int
        How many simulated clients to create. Proposal uses 5 for Phase 1,
        and we will scale up to {2, 5, 10, 20, 50} in Phase 3 overhead tests.
    batch_size : int
        Mini-batch size for client-local SGD. Following McMahan et al.,
        small batches (10–32) work well on MNIST in the federated setting.
    iid : bool
        If True, shuffle then split evenly — every client sees all 10 digits.
        If False, sort by label first then split — each client sees only
        a couple of digit classes (the harder, more realistic case).
    seed : int
        RNG seed for reproducibility. We fix it so re-running yields the
        same partition; this matters when we compare Phase 1 vs Phase 3
        runs and want to attribute differences only to the protocol, not
        to random data shuffling.

    Returns
    -------
    FederatedSplit
        Container with one DataLoader per client and a shared test loader.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download (if needed) and load MNIST. `download=True` is idempotent —
    # subsequent runs find the cached files and skip the download.
    train_set = datasets.MNIST(
        root=str(DATA_DIR), train=True,
        download=True, transform=_MNIST_TRANSFORM,
    )
    test_set = datasets.MNIST(
        root=str(DATA_DIR), train=False,
        download=True, transform=_MNIST_TRANSFORM,
    )

    # MNIST training set has 60,000 examples. We compute the indices belonging
    # to each client according to the chosen partitioning strategy.
    rng = np.random.default_rng(seed)
    indices = _partition_indices(
        labels=np.array(train_set.targets),
        num_clients=num_clients,
        iid=iid,
        rng=rng,
    )

    # Wrap each client's index list in a Subset, then in a DataLoader.
    # We shuffle inside each client (shuffle=True) so SGD sees a different
    # ordering each epoch, but the *partition* itself is fixed per `seed`.
    client_loaders = [
        DataLoader(
            Subset(train_set, idx_list),
            batch_size=batch_size,
            shuffle=True,
            # num_workers=0 keeps the simulation single-process which is
            # easier to debug. On a real cluster you would tune this.
            num_workers=0,
        )
        for idx_list in indices
    ]

    # Test loader is shared — server uses it to evaluate the global model.
    # No shuffle on test: deterministic evaluation.
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=0,
    )

    return FederatedSplit(
        client_loaders=client_loaders,
        test_loader=test_loader,
        num_clients=num_clients,
        iid=iid,
    )


def _partition_indices(
    labels: np.ndarray,
    num_clients: int,
    iid: bool,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Compute per-client index lists for the training set.

    IID strategy:
      Shuffle all indices, then split into `num_clients` equal slices.
      Each slice contains a roughly uniform mixture of all 10 digit classes.

    Non-IID strategy (following McMahan et al., 2017, Section 5):
      Sort indices by label, then divide into 2 * num_clients "shards".
      Assign 2 random shards to each client. Result: each client sees only
      ~2 distinct digit classes — a much harder setting that exposes the
      heterogeneity FedAvg is supposed to handle.
    """
    n = len(labels)

    if iid:
        # Shuffle and split evenly. Some clients may get one extra sample
        # if n is not divisible by num_clients; np.array_split handles that.
        all_indices = rng.permutation(n)
        return list(np.array_split(all_indices, num_clients))

    # ---- Non-IID partition (McMahan-style shard assignment) ----
    # Sort by label so that adjacent indices share the same class.
    sorted_indices = np.argsort(labels, kind="stable")

    # Cut into 2 * num_clients shards. With 60000 samples and 5 clients that
    # is 10 shards of 6000 each — each shard is mostly one or two digits.
    num_shards = 2 * num_clients
    shards = np.array_split(sorted_indices, num_shards)

    # Randomly assign 2 shards to each client without replacement.
    shard_ids = rng.permutation(num_shards)
    client_indices = []
    for c in range(num_clients):
        my_shards = shard_ids[2 * c : 2 * c + 2]
        merged = np.concatenate([shards[s] for s in my_shards])
        # Shuffle within the client so batches are not label-sorted internally.
        rng.shuffle(merged)
        client_indices.append(merged)

    return client_indices


def describe_split(split: FederatedSplit) -> str:
    """Return a human-readable summary of a federated partition.

    Useful in notebooks to verify that the partition is what we think it is —
    especially in non-IID mode, where we want to *see* the label skew.
    """
    lines = [
        f"Federated split: {split.num_clients} clients, "
        f"{'IID' if split.iid else 'non-IID'}",
    ]
    for i, loader in enumerate(split.client_loaders):
        # The Subset wraps the underlying dataset and stores the chosen
        # indices in `.indices`. We pull labels via the parent dataset.
        subset: Subset = loader.dataset  # type: ignore[assignment]
        labels = np.array(subset.dataset.targets)[subset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        dist = ", ".join(f"{int(u)}:{int(c)}" for u, c in zip(unique, counts))
        lines.append(f"  client {i:2d}: n={len(labels):5d}  labels {{{dist}}}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test — run `python -m src.data` from the project root.
    split = load_mnist_federated(num_clients=5, iid=True)
    print(describe_split(split))
    print()
    split_niid = load_mnist_federated(num_clients=5, iid=False)
    print(describe_split(split_niid))
