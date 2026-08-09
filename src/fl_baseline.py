"""FedAvg client and server logic — Phase 1 of the project.

What this implements
--------------------
The canonical FedAvg algorithm from McMahan et al. (2017, Algorithm 1):

    For each round t = 1..T:
        Server broadcasts current global weights w(t) to all clients.
        Each client i:
            Initializes local weights w_i <- w(t).
            Runs E local epochs of mini-batch SGD on its private data.
            Sends the *update* delta_i = w_i - w(t) back to the server.
        Server averages updates:
            w(t+1) = w(t) + (1/N) * sum_i delta_i

We send weight *deltas* (not raw weights and not gradients) because that
is the form Bonawitz et al. mask in Phase 3 — a vector summed across
clients. Sending deltas instead of full weights also halves what an
attacker might exploit; the DLG attack still works on a single round's
delta because, with E=1 local epoch, delta is just (-eta) times the
gradient.

Why not use Flower?
-------------------
The proposal mentions Flower, but for this project I deliberately wrote
the FL loop by hand. Reasons:

1. Phase 2 needs to capture the per-client update vector at the server.
   Flower's strategy abstraction makes that hook awkward.
2. Phase 3 replaces the aggregation step with a custom protocol.
   Plugging Bonawitz's protocol into Flower would obscure exactly what
   I am modifying.
3. Pedagogical clarity: the FedAvg loop is ~40 lines. Reading it is
   easier than reading Flower's class hierarchy.

I document this choice in the report and acknowledge that Flower would
be the right tool for a production system or large-scale experiment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# A "state dict" in PyTorch is an OrderedDict mapping parameter names to
# tensors. We pass these around to represent model snapshots.
StateDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Client-side: local training
# ---------------------------------------------------------------------------

@dataclass
class ClientUpdate:
    """One client's contribution to a federated round.

    Attributes
    ----------
    delta : StateDict
        Per-parameter difference (local_weights - global_weights). This is
        what we aggregate. Phase 3 will mask this delta before sending it.
    num_samples : int
        How many training examples the client used. Lets the server compute
        a sample-weighted average if datasets are unequal.
    train_loss : float
        Mean local training loss in this round (for monitoring).
    """

    delta: StateDict
    num_samples: int
    train_loss: float


def client_update(
    global_state: StateDict,
    loader: DataLoader,
    *,
    local_epochs: int = 1,
    lr: float = 0.05,
    device: torch.device | str = "cpu",
    model_factory=None,
) -> ClientUpdate:
    """Run local SGD on one client and return the resulting weight delta.

    Parameters
    ----------
    global_state : StateDict
        The server's current model weights, as a state dict. The client
        starts training from these weights.
    loader : DataLoader
        This client's local training data (a Subset of MNIST in our setup).
    local_epochs : int
        How many full passes over the local dataset to perform per round.
        McMahan et al. show that increasing E reduces required rounds but
        can hurt convergence under non-IID data. We default to 1 epoch.
    lr : float
        Local SGD learning rate.
    device : torch.device | str
        "cpu" or "mps" for Apple Silicon, "cuda" for NVIDIA. Default cpu
        so that the same code runs on the lab and on a laptop.
    model_factory : Callable | None
        A zero-arg callable returning a fresh model instance. We need a
        factory rather than a model instance because we instantiate the
        model locally on every round to ensure no state leaks across calls.

    Returns
    -------
    ClientUpdate
        Carrying the per-parameter delta and bookkeeping metadata.
    """
    if model_factory is None:
        # Late import to avoid a circular dependency at module load time.
        from .model import MNIST_CNN
        model_factory = MNIST_CNN

    # 1. Build a fresh model and load the global weights.
    model = model_factory().to(device)
    model.load_state_dict(global_state)

    # 2. Set up local optimizer and loss. Plain SGD without momentum is the
    #    standard choice for FedAvg — extra optimizer state would also need
    #    to be communicated, which complicates the system.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 3. Snapshot the initial weights so we can compute the delta later.
    #    We deep-clone tensors detached from any graph; otherwise the
    #    subtraction at the end would try to backprop through stale ops.
    initial_state = {k: v.detach().clone() for k, v in global_state.items()}

    # 4. Run local epochs of mini-batch SGD.
    model.train()
    total_loss, total_samples = 0.0, 0
    for _ in range(local_epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    # 5. Compute delta = local_weights - initial_weights.
    #    Note: `local_weights` here means the weights *after* training, which
    #    we read out from the model's state dict. We .clone() to detach from
    #    autograd and ensure the returned tensors are stable (no aliasing).
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    delta = {k: final_state[k] - initial_state[k] for k in final_state}

    return ClientUpdate(
        delta=delta,
        num_samples=total_samples // max(1, local_epochs),
        train_loss=total_loss / max(1, total_samples),
    )


# ---------------------------------------------------------------------------
# Server-side: aggregation
# ---------------------------------------------------------------------------

def federated_average(
    global_state: StateDict,
    updates: List[ClientUpdate],
    *,
    weighted: bool = True,
) -> StateDict:
    """Aggregate client updates into a new global state (the FedAvg step).

    Parameters
    ----------
    global_state : StateDict
        The current global weights. We add the averaged delta to this.
    updates : list of ClientUpdate
        Updates collected from all participating clients in this round.
    weighted : bool
        If True, weight each delta by the client's sample count. Otherwise
        use a uniform mean (each client contributes 1/N regardless of size).
        For our IID equal-size partition the two are equivalent. Non-IID
        partitions can have unequal sizes — weighted average is the standard
        fix.

    Returns
    -------
    StateDict
        Updated global weights. PyTorch state dicts are dicts of tensors,
        so we just return a new dict and the caller decides what to do with
        it (typically: load it into the model for the next round).

    Phase 3 connection
    ------------------
    This function is the "trusted aggregator". The whole point of Bonawitz's
    Secure Aggregation protocol is to compute *exactly* this sum without
    the server seeing any individual `update.delta`. In Phase 3 we will
    swap this function out for the secure variant; the rest of the FL
    loop will not need to change.
    """
    if not updates:
        raise ValueError("No client updates to aggregate.")

    # Compute per-client weights for the weighted average.
    if weighted:
        total = sum(u.num_samples for u in updates)
        weights = [u.num_samples / total for u in updates]
    else:
        weights = [1.0 / len(updates)] * len(updates)

    # Weighted sum of deltas, parameter by parameter.
    new_state: StateDict = {}
    for key in global_state.keys():
        # Initialize accumulator with zeros of matching shape and dtype.
        agg = torch.zeros_like(global_state[key])
        for w, u in zip(weights, updates):
            agg = agg + w * u.delta[key]
        new_state[key] = global_state[key] + agg

    return new_state


# ---------------------------------------------------------------------------
# Server-side: evaluation
# ---------------------------------------------------------------------------

def evaluate(
    state: StateDict,
    loader: DataLoader,
    *,
    device: torch.device | str = "cpu",
    model_factory=None,
) -> Tuple[float, float]:
    """Evaluate a model snapshot on a held-out loader.

    Returns
    -------
    (loss, accuracy) : Tuple[float, float]
        Mean cross-entropy loss and top-1 accuracy in [0, 1].
    """
    if model_factory is None:
        from .model import MNIST_CNN
        model_factory = MNIST_CNN

    model = model_factory().to(device)
    model.load_state_dict(state)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, total_correct / total


# ---------------------------------------------------------------------------
# Top-level federated training loop
# ---------------------------------------------------------------------------

def run_federated_training(
    client_loaders: List[DataLoader],
    test_loader: DataLoader,
    *,
    num_rounds: int = 30,
    local_epochs: int = 1,
    lr: float = 0.05,
    device: torch.device | str = "cpu",
    seed: int = 42,
    log_every: int = 1,
    model_factory=None,
) -> Tuple[StateDict, List[Dict]]:
    """Drive a complete vanilla FedAvg run.

    This is the function the Phase 1 notebook calls. It returns the final
    global state plus a per-round log so we can plot accuracy curves.

    Parameters
    ----------
    client_loaders : list of DataLoader
        One per client. We assume all clients participate every round
        (C = 1.0 in McMahan's notation). For 5 clients this is the standard
        toy setting and matches the proposal.
    test_loader : DataLoader
        Server-side held-out evaluation set.
    num_rounds : int
        Number of communication rounds. 30 is enough for the small CNN to
        plateau on MNIST; we'll see this in the convergence plot.
    local_epochs : int
        Local epochs per round (E in McMahan's notation).
    lr : float
        Client SGD learning rate.
    device : str | torch.device
        "cpu", "mps" (Apple Silicon), or "cuda".
    seed : int
        For reproducible weight initialization.
    log_every : int
        Evaluate and log every K rounds. 1 = log every round.

    Returns
    -------
    (final_state, history) : (StateDict, list of dict)
        final_state : trained global weights
        history    : list of {"round", "test_loss", "test_acc", "train_loss"}
    """
    if model_factory is None:
        from .model import MNIST_CNN
        model_factory = MNIST_CNN

    # Reproducible model initialization. PyTorch's default init is random;
    # fixing the seed makes runs comparable across phases.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize global weights from a fresh model.
    init_model = model_factory().to(device)
    global_state: StateDict = {
        k: v.detach().clone() for k, v in init_model.state_dict().items()
    }

    history: List[Dict] = []

    for rnd in range(1, num_rounds + 1):
        # ---- Each client trains locally ----
        updates: List[ClientUpdate] = []
        for loader in client_loaders:
            up = client_update(
                global_state=global_state,
                loader=loader,
                local_epochs=local_epochs,
                lr=lr,
                device=device,
                model_factory=model_factory,
            )
            updates.append(up)

        # ---- Server aggregates ----
        global_state = federated_average(global_state, updates, weighted=True)

        # ---- Optionally evaluate ----
        if rnd % log_every == 0 or rnd == num_rounds:
            test_loss, test_acc = evaluate(
                global_state, test_loader,
                device=device, model_factory=model_factory,
            )
            mean_train = float(np.mean([u.train_loss for u in updates]))
            history.append({
                "round": rnd,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "train_loss": mean_train,
            })
            print(
                f"[round {rnd:3d}] "
                f"train_loss={mean_train:.4f}  "
                f"test_loss={test_loss:.4f}  "
                f"test_acc={test_acc:.4f}"
            )

    return global_state, history
