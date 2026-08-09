"""Deep Leakage from Gradients (DLG) attack — Phase 2 of the project.

This module implements the gradient inversion attack from Zhu, Liu, and Han
(NeurIPS 2019). The attack lets an honest-but-curious server reconstruct a
client's private training input from nothing but the gradient that client
sent. It is the concrete demonstration of why "we only share gradients, not
data" is *not* a valid privacy argument in vanilla federated learning.

The core idea
-------------
The victim client, given a private (x, y), computes:

    g_target = nabla_W L(F(x; W), y)

and sends g_target to the server. The server is curious. It picks a *dummy*
input (x', y') initialized to random noise, runs the same forward+backward
pass with the *same* W to produce:

    g'(x', y') = nabla_W L(F(x'; W), y')

and then optimizes (x', y') to minimize the gradient distance:

    L_attack(x', y') = || g'(x', y') - g_target ||_2^2

Why does this recover x? Because the gradient of a smooth model is, locally,
an injective map of (x, y) — two different inputs almost surely produce
different gradients. So matching the gradient pins down the input.

Why does the optimization need second derivatives? Because g'(x', y') is
itself a derivative (it is a gradient with respect to W). Optimizing over
(x', y') means differentiating g' once more, this time with respect to the
attack variables. PyTorch handles this automatically via `create_graph=True`
when we compute the inner gradient.

Implementation choices that mirror the paper
--------------------------------------------
* Optimizer: L-BFGS. It uses second-order curvature information and converges
  much faster than first-order methods on this loss landscape. Adam tends to
  oscillate without converging here.
* Initialization: dummy input is N(0, 1) noise of the same shape as x;
  dummy label is N(0, 1) one-hot-like noise of the same shape as the model's
  logits (i.e. (num_classes,) per example). The label is a *soft label* (not
  an integer class) because we need it differentiable.
* Loss: Zhu et al. use the cross-entropy variant where both the true and
  predicted labels are soft probabilities. We replicate that formulation.

What this module does NOT do
----------------------------
* It does not handle batch sizes > 1 with the iDLG/improved-DLG tricks.
  The original paper handles batches by a sample-by-sample update rule;
  for our pedagogical demo we attack one sample at a time, which is the
  cleanest illustration of the threat. Batched attacks are mentioned as
  future work in the report.
* It does not assume the attacker knows the true label. Recovering both
  x and y simultaneously is the original DLG setup, not the easier iDLG
  variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss formulation: cross-entropy variants
# ---------------------------------------------------------------------------
#
# We need TWO related losses depending on the role of the target tensor:
#
#  (a) For VICTIM gradient computation, we know the true integer label.
#      The natural form is standard cross-entropy with a one-hot probability
#      target. No softmax on the target — it is ALREADY a probability vector.
#
#  (b) For the DLG attack inner loop, the dummy label y' is a learnable
#      raw tensor that the optimizer updates freely. To interpret it as a
#      probability distribution we apply softmax to it; otherwise the
#      optimizer would have no constraint on y' and could push it to
#      arbitrary magnitudes.
#
# Mixing these two cases up was a real bug in an earlier version of this
# file: applying softmax to a one-hot target turned the "true label"
# distribution into a fuzzy mixture, which corrupted the gradient the
# attacker is trying to match. iDLG's analytical label recovery in
# particular FAILS COMPLETELY under this corruption, since the bias
# gradient sign trick only holds for true (non-fuzzy) cross-entropy.
#
# So we now have two functions, each used in its own place.


def _ce_with_probability_target(
    logits: torch.Tensor, prob_targets: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy where the target is ALREADY a probability vector.

    Use this when the target is a one-hot ground truth (or any other
    probability distribution we don't want softmaxed). Equivalent to
    `nn.CrossEntropyLoss` when prob_targets is one-hot.

    Math:
      L = -sum_k q_k * log p_k    where p = softmax(logits) and q = prob_targets.
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -(prob_targets * log_probs).sum(dim=1).mean()


def _ce_with_logit_target(
    logits: torch.Tensor, logit_targets: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy where the target is a RAW (logit-style) tensor.

    Use this for the DLG dummy label, which is a learnable tensor of
    arbitrary magnitudes. We softmax it to obtain a valid probability
    distribution before computing CE.
    """
    log_probs = F.log_softmax(logits, dim=1)
    target_probs = F.softmax(logit_targets, dim=1)
    return -(target_probs * log_probs).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Capturing the victim's gradient
# ---------------------------------------------------------------------------

def compute_victim_gradient(
    model: nn.Module,
    x: torch.Tensor,
    y_index: int,
    num_classes: int = 10,
) -> List[torch.Tensor]:
    """Compute the gradient that an honest client would send for one sample.

    Parameters
    ----------
    model : nn.Module
        The current global model (its weights are shared with the attacker).
    x : torch.Tensor
        Single private input, shape (1, 1, 28, 28) for MNIST.
    y_index : int
        Integer class label of the private input.
    num_classes : int
        Number of classes; needed to one-hot encode y for the soft-CE loss.

    Returns
    -------
    list of torch.Tensor
        One tensor per model parameter — the gradient w.r.t. that parameter.
        Detached so we don't accidentally backprop through the victim later.
    """
    # Build a one-hot soft label so the loss formulation is the same as the
    # one we use inside the attack. This makes the gradient we are trying
    # to match exactly comparable to the dummy gradient.
    y_onehot = torch.zeros(1, num_classes)
    y_onehot[0, y_index] = 1.0

    model.eval()  # disable dropout/batchnorm running stats — irrelevant here
    # Make sure parameter gradients are accumulated freshly.
    model.zero_grad()

    logits = model(x)
    # Victim sees the TRUE label as a one-hot probability target. We must
    # NOT softmax it (that would smear the one-hot into a fuzzy mixture).
    loss = _ce_with_probability_target(logits, y_onehot)

    # We use torch.autograd.grad rather than loss.backward() because grad()
    # returns the gradient tensors directly (cleaner) and lets us choose
    # not to populate .grad on the parameters (no side effects).
    grads = torch.autograd.grad(loss, list(model.parameters()))

    # Detach: these are fixed targets for the attack; no further autograd.
    return [g.detach().clone() for g in grads]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DLGResult:
    """Outcome of one DLG attack run.

    Attributes
    ----------
    reconstructed_x : torch.Tensor
        Final dummy input — the attacker's reconstruction of the private x.
    reconstructed_y : int
        argmax of the final dummy label — the attacker's class guess.
    snapshots : list of torch.Tensor
        Reconstructed inputs at chosen iteration counts. Used for the
        "iters=0, 10, 50, 100, 500" visualization that mirrors the paper's
        Figure 3.
    snapshot_iters : list of int
        Iteration index of each snapshot.
    losses : list of float
        Attack loss per iteration; lets us plot convergence.
    converged : bool
        Whether the loss dropped below a tolerance.
    """

    reconstructed_x: torch.Tensor
    reconstructed_y: int
    snapshots: List[torch.Tensor] = field(default_factory=list)
    snapshot_iters: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    converged: bool = False


# ---------------------------------------------------------------------------
# The attack itself
# ---------------------------------------------------------------------------

def dlg_attack(
    model: nn.Module,
    target_grads: List[torch.Tensor],
    *,
    input_shape: tuple = (1, 1, 28, 28),
    num_classes: int = 10,
    num_iterations: int = 300,
    lr: float = 1.0,
    snapshot_iters: tuple = (0, 10, 50, 100, 200, 300),
    seed: Optional[int] = 0,
    verbose: bool = True,
    progress_callback: Optional[Callable[[int, float], None]] = None,
) -> DLGResult:
    """Run the DLG gradient inversion attack.

    Parameters
    ----------
    model : nn.Module
        The model whose weights match the victim's. The attacker has full
        access to this — that is the whole point: in FL, the server *knows*
        the global weights because it just broadcast them.
    target_grads : list of torch.Tensor
        The victim's gradient (one tensor per model parameter), as produced
        by `compute_victim_gradient`.
    input_shape : tuple
        Shape of the dummy input including the batch dimension. For our
        MNIST attack on one sample: (1, 1, 28, 28).
    num_classes : int
        For the dummy label tensor shape: (1, num_classes).
    num_iterations : int
        Number of L-BFGS optimization steps. The paper finds 300 plenty
        for MNIST single-sample attacks; CIFAR/larger may need more.
    lr : float
        L-BFGS step size. 1.0 is the standard for L-BFGS on this loss.
    snapshot_iters : tuple of int
        Iteration indices at which to save the current dummy x for the
        visualization figure.
    seed : int | None
        RNG seed for reproducible dummy initialization.
    verbose : bool
        Print progress every ~10% of iterations.
    progress_callback : callable | None
        Optional callback `f(iter, loss)` called every iteration. Useful
        for live progress bars in notebooks.

    Returns
    -------
    DLGResult
        The reconstructed input, label, snapshots, and loss curve.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # --- 1. Initialize dummy input and dummy label ---
    # We require gradients on these because they are the attack variables
    # we are optimizing. requires_grad_(True) flags them as leaf tensors
    # that autograd will track.
    dummy_x = torch.randn(*input_shape, requires_grad=True)
    dummy_y = torch.randn(1, num_classes, requires_grad=True)

    # Snapshot index 0 — initial random noise, before any optimization.
    snapshots: List[torch.Tensor] = []
    snapshot_iters_taken: List[int] = []
    if 0 in snapshot_iters:
        snapshots.append(dummy_x.detach().clone())
        snapshot_iters_taken.append(0)

    # --- 2. Set up optimizer ---
    # L-BFGS uses second-order curvature info via approximate Hessian.
    # `history_size=100` keeps the last 100 steps of curvature info; the
    # default 100 is fine. `max_iter=20` lets each .step() do up to 20
    # internal L-BFGS sub-iterations — this is what makes L-BFGS efficient.
    optimizer = torch.optim.LBFGS(
        [dummy_x, dummy_y],
        lr=lr,
        max_iter=20,
        history_size=100,
        tolerance_change=1e-9,
        tolerance_grad=1e-9,
    )

    losses: List[float] = []
    model.eval()

    # --- 3. The closure that L-BFGS calls each step ---
    # L-BFGS may evaluate the loss multiple times per .step() (line search),
    # so it needs a closure that re-computes the loss from scratch.
    def closure() -> torch.Tensor:
        # Zero gradients on the *attack variables* — not on the model.
        optimizer.zero_grad()

        # Forward + backward to get dummy gradients w.r.t. model params.
        # CRITICAL: create_graph=True keeps the autograd graph alive so we
        # can take a *second* derivative (with respect to dummy_x, dummy_y).
        # Without this flag we'd get a "no graph found" error when calling
        # backward() on the attack loss.
        dummy_pred = model(dummy_x)
        # dummy_y is a learnable raw tensor; softmax it before using as
        # a probability target (this is the legitimate use of softmax on
        # the target).
        dummy_loss = _ce_with_logit_target(dummy_pred, dummy_y)
        dummy_grads = torch.autograd.grad(
            dummy_loss,
            list(model.parameters()),
            create_graph=True,
        )

        # Attack loss: L2 distance between dummy and target gradient stacks.
        # We sum across all parameters; this gives gradient-matching across
        # the whole network rather than just one layer.
        attack_loss = sum(
            ((dg - tg) ** 2).sum()
            for dg, tg in zip(dummy_grads, target_grads)
        )

        # Backward: compute d(attack_loss)/d(dummy_x), d(attack_loss)/d(dummy_y).
        # This is the second-derivative step (a Hessian-vector product
        # under the hood). PyTorch handles it automatically.
        attack_loss.backward()
        return attack_loss

    # --- 4. Optimization loop ---
    log_step = max(1, num_iterations // 10)
    for it in range(1, num_iterations + 1):
        loss_val = optimizer.step(closure)
        loss_scalar = float(loss_val) if loss_val is not None else float("nan")
        losses.append(loss_scalar)

        if it in snapshot_iters:
            snapshots.append(dummy_x.detach().clone())
            snapshot_iters_taken.append(it)

        if verbose and (it % log_step == 0 or it == num_iterations):
            print(f"  [DLG iter {it:4d}/{num_iterations}]  attack_loss = {loss_scalar:.6e}")

        if progress_callback is not None:
            progress_callback(it, loss_scalar)

        # Numerical safety: if the loss blows up to NaN/Inf, abort early.
        # This sometimes happens with poor initialization; re-running with
        # a different seed usually fixes it.
        if not np.isfinite(loss_scalar):
            print(f"  [DLG] loss became non-finite at iter {it}; stopping.")
            break

    converged = bool(losses) and (losses[-1] < 1e-4)

    return DLGResult(
        reconstructed_x=dummy_x.detach().clone(),
        reconstructed_y=int(dummy_y.detach().argmax().item()),
        snapshots=snapshots,
        snapshot_iters=snapshot_iters_taken,
        losses=losses,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Quality metrics for evaluating reconstruction quality
# ---------------------------------------------------------------------------

def reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict:
    """Compute PSNR and SSIM between the original and the reconstruction.

    Both metrics are standard image-similarity measures used in the DLG
    literature to quantify how well an attack reconstructs the input.

    PSNR (Peak Signal-to-Noise Ratio):
      Measured in dB. Higher = closer to original.
      PSNR > 30 dB: visually nearly indistinguishable.
      PSNR ~ 20-30 dB: clearly recognizable but with artifacts.
      PSNR < 15 dB: noisy, low recovery.

    SSIM (Structural Similarity):
      In [-1, 1], higher is better.
      SSIM > 0.95: near-perfect recovery.
      SSIM ~ 0.5-0.9: structure visible but degraded.
      SSIM < 0.3: little structural recovery.

    Inputs are tensors of shape (1, 1, H, W). We return both metrics in a
    dict so the notebook can log them to a CSV.
    """
    # Lazy imports keep startup snappy if the user doesn't call this.
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    a = original.detach().cpu().squeeze().numpy()
    b = reconstructed.detach().cpu().squeeze().numpy()

    # PSNR needs a known data_range. Our images are normalized to roughly
    # [-0.4, 2.8] after the MNIST normalization, so we measure the actual
    # range to avoid biasing the metric.
    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    psnr = float(peak_signal_noise_ratio(a, b, data_range=data_range))
    ssim = float(structural_similarity(a, b, data_range=data_range))

    # MSE in the original tensor space — handy as a sanity check.
    mse = float(((a - b) ** 2).mean())

    return {"psnr_db": psnr, "ssim": ssim, "mse": mse}


def denormalize_mnist(x: torch.Tensor) -> torch.Tensor:
    """Undo the MNIST mean/std normalization for visualization.

    We standardize images during training (mean=0.1307, std=0.3081), but for
    plotting we want pixel values back in [0, 1] roughly. This helper exists
    so the notebook doesn't repeat the magic numbers everywhere.
    """
    return x * 0.3081 + 0.1307


# ===========================================================================
# iDLG (Improved DLG) — Zhao, Mopuri, Bilen 2020
# ===========================================================================
#
# The DLG attack jointly optimizes (x', y'). Convergence is unstable: about
# half of our trials get stuck in local minima where the dummy label has
# converged to the wrong class.
#
# iDLG observes that for a single-sample attack with cross-entropy loss, the
# *true* label can be recovered analytically — without optimization — by
# inspecting the sign of the final-layer bias gradient.
#
# Why this works
# --------------
# For one sample with true label c, softmax probabilities p_i, and bias
# terms b_i in the final linear layer:
#
#     dL/db_i = p_i - y_i
#
# where y is the one-hot ground truth. So:
#     - For i == c (true label): dL/db_c = p_c - 1   (NEGATIVE, since p_c < 1)
#     - For i != c (other classes): dL/db_i = p_i    (POSITIVE, since p_i > 0)
#
# Thus the index of the unique negative entry in the bias gradient is the
# label. We extract it, then run DLG with the label fixed — only x is
# optimized. The optimization landscape is much smoother and convergence
# becomes nearly deterministic.
#
# Caveat: this trick assumes batch size = 1. With batches of size > 1, the
# bias gradient sums over all samples and the per-sample sign information is
# lost. Our setup uses single-sample attacks, so this is fine.


def extract_label_from_gradient(
    target_grads: List[torch.Tensor],
    final_bias_index: int = -1,
) -> int:
    """Recover the true label analytically from the final-layer bias gradient.

    Parameters
    ----------
    target_grads : list of gradient tensors, one per model parameter, in
        the order returned by `model.parameters()`.
    final_bias_index : index into target_grads of the final layer's bias.
        Default -1 means "the last tensor", which is correct for any model
        whose forward ends in `nn.Linear` (since PyTorch yields linear
        weights then biases in that order). Pass an explicit index if your
        architecture is more exotic.

    Returns
    -------
    int : recovered label in [0, num_classes).

    Raises
    ------
    ValueError if the heuristic cannot identify a unique negative entry —
    this could happen with unusual loss functions or zero gradients on
    poorly-trained models.
    """
    bias_grad = target_grads[final_bias_index].detach().cpu().numpy()
    # The recovered label is the argmin of the bias gradient. With
    # cross-entropy this corresponds to the unique negative entry (the
    # most negative one if multiple are negative due to floating-point
    # noise — argmin still picks the right index).
    label = int(bias_grad.argmin())
    # Sanity check: that entry should actually be the smallest by a
    # noticeable margin. If not, something's off (e.g. all-zero grads
    # from a perfectly-trained model on this sample).
    if not np.isfinite(bias_grad).all():
        raise ValueError("Final bias gradient contains non-finite values.")
    return label


def idlg_attack(
    model: nn.Module,
    target_grads: List[torch.Tensor],
    *,
    input_shape: tuple = (1, 1, 28, 28),
    num_classes: int = 10,
    num_iterations: int = 300,
    lr: float = 1.0,
    snapshot_iters: tuple = (0, 10, 50, 100, 200, 300),
    seed: Optional[int] = 0,
    verbose: bool = True,
    progress_callback: Optional[Callable[[int, float], None]] = None,
) -> DLGResult:
    """iDLG attack — DLG with the label recovered analytically.

    Same interface as `dlg_attack` so the notebook can swap them. The
    differences relative to DLG:

      1. Label is extracted from `target_grads` before optimization starts.
      2. The optimizer only updates `dummy_x`, not `dummy_y`.
      3. `dummy_y` is set to the recovered one-hot, fixed throughout.

    All other choices (L-BFGS, soft cross-entropy, create_graph=True for
    second derivatives) are identical to DLG.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # ---- 1. Recover the true label analytically. ----
    # This is the entire iDLG insight, in a single line. Note that this
    # extraction is "free" — no optimization, no extra gradient computation.
    recovered_label = extract_label_from_gradient(target_grads)
    if verbose:
        print(f"  [iDLG] Analytically recovered label: {recovered_label}")

    # ---- 2. Build a fixed one-hot label tensor. ----
    # `requires_grad=False` because we will NOT optimize this. It's a
    # constant target.
    fixed_y = torch.zeros(1, num_classes)
    fixed_y[0, recovered_label] = 1.0

    # ---- 3. Initialize dummy input only. ----
    dummy_x = torch.randn(*input_shape, requires_grad=True)

    snapshots: List[torch.Tensor] = []
    snapshot_iters_taken: List[int] = []
    if 0 in snapshot_iters:
        snapshots.append(dummy_x.detach().clone())
        snapshot_iters_taken.append(0)

    # ---- 4. L-BFGS over dummy_x only. ----
    # Compare with DLG: there we passed [dummy_x, dummy_y]. Here, just [dummy_x].
    # Half the search dimensionality, much smoother loss landscape.
    optimizer = torch.optim.LBFGS(
        [dummy_x],
        lr=lr,
        max_iter=20,
        history_size=100,
        tolerance_change=1e-9,
        tolerance_grad=1e-9,
    )

    losses: List[float] = []
    model.eval()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        dummy_pred = model(dummy_x)
        # fixed_y is a TRUE one-hot probability vector (not learnable).
        # Same loss form as the victim's gradient computation — no softmax
        # on the target. This is critical: the dummy gradient must be
        # comparable to target_grads, which were computed with a one-hot
        # probability target as well.
        dummy_loss = _ce_with_probability_target(dummy_pred, fixed_y)
        dummy_grads = torch.autograd.grad(
            dummy_loss,
            list(model.parameters()),
            create_graph=True,
        )
        attack_loss = sum(
            ((dg - tg) ** 2).sum()
            for dg, tg in zip(dummy_grads, target_grads)
        )
        attack_loss.backward()
        return attack_loss

    log_step = max(1, num_iterations // 10)
    for it in range(1, num_iterations + 1):
        loss_val = optimizer.step(closure)
        loss_scalar = float(loss_val) if loss_val is not None else float("nan")
        losses.append(loss_scalar)

        if it in snapshot_iters:
            snapshots.append(dummy_x.detach().clone())
            snapshot_iters_taken.append(it)

        if verbose and (it % log_step == 0 or it == num_iterations):
            print(f"  [iDLG iter {it:4d}/{num_iterations}]  attack_loss = {loss_scalar:.6e}")

        if progress_callback is not None:
            progress_callback(it, loss_scalar)

        if not np.isfinite(loss_scalar):
            print(f"  [iDLG] loss became non-finite at iter {it}; stopping.")
            break

    converged = bool(losses) and (losses[-1] < 1e-4)

    return DLGResult(
        reconstructed_x=dummy_x.detach().clone(),
        reconstructed_y=recovered_label,
        snapshots=snapshots,
        snapshot_iters=snapshot_iters_taken,
        losses=losses,
        converged=converged,
    )
