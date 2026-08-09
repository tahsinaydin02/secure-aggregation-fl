"""Glue layer integrating Secure Aggregation into the FedAvg loop.

Why this module exists
----------------------
The Bonawitz protocol in protocol.py operates on integer vectors. PyTorch
gradients/deltas are float32 tensors arranged across many parameters. We
need to:

    1. Flatten each client's delta dict into one float vector.
    2. Quantize the floats to int32 (the fixed-point step the paper calls
       "encoding into Z_R").
    3. Sum-aggregate across clients via Secure Aggregation.
    4. Dequantize back to floats.
    5. Reshape back into per-parameter tensors.

Quantization
------------
Floats need to be encoded into a finite-modulus ring (we use Z_{2^32}) for
the mask arithmetic to work. We use a simple symmetric fixed-point encoding:

    encode(x) = round(x * SCALE)  in int32 representation
    decode(y) = y / SCALE / N      (divide by N to recover the *mean*)

SCALE = 2^16 gives ~5-decimal-digit precision — far finer than the
quantization error of typical float32 gradients. This matches the
"linearly-quantize" step used in real Secure Aggregation deployments
(e.g. Bonawitz et al. 2019, Section 4.2 of the systems paper).

Two's-complement interpretation
-------------------------------
Negative gradient values are encoded into the upper half of [0, 2^32) via
two's complement. The protocol sums everything mod 2^32. As long as the
true sum stays in [-2^31, 2^31) — true for any realistic gradient sum
with N <= a few thousand clients — the wrap-around simulates signed
arithmetic exactly.

What this module exposes
------------------------
    secure_federated_average(global_state, updates)
        Drop-in replacement for fl_baseline.federated_average that uses
        the Bonawitz protocol under the hood. The Phase 3 notebooks
        compare the two in correctness, accuracy, and performance.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..fl_baseline import ClientUpdate, StateDict
from .protocol import MASK_MODULUS, run_secure_aggregation


# 2^16 = 65536 quantization levels per unit. Plenty of precision for
# gradients which usually live in [-1, 1].
QUANT_SCALE = 1 << 16


# ===========================================================================
# Tensor <-> int-vector encoding
# ===========================================================================

def _flatten_state(state: StateDict) -> Tuple[np.ndarray, List[Tuple[str, tuple]]]:
    """Flatten a state dict into a single 1-D float numpy array.

    Returns (flat_array, schema). The schema records each tensor's name
    and shape so we can rebuild the dict after aggregation.
    """
    schema: List[Tuple[str, tuple]] = []
    pieces: List[np.ndarray] = []
    for name, tensor in state.items():
        schema.append((name, tuple(tensor.shape)))
        pieces.append(tensor.detach().cpu().numpy().reshape(-1).astype(np.float64))
    flat = np.concatenate(pieces) if pieces else np.array([], dtype=np.float64)
    return flat, schema


def _unflatten_state(
    flat: np.ndarray,
    schema: List[Tuple[str, tuple]],
    reference_state: StateDict,
) -> StateDict:
    """Inverse of _flatten_state — rebuild the state dict.

    `reference_state` is used only to copy dtype/device info — its values
    are ignored.
    """
    out: StateDict = {}
    offset = 0
    for name, shape in schema:
        size = int(np.prod(shape)) if shape else 1
        chunk = flat[offset : offset + size].reshape(shape)
        ref = reference_state[name]
        out[name] = torch.tensor(chunk, dtype=ref.dtype, device=ref.device)
        offset += size
    return out


def encode_floats(x: np.ndarray) -> np.ndarray:
    """Float64 array -> int64 array, quantized and mapped into [0, 2^32).

    Negative values become large positive values via two's-complement.
    The protocol works modulo 2^32, so the wrap-around is automatic.
    """
    # 1. Quantize. round() is closest to nearest integer; np.round gives
    #    banker's rounding which is fine for our purposes.
    quantized = np.round(x * QUANT_SCALE).astype(np.int64)

    # 2. Reduce mod 2^32. Numpy's % with negative numbers returns positive
    #    residues, which is exactly the two's-complement wrap we want.
    return quantized % MASK_MODULUS


def decode_int_sum(y: np.ndarray, num_clients: int) -> np.ndarray:
    """Inverse of encode_floats, applied to the SUM of N encoded vectors.

    Two corrections:
      a) Map values in (2^31, 2^32) back to negative (signed reinterpret).
      b) Divide by SCALE * N to recover the mean float value.
    """
    # a) Signed reinterpret: anything >= 2^31 is actually negative.
    signed = np.where(y >= (1 << 31), y - MASK_MODULUS, y).astype(np.int64)
    # b) Recover means.
    return signed.astype(np.float64) / (QUANT_SCALE * num_clients)


# ===========================================================================
# The drop-in secure aggregation step for FedAvg
# ===========================================================================

def secure_federated_average(
    global_state: StateDict,
    updates: List[ClientUpdate],
    *,
    threshold: Optional[int] = None,
    dropouts_after_round1: Optional[List[int]] = None,
    return_timings: bool = False,
) -> StateDict:
    """Aggregate client deltas using the Bonawitz Secure Aggregation protocol.

    Drop-in replacement for fl_baseline.federated_average.

    Parameters
    ----------
    global_state : current global weights.
    updates : list of ClientUpdate, one per client.
    threshold : SSS threshold; defaults to floor(N/2)+1.
    dropouts_after_round1 : 1-indexed client IDs to simulate as dropouts
        (for dropout robustness experiments). The server learns the sum
        of the survivors' deltas and ignores the dropouts.
    return_timings : if True, returns (new_state, timings_dict) instead of
        just new_state. Used by the overhead-scaling notebook.

    Returns
    -------
    StateDict (or tuple of (StateDict, dict)).

    Note on accuracy
    ----------------
    The output is mathematically equivalent to a UNIFORM mean over the
    surviving clients (each contributes 1/N_surv). It does NOT support
    sample-weighted averaging because the masks must cancel exactly,
    which requires every client's contribution to enter the sum with the
    same coefficient. For our IID equal-size partition this matches what
    federated_average(..., weighted=True) computes anyway. For non-IID
    partitions with unequal client sizes, the report should note this as
    a known limitation that more recent protocols (e.g. SecAgg+) lift.
    """
    if not updates:
        raise ValueError("No client updates to aggregate.")

    N = len(updates)
    if dropouts_after_round1 is None:
        dropouts_after_round1 = []

    timings: Dict[str, float] = {}

    # ---- 1. Flatten each delta to a single 1-D vector. ----
    t0 = time.perf_counter()
    schema: Optional[List[Tuple[str, tuple]]] = None
    encoded_inputs: List[np.ndarray] = []
    for u in updates:
        flat, sch = _flatten_state(u.delta)
        if schema is None:
            schema = sch
        encoded_inputs.append(encode_floats(flat))
    timings["encode"] = time.perf_counter() - t0

    # ---- 2. Run the Bonawitz protocol on the encoded vectors. ----
    t0 = time.perf_counter()
    summed = run_secure_aggregation(
        inputs=encoded_inputs,
        threshold=threshold,
        dropouts_after_round1=dropouts_after_round1,
    )
    timings["protocol"] = time.perf_counter() - t0

    # ---- 3. Decode back to float means. ----
    t0 = time.perf_counter()
    num_survivors = N - len(dropouts_after_round1)
    averaged_flat = decode_int_sum(summed, num_clients=num_survivors)
    timings["decode"] = time.perf_counter() - t0

    # ---- 4. Reshape into a state dict and add to the global. ----
    assert schema is not None
    averaged_state = _unflatten_state(averaged_flat, schema, global_state)
    new_state: StateDict = {}
    for k, v in global_state.items():
        new_state[k] = v + averaged_state[k].to(v.device)

    if return_timings:
        timings["total"] = sum(timings.values())
        return new_state, timings
    return new_state


# ===========================================================================
# Smoke test — verify equivalence with plain federated_average
# ===========================================================================

if __name__ == "__main__":
    """Verify that secure_federated_average produces the same result as
    plain federated_average (up to quantization noise).
    """
    from ..fl_baseline import federated_average, ClientUpdate
    from ..model import MNIST_CNN

    print("=== Equivalence test: secure vs plain federated_average ===")
    torch.manual_seed(0)
    model = MNIST_CNN()
    global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Build N synthetic client updates (random deltas).
    N = 5
    updates = []
    for i in range(N):
        delta = {k: torch.randn_like(v) * 0.01 for k, v in global_state.items()}
        updates.append(ClientUpdate(delta=delta, num_samples=100, train_loss=0.0))

    # Plain aggregation (uniform mean).
    plain = federated_average(global_state, updates, weighted=False)

    # Secure aggregation.
    secure = secure_federated_average(global_state, updates)

    # Compare each parameter — should match within quantization tolerance
    # (~ 1/QUANT_SCALE = 1/65536 ≈ 1.5e-5).
    max_abs_diff = 0.0
    for k in plain:
        diff = (plain[k] - secure[k]).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
    print(f"Max |plain - secure|: {max_abs_diff:.2e}")
    print(f"Quantization tolerance (1/SCALE): {1 / QUANT_SCALE:.2e}")
    assert max_abs_diff < 1e-3, "Secure aggregation differs from plain too much!"
    print("OK -- secure aggregation matches plain aggregation.")
