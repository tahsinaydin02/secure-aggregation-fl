"""Smoke tests for the cryptographic primitives and the Secure Aggregation
protocol.

Run with `pytest -q` from the project root.

The crypto tests (primitives + protocol) need only numpy and `cryptography`.
The tests that touch the model or the FedAvg glue need torch, and are skipped
automatically if torch is not installed — so the security-critical layer can be
checked in a minimal environment.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.secure_agg.primitives import (
    AEAD,
    KeyAgreement,
    PRG,
    ShamirSecretSharing,
)
from src.secure_agg.protocol import MASK_MODULUS, run_secure_aggregation

try:  # pragma: no cover - environment dependent
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None

requires_torch = pytest.mark.skipif(_torch is None, reason="torch not installed")


# ---------------------------------------------------------------------------
# Key agreement
# ---------------------------------------------------------------------------

def test_ecdh_both_parties_derive_the_same_secret():
    ka = KeyAgreement()
    alice, bob = ka.generate_keypair(), ka.generate_keypair()
    assert ka.agree(alice.private_key, bob.public_bytes()) == \
           ka.agree(bob.private_key, alice.public_bytes())


def test_ecdh_secret_is_32_bytes_and_pair_specific():
    ka = KeyAgreement()
    a, b, c = (ka.generate_keypair() for _ in range(3))
    s_ab = ka.agree(a.private_key, b.public_bytes())
    s_ac = ka.agree(a.private_key, c.public_bytes())
    assert len(s_ab) == 32
    assert s_ab != s_ac


def test_ecdh_info_string_separates_derived_keys():
    """Different HKDF context strings must not collide on the same DH pair."""
    ka = KeyAgreement()
    a, b = ka.generate_keypair(), ka.generate_keypair()
    assert ka.agree(a.private_key, b.public_bytes(), info=b"mask") != \
           ka.agree(a.private_key, b.public_bytes(), info=b"encrypt")


# ---------------------------------------------------------------------------
# Authenticated encryption
# ---------------------------------------------------------------------------

def test_aead_roundtrip():
    key = os.urandom(32)
    msg = b"shares for client 7"
    assert AEAD.open(key, AEAD.seal(key, msg)) == msg


def test_aead_rejects_tampered_ciphertext():
    key = os.urandom(32)
    blob = bytearray(AEAD.seal(key, b"shares for client 7"))
    blob[-1] ^= 0x01  # flip one bit of the GCM tag
    with pytest.raises(Exception):
        AEAD.open(key, bytes(blob))


def test_aead_rejects_wrong_key():
    blob = AEAD.seal(os.urandom(32), b"secret")
    with pytest.raises(Exception):
        AEAD.open(os.urandom(32), blob)


def test_aead_nonce_is_fresh_per_encryption():
    key = os.urandom(32)
    assert AEAD.seal(key, b"same message") != AEAD.seal(key, b"same message")


# ---------------------------------------------------------------------------
# PRG
# ---------------------------------------------------------------------------

def test_prg_is_deterministic_in_the_seed():
    seed = os.urandom(32)
    assert np.array_equal(PRG(seed).int_vector(64), PRG(seed).int_vector(64))


def test_prg_differs_across_seeds():
    assert not np.array_equal(
        PRG(os.urandom(32)).int_vector(64),
        PRG(os.urandom(32)).int_vector(64),
    )


def test_prg_rejects_wrong_seed_length():
    with pytest.raises(ValueError):
        PRG(os.urandom(16))


# ---------------------------------------------------------------------------
# Shamir secret sharing
# ---------------------------------------------------------------------------

def test_sss_reconstructs_from_exactly_t_shares():
    sss = ShamirSecretSharing(threshold=3, num_shares=5)
    secret = 12345678901234567890
    shares = sss.share(secret)
    assert sss.reconstruct(shares[:3]) == secret


def test_sss_reconstructs_from_any_subset_of_size_t():
    sss = ShamirSecretSharing(threshold=3, num_shares=5)
    secret = 987654321
    shares = sss.share(secret)
    for subset in ([0, 2, 4], [1, 3, 4], [0, 1, 4]):
        assert sss.reconstruct([shares[i] for i in subset]) == secret


def test_sss_refuses_fewer_than_t_shares():
    sss = ShamirSecretSharing(threshold=3, num_shares=5)
    with pytest.raises(ValueError):
        sss.reconstruct(sss.share(42)[:2])


def test_sss_roundtrips_byte_secrets():
    """The protocol shares 32-byte keys and seeds, not integers."""
    sss = ShamirSecretSharing(threshold=4, num_shares=6)
    secret = os.urandom(32)
    shares = sss.share_bytes(secret)
    assert sss.reconstruct_bytes(shares[:4], length=32) == secret


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------

def _random_inputs(n, m, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 1000, size=m, dtype=np.int64) for _ in range(n)]


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_aggregation_recovers_the_exact_sum(n):
    inputs = _random_inputs(n, 32, seed=n)
    expected = sum(inputs) % MASK_MODULUS
    assert np.array_equal(run_secure_aggregation(inputs), expected)


def test_aggregation_survives_a_dropout():
    inputs = _random_inputs(5, 32, seed=1)
    result = run_secure_aggregation(inputs, dropouts_after_round1=[3])
    expected = sum(inputs[i] for i in range(5) if i + 1 != 3) % MASK_MODULUS
    assert np.array_equal(result, expected)


def test_aggregation_survives_multiple_dropouts_up_to_threshold():
    """With n=7 and t=4, three clients may vanish and the sum still resolves."""
    inputs = _random_inputs(7, 32, seed=2)
    dropouts = [2, 5, 7]
    result = run_secure_aggregation(inputs, dropouts_after_round1=dropouts)
    expected = sum(
        inputs[i] for i in range(7) if i + 1 not in dropouts
    ) % MASK_MODULUS
    assert np.array_equal(result, expected)


def test_masked_vectors_do_not_reveal_the_input():
    """What the server sees in Round 2 must not be the plaintext input.

    This is a sanity check, not a security proof: it confirms the masks are
    actually applied and are input-independent, i.e. that no client ships a
    vector that happens to equal its own contribution.
    """
    from src.secure_agg.protocol import SecAggClient, SecAggServer

    n, m, t = 4, 16, 3
    clients = {
        i: SecAggClient(client_id=i, num_clients=n, threshold=t, vector_length=m)
        for i in range(1, n + 1)
    }
    server = SecAggServer(num_clients=n, threshold=t, vector_length=m)
    broadcast = server.collect_round0([c.round0() for c in clients.values()])
    for c in clients.values():
        c.receive_round0_broadcast(broadcast)
    routed = server.route_round1([c.round1() for c in clients.values()])
    for c in clients.values():
        c.receive_round1(routed.get(c.client_id, {}))

    x = np.arange(m, dtype=np.int64)
    for c in clients.values():
        y = c.round2(x.copy()).masked_vector
        assert not np.array_equal(y, x)
        # A masked vector should look uniform over Z_{2^32}: with 16 entries
        # the chance of any entry staying below 2^20 by luck is negligible.
        assert (y > (1 << 20)).sum() >= m - 1


def test_protocol_rejects_a_single_client():
    with pytest.raises(ValueError):
        run_secure_aggregation(_random_inputs(1, 8))


def test_protocol_rejects_ragged_inputs():
    with pytest.raises(ValueError):
        run_secure_aggregation([np.zeros(8, dtype=np.int64),
                                np.zeros(9, dtype=np.int64)])


# ---------------------------------------------------------------------------
# Quantization and the FedAvg integration (need torch)
# ---------------------------------------------------------------------------

@requires_torch
def test_quantization_roundtrip_is_within_tolerance():
    from src.secure_agg.client_server import QUANT_SCALE, decode_int_sum, encode_floats

    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.05, size=500)          # gradient-scale values
    recovered = decode_int_sum(encode_floats(x), num_clients=1)
    assert np.max(np.abs(recovered - x)) < 1.0 / QUANT_SCALE


@requires_torch
def test_quantization_handles_negative_values():
    from src.secure_agg.client_server import decode_int_sum, encode_floats

    x = np.array([-1.5, -0.001, 0.0, 0.001, 1.5])
    recovered = decode_int_sum(encode_floats(x), num_clients=1)
    assert np.allclose(recovered, x, atol=1e-4)


@requires_torch
def test_secure_average_matches_plain_average():
    """The whole point of the protocol: same answer, without seeing the parts."""
    import torch

    from src.fl_baseline import ClientUpdate, federated_average
    from src.model import MNIST_CNN
    from src.secure_agg.client_server import QUANT_SCALE, secure_federated_average

    torch.manual_seed(0)
    global_state = {k: v.detach().clone()
                    for k, v in MNIST_CNN().state_dict().items()}
    updates = [
        ClientUpdate(
            delta={k: torch.randn_like(v) * 0.01 for k, v in global_state.items()},
            num_samples=100,
            train_loss=0.0,
        )
        for _ in range(5)
    ]

    plain = federated_average(global_state, updates, weighted=False)
    secure = secure_federated_average(global_state, updates)

    max_diff = max((plain[k] - secure[k]).abs().max().item() for k in plain)
    assert max_diff < 10.0 / QUANT_SCALE


@requires_torch
def test_model_is_twice_differentiable():
    """DLG differentiates through the gradient; verify that actually works."""
    import torch

    from src.model import MNIST_CNN

    model = MNIST_CNN()
    x = torch.randn(1, 1, 28, 28, requires_grad=True)
    loss = model(x).sum()
    grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
    second = torch.autograd.grad(sum(g.pow(2).sum() for g in grads), x)[0]
    assert torch.isfinite(second).all()
    assert second.abs().sum() > 0


@requires_torch
def test_idlg_label_extraction_is_exact():
    """iDLG's analytic label recovery must be right for every class."""
    import torch

    from src.dlg_attack import compute_victim_gradient, extract_label_from_gradient
    from src.model import MNIST_CNN

    torch.manual_seed(0)
    model = MNIST_CNN()
    for label in range(10):
        x = torch.randn(1, 1, 28, 28)
        grads = compute_victim_gradient(model, x, label)
        assert extract_label_from_gradient(grads) == label
