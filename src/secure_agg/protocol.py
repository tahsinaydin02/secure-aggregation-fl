"""Bonawitz Secure Aggregation protocol — Phase 3.

This module implements the four-round protocol from Bonawitz et al. (2017),
Section 5 (Figure 4), in the honest-but-curious setting (no PKI/signatures).

What the protocol achieves
--------------------------
Given N clients each holding a private vector x_u, the server learns

    z = sum_{u in survivors} x_u

without learning any individual x_u. Even if some clients drop out partway
through, the server can still recover z as long as at least t = ceil(N/2)+1
clients survive to the final round.

Why this defeats DLG
--------------------
In Phase 2 we showed that one client's gradient g_u leaks the input. After
Secure Aggregation, the server only sees the sum z = sum_u g_u — never any
individual g_u. The DLG attack, applied to z, faces an under-determined
problem: many possible (x_1, ..., x_N) tuples produce the same sum. Even if
the attacker tried to invert z, they would recover an *aggregate* shape,
not any individual client's input. We will demonstrate this in
experiments/04_dlg_vs_secure_agg.ipynb.

Round-by-round summary (honest-but-curious variant)
---------------------------------------------------
Round 0 - AdvertiseKeys
  Each client generates two ECDH keypairs:
    (c_pk, c_sk) -- used to derive AE keys (encryption between clients)
    (s_pk, s_sk) -- used to derive pairwise masking seeds
  Sends both public keys to the server, which broadcasts them to all clients.

Round 1 - ShareKeys
  Each client samples a random PRG seed b_u (the "self mask" seed).
  Splits s_sk_u and b_u into t-of-n Shamir shares.
  Encrypts each share for its destination client using the AE key derived
  from the c-keypair pair. Sends the ciphertexts (addressed) to the server,
  which routes them.

Round 2 - MaskedInputCollection
  Each client computes pairwise masks p_{u,v} = +/- PRG(KA(s_sk_u, s_pk_v))
  for every other client v, with sign +1 if u > v and -1 if u < v.
  Computes its self mask p_u = PRG(b_u).
  Sends y_u = x_u + p_u + sum_{v != u} p_{u,v}  (mod R)  to the server.

Round 3 - Unmasking  (Round 4 in paper; we skip Round 3 ConsistencyCheck)
  For each surviving client u, the server collects shares of b_u from t
  other survivors and reconstructs b_u, hence p_u. The server subtracts
  this from the sum.
  For each *dropped* client u, the server collects shares of s_sk_u from
  t survivors and reconstructs it. With s_sk_u in hand, the server can
  recompute p_{u,v} for every survivor v and subtract those from the sum.
  Final output: z = sum_u y_u - sum_{survivor u} p_u
                                - sum_{dropped u, survivor v} p_{u,v}.

The CRITICAL security invariant
-------------------------------
For any single client u, the server learns *exactly one* of:
    - b_u  (allowing it to remove p_u)        OR
    - s_sk_u  (allowing it to recompute pairwise masks involving u)
NEVER both. If the server had both, it could compute u's full mask and
hence u's input. The protocol enforces this by having survivors send
shares of b_u for survivors and shares of s_sk_u for dropouts — the share
sets are disjoint at the per-user level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .primitives import (
    AEAD,
    ECDHKeyPair,
    KeyAgreement,
    PRG,
    ShamirSecretSharing,
)


# Range for masking arithmetic. We work modulo R = 2^32 because:
#   - Gradients are quantized to int32 before masking (see encode_gradient
#     in client_server.py).
#   - PRG produces int64 values which we reduce mod R when summing.
#   - 2^32 is large enough that no realistic gradient sum overflows after
#     N <= 1000 clients.
MASK_MODULUS = 1 << 32

# Length of an ECDH P-256 private key in bytes — we share these via SSS,
# so we need to know the byte length to round-trip through bytes_to_int.
ECDH_PRIV_KEY_LEN = 32  # NIST P-256 has 256-bit private keys


# ===========================================================================
# Helpers for serializing private keys to bytes (so we can Shamir-share them)
# ===========================================================================

def serialize_private_key(kp: ECDHKeyPair) -> bytes:
    """Return the 32-byte big-endian encoding of an ECDH private scalar.

    The cryptography library exposes the private number directly. We pad to
    a fixed 32 bytes so reconstruct_bytes(...) always uses the same length.
    """
    private_number = kp.private_key.private_numbers().private_value
    return private_number.to_bytes(ECDH_PRIV_KEY_LEN, "big")


def deserialize_private_key(b: bytes):
    """Recreate an EllipticCurvePrivateKey from its 32-byte scalar.

    Used by the server in the unmasking round to reconstruct a dropped
    client's private key from Shamir shares, then derive the pairwise
    masks that involved that client.
    """
    from cryptography.hazmat.primitives.asymmetric import ec
    private_int = int.from_bytes(b, "big")
    return ec.derive_private_key(private_int, ec.SECP256R1())


# ===========================================================================
# Round message data structures
# ===========================================================================

@dataclass
class Round0Output:
    """What client u sends out in Round 0."""
    client_id: int
    c_public: bytes  # for AE key derivation
    s_public: bytes  # for pairwise mask seed derivation


@dataclass
class Round1Output:
    """What client u sends out in Round 1.

    The dict keys are the *destination* client IDs — one ciphertext per
    other client. Each ciphertext is an AE-sealed packet containing this
    client's Shamir shares of (s_sk_u, b_u) destined for that other client.
    """
    client_id: int
    encrypted_shares: Dict[int, bytes]  # to_id -> ciphertext


@dataclass
class Round2Output:
    """What client u sends out in Round 2: its masked input vector."""
    client_id: int
    masked_vector: np.ndarray  # shape (m,), dtype int64, values mod MASK_MODULUS


@dataclass
class Round3Request:
    """What the server asks each surviving client in Round 3.

    The server tells the client which other clients survived (so the client
    sends shares of b_u for those) and which dropped (so the client sends
    shares of s_sk_u for those). At no point are both share types requested
    for the same client — that would break the security invariant.
    """
    survivors: List[int]   # IDs of clients still alive
    dropouts: List[int]    # IDs of clients who dropped after Round 1


@dataclass
class Round3Output:
    """What client u sends back in Round 3.

    For each surviving v != u: a share of b_v that this client holds.
    For each dropped v: a share of s_sk_v that this client holds.
    """
    client_id: int
    b_shares: Dict[int, Tuple[int, int]]      # owner_id -> (x, f(x))
    s_sk_shares: Dict[int, Tuple[int, int]]   # owner_id -> (x, f(x))


# ===========================================================================
# The Client
# ===========================================================================

class SecAggClient:
    """A single client in the Secure Aggregation protocol.

    Lifecycle: instantiate -> round0() -> round1() -> round2(x) -> round3(...)
    Each method returns the message this client sends out in that round.
    The server orchestrates the calls and routes messages.
    """

    def __init__(
        self,
        client_id: int,
        num_clients: int,
        threshold: int,
        vector_length: int,
    ):
        """
        Parameters
        ----------
        client_id : int
            This client's logical ID, in 1..num_clients (1-indexed because
            Shamir shares use x=1..n; using 0 conflicts with f(0) being
            the secret).
        num_clients : int
            Total clients participating.
        threshold : int
            t for the t-of-n SSS — minimum survivors needed for recovery.
            Bonawitz recommends t = floor(n/2) + 1 for honest-but-curious.
        vector_length : int
            Length m of the input vector x_u that this client will mask.
            Used to size the PRG outputs.
        """
        if not (1 <= client_id <= num_clients):
            raise ValueError("client_id must be in 1..num_clients (1-indexed)")
        self.client_id = client_id
        self.n = num_clients
        self.t = threshold
        self.m = vector_length

        self.ka = KeyAgreement()
        self.sss = ShamirSecretSharing(threshold=threshold, num_shares=num_clients)

        # Filled in across rounds.
        self.c_kp: Optional[ECDHKeyPair] = None
        self.s_kp: Optional[ECDHKeyPair] = None
        self.b_seed: Optional[bytes] = None  # this client's PRG seed (self mask)

        # Other clients' public keys (received in Round 0 broadcast).
        self.other_c_publics: Dict[int, bytes] = {}
        self.other_s_publics: Dict[int, bytes] = {}

        # Shares of *other* clients' secrets that we hold. Keyed by owner ID.
        self.s_sk_shares_held: Dict[int, Tuple[int, int]] = {}
        self.b_shares_held: Dict[int, Tuple[int, int]] = {}

    # ----- Round 0 -----

    def round0(self) -> Round0Output:
        """Generate two key pairs and advertise the public keys."""
        self.c_kp = self.ka.generate_keypair()
        self.s_kp = self.ka.generate_keypair()
        return Round0Output(
            client_id=self.client_id,
            c_public=self.c_kp.public_bytes(),
            s_public=self.s_kp.public_bytes(),
        )

    def receive_round0_broadcast(self, all_round0: List[Round0Output]) -> None:
        """Store the public keys of all other clients."""
        for msg in all_round0:
            if msg.client_id == self.client_id:
                continue
            self.other_c_publics[msg.client_id] = msg.c_public
            self.other_s_publics[msg.client_id] = msg.s_public

    # ----- Round 1 -----

    def round1(self) -> Round1Output:
        """Sample b_u, share s_sk_u and b_u via SSS, encrypt for each peer."""
        assert self.s_kp is not None and self.c_kp is not None

        # 1. Sample our PRG seed b_u (the self mask seed).
        from os import urandom
        self.b_seed = urandom(32)

        # 2. Shamir-share our s-private-key and b-seed.
        s_sk_bytes = serialize_private_key(self.s_kp)
        s_sk_shares = self.sss.share_bytes(s_sk_bytes)        # n shares
        b_shares = self.sss.share_bytes(self.b_seed)          # n shares

        # The shares are 1-indexed by Shamir share id (1..n). We assume the
        # share with id v is destined for client v. This is the standard
        # mapping in Bonawitz Figure 4.
        #
        # IMPORTANT: each client also keeps a share of its OWN secret. This
        # share is needed at unmasking time because the server collects t
        # shares per secret, and a client's own share counts toward that
        # threshold. Without this, the protocol fails for small N where
        # t can equal N (every client's share is needed).
        own_s_sk_share = s_sk_shares[self.client_id - 1]  # share addressed to self
        own_b_share = b_shares[self.client_id - 1]
        self.s_sk_shares_held[self.client_id] = own_s_sk_share
        self.b_shares_held[self.client_id] = own_b_share

        encrypted: Dict[int, bytes] = {}
        for to_id in range(1, self.n + 1):
            if to_id == self.client_id:
                # We already stored our own shares above; nothing to send.
                continue

            # Find the share addressed to client `to_id`. Shares are
            # ordered (x, y) starting at x=1.
            s_sk_share = s_sk_shares[to_id - 1]   # (to_id, value)
            b_share = b_shares[to_id - 1]

            # Build the plaintext packet. We tag it with metadata so the
            # recipient can verify it is intended for them and from us.
            # Format (simple length-prefix encoding):
            #   from_id (4 bytes BE) | to_id (4) |
            #   s_sk_share x (4) | s_sk_share y (66 bytes, big-endian, padded)
            #   b_share x   (4) | b_share y   (66 bytes)
            # 66 bytes covers any value < 2^521.
            packet = self._encode_share_packet(
                self.client_id, to_id, s_sk_share, b_share,
            )

            # Derive the AE key from the c-keypair shared secret with `to_id`.
            ae_key = self.ka.agree(
                self.c_kp.private_key, self.other_c_publics[to_id],
            )
            encrypted[to_id] = AEAD.seal(ae_key, packet)

        return Round1Output(
            client_id=self.client_id,
            encrypted_shares=encrypted,
        )

    def receive_round1(self, ciphertexts_for_me: Dict[int, bytes]) -> None:
        """Decrypt incoming share ciphertexts. ciphertexts_for_me: from_id -> ct."""
        assert self.c_kp is not None
        for from_id, ct in ciphertexts_for_me.items():
            ae_key = self.ka.agree(
                self.c_kp.private_key, self.other_c_publics[from_id],
            )
            packet = AEAD.open(ae_key, ct)
            f_id, t_id, s_sk_share, b_share = self._decode_share_packet(packet)
            assert f_id == from_id and t_id == self.client_id
            self.s_sk_shares_held[from_id] = s_sk_share
            self.b_shares_held[from_id] = b_share

    # ----- Round 2 -----

    def round2(self, x: np.ndarray) -> Round2Output:
        """Compute and send the masked vector y_u = x + p_u + sum p_{u,v}."""
        assert self.s_kp is not None and self.b_seed is not None
        if x.shape != (self.m,):
            raise ValueError(f"Expected x of shape ({self.m},), got {x.shape}")

        # All arithmetic is in int64 mod MASK_MODULUS. Inputs are converted
        # to int64 by the caller (see encode_gradient in client_server.py).
        y = x.astype(np.int64).copy()

        # 1. Self mask p_u = PRG(b_u).
        self_mask = PRG(self.b_seed).int_vector(self.m)
        y = (y + self_mask) % MASK_MODULUS

        # 2. Pairwise masks p_{u,v}, with sign convention +1 if u>v, -1 if u<v.
        for v_id, v_pub in self.other_s_publics.items():
            shared = self.ka.agree(self.s_kp.private_key, v_pub)
            mask = PRG(shared).int_vector(self.m)
            sign = 1 if self.client_id > v_id else -1
            y = (y + sign * mask) % MASK_MODULUS

        return Round2Output(client_id=self.client_id, masked_vector=y)

    # ----- Round 3 -----

    def round3(self, request: Round3Request) -> Round3Output:
        """Send back the appropriate Shamir shares.

        For survivors: shares of THEIR b-seed (we hold one share per peer).
        For dropouts:  shares of THEIR s-private-key.

        Note: we never send shares of both for the same client. The server
        promises in Round3Request to ask for at most one type per peer.
        """
        b_shares_to_send: Dict[int, Tuple[int, int]] = {}
        for surv_id in request.survivors:
            # We send the share we hold for survivor surv_id. This includes
            # OUR OWN share (the one we kept in round1) when surv_id ==
            # self.client_id — that share counts toward the server's t-share
            # reconstruction threshold for our own secret.
            if surv_id in self.b_shares_held:
                b_shares_to_send[surv_id] = self.b_shares_held[surv_id]

        s_sk_shares_to_send: Dict[int, Tuple[int, int]] = {}
        for drop_id in request.dropouts:
            if drop_id in self.s_sk_shares_held:
                s_sk_shares_to_send[drop_id] = self.s_sk_shares_held[drop_id]

        return Round3Output(
            client_id=self.client_id,
            b_shares=b_shares_to_send,
            s_sk_shares=s_sk_shares_to_send,
        )

    # ----- Internal helpers -----

    @staticmethod
    def _encode_share_packet(
        from_id: int,
        to_id: int,
        s_sk_share: Tuple[int, int],
        b_share: Tuple[int, int],
    ) -> bytes:
        """Pack a peer-to-peer share packet into bytes for AE-sealing.

        Format (all big-endian):
          from_id                                - 4 bytes
          to_id                                  - 4 bytes
          s_sk x                                 - 4 bytes
          s_sk y (length 66)                     - 66 bytes
          b    x                                 - 4 bytes
          b    y (length 66)                     - 66 bytes
        Total: 148 bytes
        """
        # 66 bytes covers any value < 2^528, comfortably above 2^521-1.
        Y_LEN = 66
        return b"".join([
            from_id.to_bytes(4, "big"),
            to_id.to_bytes(4, "big"),
            s_sk_share[0].to_bytes(4, "big"),
            s_sk_share[1].to_bytes(Y_LEN, "big"),
            b_share[0].to_bytes(4, "big"),
            b_share[1].to_bytes(Y_LEN, "big"),
        ])

    @staticmethod
    def _decode_share_packet(
        packet: bytes,
    ) -> Tuple[int, int, Tuple[int, int], Tuple[int, int]]:
        """Inverse of _encode_share_packet."""
        Y_LEN = 66
        offset = 0
        from_id = int.from_bytes(packet[offset : offset + 4], "big"); offset += 4
        to_id = int.from_bytes(packet[offset : offset + 4], "big"); offset += 4
        s_x = int.from_bytes(packet[offset : offset + 4], "big"); offset += 4
        s_y = int.from_bytes(packet[offset : offset + Y_LEN], "big"); offset += Y_LEN
        b_x = int.from_bytes(packet[offset : offset + 4], "big"); offset += 4
        b_y = int.from_bytes(packet[offset : offset + Y_LEN], "big"); offset += Y_LEN
        return from_id, to_id, (s_x, s_y), (b_x, b_y)


# ===========================================================================
# The Server
# ===========================================================================

class SecAggServer:
    """The server in the Secure Aggregation protocol.

    The server is "honest-but-curious": it follows the protocol but is
    interested in inferring individual x_u from what it sees. Our job is
    to ensure that after running the protocol, the server has learned
    *only* the sum z = sum_u x_u and *nothing* about any individual x_u.
    """

    def __init__(
        self,
        num_clients: int,
        threshold: int,
        vector_length: int,
    ):
        self.n = num_clients
        self.t = threshold
        self.m = vector_length
        self.ka = KeyAgreement()
        self.sss = ShamirSecretSharing(threshold=threshold, num_shares=num_clients)

        # Per-round storage.
        self._round0_msgs: Dict[int, Round0Output] = {}
        self._round2_msgs: Dict[int, Round2Output] = {}

    def collect_round0(self, msgs: List[Round0Output]) -> List[Round0Output]:
        """Receive round 0 from clients, return broadcast message."""
        for m in msgs:
            self._round0_msgs[m.client_id] = m
        # Broadcast = the same set, sent to everyone.
        return list(self._round0_msgs.values())

    def route_round1(
        self, msgs: List[Round1Output],
    ) -> Dict[int, Dict[int, bytes]]:
        """Re-route ciphertexts: for each recipient, gather all incoming.

        Returns a dict: recipient_id -> {sender_id -> ciphertext}.
        The server cannot decrypt these (no AE key); it just routes.
        """
        out: Dict[int, Dict[int, bytes]] = {}
        for sender_msg in msgs:
            sender = sender_msg.client_id
            for recipient, ct in sender_msg.encrypted_shares.items():
                out.setdefault(recipient, {})[sender] = ct
        return out

    def collect_round2(self, msgs: List[Round2Output]) -> None:
        """Store the masked vectors received from all surviving clients."""
        for m in msgs:
            self._round2_msgs[m.client_id] = m

    def make_round3_request(
        self, dropouts_after_round1: List[int],
    ) -> Round3Request:
        """Tell each survivor which shares to release."""
        survivors = sorted(self._round2_msgs.keys())
        return Round3Request(
            survivors=survivors,
            dropouts=sorted(dropouts_after_round1),
        )

    def aggregate(
        self,
        round3_outputs: List[Round3Output],
        dropouts_after_round1: List[int],
    ) -> np.ndarray:
        """Final unmasking — produce the aggregate sum.

        We have:
          - sum of all masked vectors from survivors:
              S = sum_{u in survivors} y_u
              = sum_u (x_u + p_u + sum_{v != u} +/- p_{u,v})
            Pairwise masks among survivors cancel (each appears once with +
            and once with -). Pairwise masks involving a dropout DO NOT
            cancel — they are the dropout's contribution that we must
            reconstruct and subtract.

          - For each survivor u: shares of b_u from t-1 other survivors.
            We reconstruct b_u, compute p_u = PRG(b_u), subtract p_u from S.

          - For each dropout d: shares of s_sk_d from t survivors.
            We reconstruct s_sk_d, derive d's pairwise shared secret with
            each *survivor* v, expand to a mask, and subtract (with the
            appropriate sign) from S.
        """
        # 1. Sum the masked vectors.
        survivor_ids = sorted(self._round2_msgs.keys())
        S = np.zeros(self.m, dtype=np.int64)
        for sid in survivor_ids:
            S = (S + self._round2_msgs[sid].masked_vector) % MASK_MODULUS

        # 2. Aggregate the round-3 share contributions by owner ID.
        #    b_pool[owner] = list of (x, y) shares we have for owner's b_seed.
        #    s_pool[owner] = list of (x, y) shares we have for owner's s_sk.
        b_pool: Dict[int, List[Tuple[int, int]]] = {}
        s_pool: Dict[int, List[Tuple[int, int]]] = {}
        for r3 in round3_outputs:
            for owner, share in r3.b_shares.items():
                b_pool.setdefault(owner, []).append(share)
            for owner, share in r3.s_sk_shares.items():
                s_pool.setdefault(owner, []).append(share)

        # 3. Remove each survivor's self mask.
        for u in survivor_ids:
            shares = b_pool.get(u, [])
            if len(shares) < self.t:
                raise RuntimeError(
                    f"Cannot reconstruct b_seed for survivor {u}: "
                    f"only {len(shares)} shares, need {self.t}"
                )
            b_seed = self.sss.reconstruct_bytes(shares, length=32)
            self_mask = PRG(b_seed).int_vector(self.m)
            S = (S - self_mask) % MASK_MODULUS

        # 4. Remove each dropout's pairwise contributions to surviving y_u.
        for d in dropouts_after_round1:
            shares = s_pool.get(d, [])
            if len(shares) < self.t:
                raise RuntimeError(
                    f"Cannot reconstruct s_sk for dropout {d}: "
                    f"only {len(shares)} shares, need {self.t}"
                )
            s_sk_bytes = self.sss.reconstruct_bytes(
                shares, length=ECDH_PRIV_KEY_LEN,
            )
            d_private = deserialize_private_key(s_sk_bytes)

            # For each survivor v, the masked y_v contained the pairwise
            # mask between d and v with sign (+1 if v>d else -1) from v's
            # perspective. We need to subtract those contributions.
            for v in survivor_ids:
                if v == d:
                    continue
                v_pub = self._round0_msgs[v].s_public
                shared = self.ka.agree(d_private, v_pub)
                mask = PRG(shared).int_vector(self.m)
                # v added sign * mask, where sign = +1 if v > d else -1.
                # So we subtract that same sign * mask from S.
                sign = 1 if v > d else -1
                S = (S - sign * mask) % MASK_MODULUS

        # Return as positive int64 mod MASK_MODULUS. Caller (client_server.py)
        # will reinterpret as signed if needed.
        return S % MASK_MODULUS


# ===========================================================================
# Convenience: run the protocol end-to-end in-memory (no real network)
# ===========================================================================

def run_secure_aggregation(
    inputs: List[np.ndarray],
    threshold: Optional[int] = None,
    dropouts_after_round1: Optional[List[int]] = None,
) -> np.ndarray:
    """Drive the full protocol on a list of input vectors and return the sum.

    This is the convenience wrapper that tests and notebooks call. It
    creates N clients and one server, runs all four rounds, and returns
    the unmasked aggregate.

    Parameters
    ----------
    inputs : list of np.ndarray, length N
        inputs[i] is client (i+1)'s private vector. All same length m.
    threshold : int | None
        SSS threshold t. Defaults to floor(N/2) + 1, the Bonawitz
        recommendation for honest-but-curious.
    dropouts_after_round1 : list of int | None
        Client IDs (1-indexed) that drop out after round 1 — i.e. they
        completed the share distribution but never sent their masked
        vector. Used to test dropout robustness. Defaults to none.

    Returns
    -------
    np.ndarray of dtype int64, shape (m,)
        The aggregate sum of inputs from clients that survived to round 2.
    """
    N = len(inputs)
    if N < 2:
        raise ValueError("Need at least 2 clients.")
    m = inputs[0].shape[0]
    for v in inputs:
        if v.shape != (m,):
            raise ValueError("All input vectors must have the same length.")

    if threshold is None:
        threshold = N // 2 + 1
    if dropouts_after_round1 is None:
        dropouts_after_round1 = []

    # 1. Build N clients (1-indexed) and one server.
    clients: Dict[int, SecAggClient] = {
        i + 1: SecAggClient(
            client_id=i + 1, num_clients=N, threshold=threshold, vector_length=m,
        )
        for i in range(N)
    }
    server = SecAggServer(num_clients=N, threshold=threshold, vector_length=m)

    # 2. Round 0: clients advertise public keys; server broadcasts.
    r0_msgs = [c.round0() for c in clients.values()]
    broadcast = server.collect_round0(r0_msgs)
    for c in clients.values():
        c.receive_round0_broadcast(broadcast)

    # 3. Round 1: clients distribute encrypted shares.
    r1_msgs = [c.round1() for c in clients.values()]
    routed = server.route_round1(r1_msgs)  # recipient -> {sender -> ct}
    for c in clients.values():
        c.receive_round1(routed.get(c.client_id, {}))

    # 4. Round 2: surviving clients send masked vectors. Dropouts skip.
    r2_msgs = []
    for cid, c in clients.items():
        if cid in dropouts_after_round1:
            continue
        r2_msgs.append(c.round2(inputs[cid - 1]))
    server.collect_round2(r2_msgs)

    # 5. Round 3: server requests shares from surviving clients; aggregates.
    request = server.make_round3_request(dropouts_after_round1)
    r3_msgs = []
    for cid, c in clients.items():
        if cid in dropouts_after_round1:
            continue
        r3_msgs.append(c.round3(request))

    return server.aggregate(r3_msgs, dropouts_after_round1)


if __name__ == "__main__":
    # End-to-end smoke test: 5 clients sum five small vectors.
    print("=== Secure Aggregation smoke test ===")
    rng = np.random.default_rng(0)
    N = 5
    M = 10
    # Use small int values so we can verify the sum exactly.
    inputs = [rng.integers(0, 100, size=M, dtype=np.int64) for _ in range(N)]

    expected = np.zeros(M, dtype=np.int64)
    for v in inputs:
        expected += v
    print(f"Expected sum: {expected}")

    aggregated = run_secure_aggregation(inputs)
    # The protocol works mod 2^32. Since our values are small, the result
    # equals the expected sum exactly.
    print(f"Computed sum: {aggregated}")
    assert np.array_equal(aggregated, expected % MASK_MODULUS), \
        "Aggregation correctness failed!"
    print("OK -- protocol correctly recovers the sum.\n")

    # Now test with a dropout.
    print("=== Dropout test (1 client drops after round 1) ===")
    aggregated_drop = run_secure_aggregation(
        inputs, dropouts_after_round1=[3],
    )
    expected_drop = sum(inputs[i] for i in range(N) if i + 1 != 3)
    print(f"Expected (without client 3): {expected_drop}")
    print(f"Computed:                    {aggregated_drop}")
    assert np.array_equal(aggregated_drop, expected_drop.astype(np.int64) % MASK_MODULUS), \
        "Dropout aggregation failed!"
    print("OK -- protocol survives a client dropout.")
