"""Cryptographic primitives for Secure Aggregation (Phase 3).

This module wraps the building blocks described in Section 3 of Bonawitz
et al. (2017). Each primitive maps to one of the paper's subsections:

    Section 3.1 Secret Sharing       -> ShamirSecretSharing class
    Section 3.2 Key Agreement        -> KeyAgreement class (ECDH wrapper)
    Section 3.3 Authenticated Encryption -> AEAD class (AES-GCM wrapper)
    Section 3.4 Pseudorandom Generator   -> PRG class (AES-CTR based)
    Section 3.5 Signatures, 3.6 PKI      -> NOT implemented (only needed
                                            for active-adversary variant)

Why I implement my own thin wrappers
------------------------------------
The `cryptography` library is the right thing to use for the actual
crypto primitives — rolling our own AES is the cardinal sin of
applied cryptography. But the library's API is designed for general use,
and the protocol uses a very specific subset. Wrapping the primitives
in named classes here lets the protocol code in `protocol.py` read
naturally — "ka.agree(my_sk, their_pk)" instead of "ec.derive_private...
" with cumbersome key-loading boilerplate.

For Shamir Secret Sharing I implement the polynomial interpolation by
hand. SSS is rarely a packaged primitive in mainstream crypto libraries
(it is more of a specialized tool), and writing it ourselves makes the
math visible — useful for the report's discussion section.

Security parameters
-------------------
We follow the choices Bonawitz et al. make in their prototype (Section 7.3):
    - ECDH on NIST P-256 curve, derived to 32-byte shared secrets via SHA-256
    - AES-256-GCM for authenticated encryption (paper uses 128-bit; we use 256
      because there is no cost difference for our small messages and 256 is
      the modern default)
    - AES-CTR with 256-bit key as the PRG
    - SSS over a large prime field (we use 2^521 - 1, a Mersenne prime, large
      enough to hold any 64-bit-ish secret we put through it)
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ===========================================================================
# 3.2  Key Agreement (Elliptic-Curve Diffie-Hellman, NIST P-256 + HKDF-SHA256)
# ===========================================================================

@dataclass
class ECDHKeyPair:
    """An ECDH key pair, with utility for serializing the public key.

    The protocol needs to send public keys through the server (which is
    untrusted), so we serialize them to raw bytes that can be transported
    in the round messages.
    """

    private_key: ec.EllipticCurvePrivateKey
    public_key: ec.EllipticCurvePublicKey

    def public_bytes(self) -> bytes:
        """Public key as a compact byte string suitable for transport."""
        # SubjectPublicKeyInfo is overkill for our use case but it round-trips
        # cleanly through cryptography.serialization.
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )


class KeyAgreement:
    """Wraps Elliptic-Curve Diffie-Hellman over NIST P-256.

    Maps directly to Bonawitz Section 3.2:
        KA.gen()       -> generate_keypair()
        KA.agree()     -> agree(my_private, their_public_bytes)

    We add an HKDF-SHA256 step on top of the raw shared secret to convert
    the elliptic curve point into a uniformly random 32-byte key, which is
    what the rest of the protocol expects (PRG seed, AE key, etc.). This
    HKDF step is the H(g^{ab}) hash extraction the paper folds into the
    KA.agree definition.
    """

    CURVE = ec.SECP256R1()

    def generate_keypair(self) -> ECDHKeyPair:
        """KA.gen — sample a fresh ephemeral key pair."""
        sk = ec.generate_private_key(self.CURVE)
        pk = sk.public_key()
        return ECDHKeyPair(private_key=sk, public_key=pk)

    def agree(
        self,
        my_private: ec.EllipticCurvePrivateKey,
        their_public_bytes: bytes,
        *,
        info: bytes = b"bonawitz-secure-aggregation",
    ) -> bytes:
        """KA.agree — compute the shared secret with another party.

        Parameters
        ----------
        my_private : private key (locally stored).
        their_public_bytes : the other party's public key, serialized as
            DER bytes (as produced by ECDHKeyPair.public_bytes).
        info : HKDF context string. Same value on both sides; it binds the
            derived key to this protocol so the same ECDH pair could be
            reused for a different protocol without key collision.

        Returns
        -------
        bytes : 32-byte shared secret, suitable as a symmetric key or seed.
        """
        their_public = serialization.load_der_public_key(their_public_bytes)
        if not isinstance(their_public, ec.EllipticCurvePublicKey):
            raise TypeError("Expected an EC public key.")

        # Raw ECDH shared point's x-coordinate, as bytes. Not yet uniform
        # over the byte string space, so we feed it through HKDF.
        raw_secret = my_private.exchange(ec.ECDH(), their_public)

        # HKDF turns the raw secret into a uniformly distributed 32-byte key.
        # salt=None is fine when the input is already a high-entropy secret.
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=info,
        ).derive(raw_secret)


# ===========================================================================
# 3.3  Authenticated Encryption (AES-GCM)
# ===========================================================================

class AEAD:
    """Authenticated encryption wrapper using AES-256-GCM.

    Maps to Bonawitz Section 3.3:
        AE.enc(c, x) -> seal(key, plaintext)
        AE.dec(c, ct) -> open(key, ciphertext)

    The Bonawitz protocol uses AE for client-to-client messages routed
    through the (untrusted) server. The server cannot read or tamper with
    these messages because it does not hold the AE key (which is derived
    from the two clients' DH shared secret).

    GCM nonces must be unique per key. We sample a fresh 12-byte random
    nonce for every encryption and prepend it to the ciphertext so the
    decryptor can find it.
    """

    NONCE_LEN = 12  # AES-GCM standard nonce length

    @staticmethod
    def seal(key: bytes, plaintext: bytes, *, associated_data: bytes = b"") -> bytes:
        """AE.enc — encrypt and authenticate.

        Output layout: nonce (12 bytes) || ciphertext-with-tag.
        """
        if len(key) != 32:
            raise ValueError("AEAD requires a 32-byte key (AES-256-GCM).")
        nonce = os.urandom(AEAD.NONCE_LEN)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, plaintext, associated_data)
        return nonce + ct

    @staticmethod
    def open(key: bytes, blob: bytes, *, associated_data: bytes = b"") -> bytes:
        """AE.dec — verify and decrypt.

        Raises if the ciphertext was tampered with or the key is wrong.
        """
        if len(blob) < AEAD.NONCE_LEN + 16:  # 16 = GCM tag length
            raise ValueError("Ciphertext blob too short.")
        nonce, ct = blob[: AEAD.NONCE_LEN], blob[AEAD.NONCE_LEN :]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, associated_data)


# ===========================================================================
# 3.4  Pseudorandom Generator (AES-CTR)
# ===========================================================================

class PRG:
    """Pseudorandom Generator built from AES-256 in counter mode.

    Maps to Bonawitz Section 3.4. Given a 32-byte seed, expand it to as
    many bytes as needed. Same seed -> same output, deterministically.

    Why AES-CTR? It is the canonical CSPRNG construction: AES is a strong
    PRP, and counter mode turns it into a stream of pseudorandom bytes.
    Standard, fast, and constant-time on modern CPUs that have AES-NI.

    Why not Python's `random`? Python's Mersenne Twister is NOT
    cryptographically secure — its state can be recovered from a few
    hundred outputs. Using it here would let an attacker who observed a
    fragment of the mask reconstruct the rest.

    Output format
    -------------
    The protocol needs to expand a seed into a vector of integers that the
    masking code then reduces modulo R (the masking range, 2^32 here). We
    provide:
        - bytes(n)      : n raw pseudorandom bytes
        - int_vector(m) : m int64 values, uniform over the full signed
                          int64 range (8 keystream bytes reinterpreted per
                          element, so roughly half of them are negative)
    The reduction mod R happens in protocol.py, not here — every use site
    adds or subtracts these values inside a `% MASK_MODULUS` expression.
    """

    def __init__(self, seed: bytes):
        if len(seed) != 32:
            raise ValueError("PRG seed must be 32 bytes (256 bits).")
        # AES-CTR with a fixed all-zero initial counter. Different seeds
        # give different keystreams; same seed gives identical output.
        # We never reuse a seed across protocol instances, so the constant
        # nonce is safe (two-time pad would only matter with reuse).
        self._cipher = Cipher(
            algorithms.AES(seed),
            modes.CTR(b"\x00" * 16),
        ).encryptor()

    def bytes(self, n: int) -> bytes:
        """Produce n pseudorandom bytes."""
        # Encrypt n zero bytes -> get n keystream bytes.
        return self._cipher.update(b"\x00" * n)

    def int_vector(self, m: int, *, dtype=np.int64) -> np.ndarray:
        """Produce m pseudorandom int64 values as a numpy array.

        We generate 8 bytes per element and reinterpret them as int64, so
        the values span the whole signed int64 range. That is deliberate:
        callers immediately reduce mod 2^32, and numpy's `%` returns a
        non-negative residue for negative operands, so the sign of the raw
        draw is irrelevant to the masking arithmetic. What matters is that
        the same seed always yields the same vector, so the pairwise masks
        cancel exactly.
        """
        raw = self.bytes(m * 8)
        return np.frombuffer(raw, dtype=np.int64)[:m].copy()


# ===========================================================================
# 3.1  Shamir's t-out-of-n Secret Sharing
# ===========================================================================

# We work in the prime field GF(p), where p is a large Mersenne prime.
# 2^521 - 1 is comfortably larger than any 32-byte secret we will share
# (which fits in 256 bits), so secrets embed unambiguously.
SHAMIR_PRIME = (1 << 521) - 1  # 13th Mersenne prime, M521


class ShamirSecretSharing:
    """Shamir's t-out-of-n secret sharing over GF(SHAMIR_PRIME).

    Maps to Bonawitz Section 3.1.

    The scheme:
      To share a secret s among n parties such that any t can reconstruct it:
        1. Pick random coefficients a_1, ..., a_{t-1} in GF(p).
        2. Form polynomial f(x) = s + a_1*x + a_2*x^2 + ... + a_{t-1}*x^(t-1).
        3. Each party i (i = 1..n) gets the share (i, f(i)).
      Reconstruction: any t shares uniquely determine f, then s = f(0).
      Privacy: any t-1 shares reveal nothing about s — the polynomial is
               consistent with all p possible values of s.

    Why I implemented this myself rather than using a library
    ---------------------------------------------------------
    Off-the-shelf SSS libraries for Python are surprisingly inconsistent
    (some encode shares as base-58 strings, some assume bytes, some don't
    work over arbitrary fields). For pedagogical clarity and to keep the
    math visible, we implement the polynomial arithmetic directly using
    Python's built-in arbitrary-precision integers. Performance is fine
    for our scale (n <= 50, t <= n).
    """

    def __init__(self, threshold: int, num_shares: int):
        """Configure for a t-of-n scheme.

        Parameters
        ----------
        threshold : t — minimum number of shares needed to reconstruct.
        num_shares : n — total number of shares produced.
        """
        if not (1 <= threshold <= num_shares):
            raise ValueError(f"Bad params: t={threshold}, n={num_shares}")
        self.t = threshold
        self.n = num_shares
        self.p = SHAMIR_PRIME

    # ----- Sharing -----

    def share(self, secret_int: int) -> List[Tuple[int, int]]:
        """Produce n shares of `secret_int`.

        Parameters
        ----------
        secret_int : the secret as a non-negative integer < p.

        Returns
        -------
        list of (share_id, share_value) pairs, length n.
        share_id is 1..n (we never use 0 as a share id since f(0) is the
        secret itself).
        """
        if not (0 <= secret_int < self.p):
            raise ValueError(f"Secret must be in [0, {self.p}). Got {secret_int}.")

        # Random polynomial coefficients a_1 .. a_{t-1}. The constant term
        # is the secret itself.
        coeffs = [secret_int] + [
            secrets.randbelow(self.p) for _ in range(self.t - 1)
        ]

        # Evaluate at x = 1, 2, ..., n via Horner's method.
        shares = []
        for x in range(1, self.n + 1):
            y = 0
            for c in reversed(coeffs):
                y = (y * x + c) % self.p
            shares.append((x, y))
        return shares

    # ----- Reconstruction -----

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        """Reconstruct the secret from at least t shares.

        Lagrange interpolation at x=0:
          s = f(0) = sum_i  y_i * prod_{j != i} (0 - x_j) / (x_i - x_j)
                   = sum_i  y_i * prod_{j != i} (-x_j) / (x_i - x_j)
        Modular division by (x_i - x_j) is done via the modular inverse.
        """
        if len(shares) < self.t:
            raise ValueError(
                f"Need at least {self.t} shares to reconstruct; got {len(shares)}."
            )

        # Use exactly t shares (extras don't help, just slow us down).
        used = shares[: self.t]
        xs = [x for x, _ in used]
        ys = [y for _, y in used]

        secret = 0
        for i in range(len(used)):
            num = 1   # numerator of Lagrange basis L_i(0)
            den = 1   # denominator
            for j in range(len(used)):
                if i == j:
                    continue
                # 0 - x_j  in the numerator
                num = (num * (-xs[j])) % self.p
                # x_i - x_j  in the denominator
                den = (den * (xs[i] - xs[j])) % self.p
            # Modular inverse of denominator (Fermat's little theorem since
            # p is prime: a^{-1} = a^{p-2} mod p).
            den_inv = pow(den, self.p - 2, self.p)
            secret = (secret + ys[i] * num * den_inv) % self.p

        return secret

    # ----- Convenience helpers for byte-string secrets -----
    #
    # The protocol shares byte strings (DH private keys, PRG seeds), not
    # integers. These helpers handle the bytes <-> int conversion so the
    # protocol code does not have to.

    @staticmethod
    def bytes_to_int(b: bytes) -> int:
        """Big-endian byte-string -> non-negative integer."""
        return int.from_bytes(b, "big")

    @staticmethod
    def int_to_bytes(i: int, length: int) -> bytes:
        """Non-negative integer -> fixed-length big-endian byte string."""
        return i.to_bytes(length, "big")

    def share_bytes(self, secret_bytes: bytes) -> List[Tuple[int, int]]:
        """Shortcut: share a byte-string secret."""
        return self.share(self.bytes_to_int(secret_bytes))

    def reconstruct_bytes(
        self, shares: List[Tuple[int, int]], length: int,
    ) -> bytes:
        """Shortcut: reconstruct a byte-string secret of known length."""
        return self.int_to_bytes(self.reconstruct(shares), length)


# ===========================================================================
# Smoke test — run as a script to verify everything works end-to-end
# ===========================================================================

if __name__ == "__main__":
    print("=== Key Agreement (ECDH P-256 + HKDF-SHA256) ===")
    ka = KeyAgreement()
    alice = ka.generate_keypair()
    bob = ka.generate_keypair()
    s_ab = ka.agree(alice.private_key, bob.public_bytes())
    s_ba = ka.agree(bob.private_key, alice.public_bytes())
    assert s_ab == s_ba, "ECDH shared secrets must match!"
    print(f"  Alice and Bob agreed on {s_ab.hex()[:32]}...  OK")

    print("\n=== Authenticated Encryption (AES-GCM) ===")
    key = os.urandom(32)
    msg = b"hello secure world"
    blob = AEAD.seal(key, msg)
    decrypted = AEAD.open(key, blob)
    assert decrypted == msg, "AE round-trip failed!"
    print(f"  Encrypted {len(msg)} bytes -> {len(blob)} bytes (incl. nonce+tag), decrypted OK")

    print("\n=== Pseudorandom Generator (AES-CTR) ===")
    seed = os.urandom(32)
    p1 = PRG(seed); v1 = p1.int_vector(10)
    p2 = PRG(seed); v2 = p2.int_vector(10)
    assert np.array_equal(v1, v2), "Same seed must give same output!"
    p3 = PRG(os.urandom(32)); v3 = p3.int_vector(10)
    assert not np.array_equal(v1, v3), "Different seeds must differ!"
    print(f"  10-element vector from seed: {v1[:3]}... (deterministic)  OK")

    print("\n=== Shamir Secret Sharing (3-of-5) ===")
    sss = ShamirSecretSharing(threshold=3, num_shares=5)
    secret = 12345678901234567890
    shares = sss.share(secret)
    print(f"  Shared secret {secret} into {len(shares)} shares.")
    # Try reconstructing with exactly t shares.
    recovered = sss.reconstruct(shares[:3])
    assert recovered == secret, f"Reconstruction failed: {recovered} != {secret}"
    print(f"  Reconstruct from 3 shares: {recovered}  OK")
    # Try with t-1 shares — should fail.
    try:
        sss.reconstruct(shares[:2])
        print("  ERROR: reconstruction with t-1 shares should have failed!")
    except ValueError:
        print("  Correctly refused to reconstruct from 2 shares (need 3)  OK")

    print("\nAll primitives working.")
