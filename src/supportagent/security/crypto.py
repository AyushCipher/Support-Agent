"""Optional-crypto helpers shared by identity, anchors and the privacy layer.

The core stays dependency-free (ADR-0001). Anything that needs a real asymmetric signature or an AEAD cipher imports the
optional `cryptography` package through this module, which raises one clear, actionable error when it is missing:

    pip install "supportagent[crypto]"

Also holds the at-rest protection used for PII (ADR-0011):
  * `SA_VAULT_KEY` (any string, ideally 32+ random bytes) -> scrypt KDF -> master key (KEK).
  * one random 256-bit data key (DEK) per data subject, wrapped by the KEK (AES-256-GCM) in `<key_dir>/<subject>.key`.
  * every sensitive value is sealed with AES-256-GCM under the subject's DEK, with the value's location bound as AAD.
  * crypto-shredding = deleting the subject's wrapped DEK. Ciphertext left in backups, WAL files or replicas is then
    permanently undecipherable even to someone who holds the master key.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from pathlib import Path

INSTALL_HINT = 'install the optional crypto extra: pip install "supportagent[crypto]"'
PREFIX = "enc:v1:"


class CryptoUnavailable(RuntimeError):
    """The `cryptography` package is not installed but a feature that needs it was requested."""


class KeyShredded(Exception):
    """The data key for this subject was destroyed (crypto-shredding) or never existed: the data is unrecoverable."""


class VaultKeyMissing(RuntimeError):
    """Production mode requires SA_VAULT_KEY; refusing to run with unencrypted PII (fail closed)."""


def crypto_available() -> bool:
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        return False


def require_crypto(feature: str = "this feature") -> None:
    if not crypto_available():
        raise CryptoUnavailable(f"{feature} needs the `cryptography` package; {INSTALL_HINT}")


def b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def production_mode() -> bool:
    return os.getenv("SA_ENV", "").lower() in ("prod", "production") or os.getenv("SA_PRODUCTION") == "1"


# ---- AES-GCM -------------------------------------------------------------------------------------
def aead_seal(key: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
    require_crypto("AES-GCM encryption")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = secrets.token_bytes(12)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def aead_open(key: bytes, blob: bytes, aad: bytes = b"") -> bytes:
    require_crypto("AES-GCM decryption")
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    try:
        return AESGCM(key).decrypt(blob[:12], blob[12:], aad)
    except InvalidTag:
        raise ValueError("decryption failed: wrong key, wrong context or tampered ciphertext") from None


# ---- KDF -----------------------------------------------------------------------------------------
def derive_master_key(passphrase: str, salt: bytes) -> bytes:
    """scrypt (N=2^15, r=8, p=1, ~100ms) from the stdlib; PBKDF2-HMAC-SHA256 (600k) only if scrypt is not compiled in."""
    try:
        return hashlib.scrypt(passphrase.encode(), salt=salt, n=2 ** 15, r=8, p=1, maxmem=128 * 1024 * 1024, dklen=32)
    except (AttributeError, ValueError):                       # pragma: no cover - OpenSSL without scrypt
        return hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 600_000, dklen=32)


def subject_for(customer_id: str = "", email: str = "") -> str:
    """Stable, filesystem-safe key-store subject. Customer ids are used as-is; emails are hashed (never a filename)."""
    if customer_id:
        return "c-" + re.sub(r"[^A-Za-z0-9_.-]", "_", customer_id)[:64]
    return "e-" + hashlib.sha256((email or "").strip().lower().encode()).hexdigest()[:32]


class KeyStore:
    """One wrapped DEK file per subject. File-per-subject so deletion is atomic, per-subject and cross-process safe."""

    def __init__(self, directory: str | os.PathLike, kek: bytes, ttl_s: float = 5.0):
        self.dir, self.kek, self.ttl = Path(directory), kek, ttl_s
        self._cache: dict[str, tuple[bytes, float]] = {}
        self._lock = threading.Lock()

    def _path(self, subject: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", subject):
            raise ValueError("invalid key subject")
        return self.dir / f"{subject}.key"

    def get(self, subject: str, create: bool) -> bytes:
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(subject)
            if hit and now - hit[1] < self.ttl:
                return hit[0]
            p = self._path(subject)
            if not p.exists():
                self._cache.pop(subject, None)
                if not create:
                    raise KeyShredded(subject)
                self.dir.mkdir(parents=True, exist_ok=True)
                dek = secrets.token_bytes(32)
                wrapped = b64(aead_seal(self.kek, dek, f"dek:{subject}".encode()))
                try:
                    with open(p, "x") as f:                    # O_EXCL: two processes cannot create different keys
                        f.write(wrapped)
                except FileExistsError:
                    return self.get(subject, create)
                try:
                    os.chmod(p, 0o600)
                except OSError:                                # pragma: no cover
                    pass
            dek = aead_open(self.kek, unb64(p.read_text().strip()), f"dek:{subject}".encode())
            self._cache[subject] = (dek, now)
            return dek

    def has(self, subject: str) -> bool:
        return self._path(subject).exists()

    def shred(self, subject: str) -> bool:
        """Overwrite then delete the wrapped DEK. Returns True if a key existed."""
        with self._lock:
            self._cache.pop(subject, None)
            p = self._path(subject)
            if not p.exists():
                return False
            try:
                with open(p, "r+b") as f:
                    n = f.seek(0, 2)
                    f.seek(0)
                    f.write(secrets.token_bytes(max(n, 64)))
                    f.flush()
                    os.fsync(f.fileno())
            except OSError:                                    # pragma: no cover
                pass
            p.unlink()
            return True


class Protector:
    """Seals/opens individual values for a subject. `seal`/`open` use the subject's DEK; AAD binds a value to its location."""

    def __init__(self, keystore: KeyStore):
        self.keys = keystore

    @classmethod
    def from_env(cls, key_dir: str | os.PathLike | None = None, passphrase: str | None = None) -> "Protector | None":
        """None when no key is configured in non-production mode (plaintext, as before). Fails closed in production."""
        pw = passphrase if passphrase is not None else os.getenv("SA_VAULT_KEY", "")
        if not pw:
            if production_mode():
                raise VaultKeyMissing("SA_VAULT_KEY is required when SA_ENV=production: refusing to store PII unencrypted")
            return None
        require_crypto("PII encryption at rest (SA_VAULT_KEY is set)")
        d = Path(key_dir or os.getenv("SA_KEY_DIR") or _default_key_dir())
        d.mkdir(parents=True, exist_ok=True)
        salt_file = d / "master.salt"
        if not salt_file.exists():
            try:
                with open(salt_file, "x") as f:
                    f.write(b64(secrets.token_bytes(16)))
            except FileExistsError:
                pass
        kek = _kek_cached(pw, salt_file.read_text().strip())
        return cls(KeyStore(d, kek))

    def seal(self, subject: str, plaintext: str, aad: str = "") -> str:
        dek = self.keys.get(subject, create=True)
        return PREFIX + b64(aead_seal(dek, plaintext.encode(), f"{subject}|{aad}".encode()))

    def open(self, subject: str, sealed: str, aad: str = "") -> str:
        if not sealed.startswith(PREFIX):
            return sealed                                      # legacy plaintext value (pre-encryption data)
        dek = self.keys.get(subject, create=False)
        return aead_open(dek, unb64(sealed[len(PREFIX):]), f"{subject}|{aad}".encode()).decode()

    def blind_index(self, subject: str, value: str) -> str:
        """Keyed hash so the vault can de-duplicate values without storing them (rev index)."""
        dek = self.keys.get(subject, create=True)
        return hmac.new(dek, b"blind|" + value.encode(), hashlib.sha256).hexdigest()[:32]

    def shred(self, subject: str) -> bool:
        return self.keys.shred(subject)


_KEK_CACHE: dict[tuple[str, str], bytes] = {}


def _kek_cached(pw: str, salt_b64: str) -> bytes:
    k = (hashlib.sha256(pw.encode()).hexdigest(), salt_b64)
    if k not in _KEK_CACHE:
        _KEK_CACHE[k] = derive_master_key(pw, unb64(salt_b64))
    return _KEK_CACHE[k]


def _default_key_dir() -> Path:
    from ..config import RUNS_DIR
    return Path(RUNS_DIR) / "keys"


def default_protector() -> "Protector | None":
    """Process-wide default derived from the environment, or None (plaintext dev mode). Raises in production without a key."""
    pw = os.getenv("SA_VAULT_KEY", "")
    if not pw:
        return Protector.from_env()                            # None in dev, VaultKeyMissing in production
    d = str(Path(os.getenv("SA_KEY_DIR") or _default_key_dir()))
    key = (hashlib.sha256(pw.encode()).hexdigest(), d)
    with _PROTECTOR_LOCK:
        if key not in _PROTECTORS:
            _PROTECTORS[key] = Protector.from_env(d, pw)
        return _PROTECTORS[key]


_PROTECTORS: dict[tuple[str, str], Protector] = {}
_PROTECTOR_LOCK = threading.Lock()
