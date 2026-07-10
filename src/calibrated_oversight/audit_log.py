"""
Tamper-evident, signed audit log for agent decisions.

This is the paper's second contribution: a concrete, ~standard-library answer to
the audit-trail requirement that Singapore's MGF, the NIST NCCoE concept paper,
and EU AI Act Art. 14 all state but none technically specify.

Design (deliberately minimal, dependency-light):
  * Each entry records one agent decision: what was asked, what was answered, the
    model's calibrated confidence, and whether the oversight gate fired.
  * Entries are hash-chained: each carries prev_hash = SHA-256 of the previous
    entry's canonical bytes. Altering any past entry changes its hash and breaks
    every subsequent link -> tamper-evident (cf. Certificate Transparency,
    Crosby & Wallach 2009 tamper-evident logging).
  * Each entry is signed with HMAC-SHA256 under a secret key -> an attacker who
    cannot access the key cannot forge or silently rewrite entries (non-repudiation
    in the sense NIST's concept paper asks for).

Scope / honesty: this is a working prototype, not a production system. HMAC gives
symmetric authentication (anyone who can verify can also sign); a production system
would use asymmetric signatures (e.g. Ed25519) and real key management / rotation.
We say this plainly in the paper. What the prototype does demonstrate -- append
chaining + signed entries + a verifier that localizes tampering to a specific
entry -- is exactly the mechanism the frameworks describe.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, asdict, field
from typing import Optional


def _canonical(obj) -> bytes:
    """Deterministic JSON encoding so hashing/signing is stable across machines."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class AuditEntry:
    step_id: int
    timestamp: float
    agent_id: str
    action: str                 # what the agent did (e.g. "predict_difficulty")
    input_hash: str             # SHA-256 of the input, not the raw input (privacy)
    prediction: int
    confidence: float           # calibrated p(prediction correct)
    oversight_flagged: bool     # did the confidence gate route this to a human?
    prev_hash: str
    signature: str = ""         # HMAC-SHA256, filled in by AuditLog.append

    def signable_dict(self) -> dict:
        d = asdict(self)
        d.pop("signature", None)
        return d

    def entry_hash(self) -> str:
        """Hash of the full signed entry, used as prev_hash for the next entry."""
        return _sha256_hex(_canonical(asdict(self)))


class AuditLog:
    def __init__(self, secret_key: bytes, agent_id: str = "agent-0"):
        self._key = secret_key
        self.agent_id = agent_id
        self.entries: list[AuditEntry] = []

    # -- construction -------------------------------------------------------
    def _sign(self, entry: AuditEntry) -> str:
        return hmac.new(self._key, _canonical(entry.signable_dict()), hashlib.sha256).hexdigest()

    def append(self, action: str, raw_input: str, prediction: int,
               confidence: float, oversight_flagged: bool,
               timestamp: Optional[float] = None) -> AuditEntry:
        prev_hash = self.entries[-1].entry_hash() if self.entries else "0" * 64
        entry = AuditEntry(
            step_id=len(self.entries),
            timestamp=timestamp if timestamp is not None else time.time(),
            agent_id=self.agent_id,
            action=action,
            input_hash=_sha256_hex(raw_input.encode("utf-8")),
            prediction=int(prediction),
            confidence=float(confidence),
            oversight_flagged=bool(oversight_flagged),
            prev_hash=prev_hash,
        )
        entry.signature = self._sign(entry)
        self.entries.append(entry)
        return entry

    # -- verification -------------------------------------------------------
    def verify(self) -> dict:
        """
        Re-walk the chain. Returns {'valid': bool, 'broken_at': step or None,
        'reason': str}. Localizes the first tampered entry.
        """
        prev_hash = "0" * 64
        for e in self.entries:
            # 1. chain link intact?
            if e.prev_hash != prev_hash:
                return {"valid": False, "broken_at": e.step_id,
                        "reason": f"prev_hash mismatch at step {e.step_id} "
                                  f"(chain broken -- an earlier entry was altered)"}
            # 2. signature valid?
            expected = self._sign(e)
            if not hmac.compare_digest(expected, e.signature):
                return {"valid": False, "broken_at": e.step_id,
                        "reason": f"signature mismatch at step {e.step_id} "
                                  f"(entry contents were altered after signing)"}
            prev_hash = e.entry_hash()
        return {"valid": True, "broken_at": None, "reason": "all entries verified"}

    # -- serialization ------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self.entries], indent=2)

    @classmethod
    def from_json(cls, data: str, secret_key: bytes, agent_id: str = "agent-0") -> "AuditLog":
        log = cls(secret_key, agent_id=agent_id)
        for d in json.loads(data):
            log.entries.append(AuditEntry(**d))
        return log
