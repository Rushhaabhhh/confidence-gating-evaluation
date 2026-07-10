#!/usr/bin/env python3
"""
Build a signed, tamper-evident audit log from real model decisions, then prove
it detects tampering.

This turns the paper's second contribution into a runnable artifact:
  1. Load a model's 150 decisions.
  2. Append each as a signed, hash-chained AuditEntry.
  3. verify() the intact chain            -> PASS
  4. Tamper with one entry (flip an oversight flag, as if suppressing a human
     review trigger) and verify() again   -> FAIL, localized to that entry.
  5. Tamper by editing a confidence value  -> FAIL, localized.

Writes audit/results/audit_log_<model>.json (the intact, signed chain).

Usage:
    python audit/build_and_verify_log.py experiments/results/results_llama33_70b.json
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrated_oversight.data import load_results          # noqa: E402
from calibrated_oversight.audit_log import AuditLog         # noqa: E402

# Fixed demo key so the run is reproducible. A real deployment would load this
# from a secret manager and would use asymmetric keys (see module docstring).
DEMO_KEY = b"calibrated-oversight-demo-key-not-for-production"
OVERSIGHT_THRESHOLD = 0.70


def build_log(results) -> AuditLog:
    log = AuditLog(DEMO_KEY, agent_id=results.stem)
    for rec in results.records:
        conf_own = rec.get("confidence_in_own_answer",
                           max(rec["confidence"], 1 - rec["confidence"]))
        log.append(
            action="predict_issue_difficulty",
            raw_input=rec.get("problem_statement", rec["instance_id"]),
            prediction=rec["prediction"],
            confidence=conf_own,
            oversight_flagged=(conf_own < OVERSIGHT_THRESHOLD),
            timestamp=1.0 + rec["step_id"] if "step_id" in rec else None,
        )
    return log


def demo_tamper(log: AuditLog, target: int):
    print(f"\n--- Tamper test: flip oversight_flagged on entry #{target} ---")
    tampered = AuditLog.from_json(log.to_json(), DEMO_KEY, agent_id=log.agent_id)
    before = tampered.entries[target].oversight_flagged
    tampered.entries[target].oversight_flagged = not before
    print(f"    changed oversight_flagged {before} -> {not before} (signature NOT updated)")
    res = tampered.verify()
    status = "PASS (undetected!)" if res["valid"] else "FAIL (detected)"
    print(f"    verify(): {status}  |  {res['reason']}")
    return res


def demo_tamper_confidence(log: AuditLog, target: int):
    print(f"\n--- Tamper test: rewrite confidence on entry #{target} ---")
    tampered = AuditLog.from_json(log.to_json(), DEMO_KEY, agent_id=log.agent_id)
    before = tampered.entries[target].confidence
    tampered.entries[target].confidence = 0.99
    print(f"    changed confidence {before:.3f} -> 0.990 (as if faking high certainty)")
    res = tampered.verify()
    status = "PASS (undetected!)" if res["valid"] else "FAIL (detected)"
    print(f"    verify(): {status}  |  {res['reason']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    r = load_results(args.path)
    print("=" * 60)
    print(f"  Audit log demo: {r.name}  ({r.n_valid} decisions)")
    print("=" * 60)

    log = build_log(r)
    n_flagged = sum(1 for e in log.entries if e.oversight_flagged)
    print(f"  Built signed chain of {len(log.entries)} entries.")
    print(f"  Oversight gate (<{OVERSIGHT_THRESHOLD:.0%}) flagged {n_flagged} entries.")

    intact = log.verify()
    print(f"\n--- Verify intact chain ---")
    print(f"    verify(): {'PASS' if intact['valid'] else 'FAIL'}  |  {intact['reason']}")

    target = min(42, len(log.entries) - 1)
    r1 = demo_tamper(log, target)
    r2 = demo_tamper_confidence(log, target)

    out_dir = ROOT / "audit" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"audit_log_{r.stem}.json"
    with open(out_path, "w") as f:
        f.write(log.to_json())
    print(f"\n-> wrote intact signed chain to {out_path.relative_to(ROOT)}")

    ok = intact["valid"] and (not r1["valid"]) and (not r2["valid"])
    print(f"\nDEMO RESULT: {'ALL CHECKS PASSED' if ok else 'SOMETHING WRONG'} "
          f"(intact verifies, both tampers detected)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
