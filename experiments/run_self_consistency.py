#!/usr/bin/env python3
"""
Self-consistency baseline experiment.

For each item in an existing result file, query the model 5x at temperature=0.3
and compute agreement rate as an alternative uncertainty estimate.

Agreement rate = (count of most-common answer) / n_samples

Usage:
  python experiments/run_self_consistency.py \
    --source-file experiments/results/results_llama33_70b_multidomain.json \
    --domains swebench gsm8k \
    --backend groq \
    --results-file experiments/results/results_llama33_70b_selfconsistency.json
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Reuse prompt infrastructure from run_elicitation
import importlib.util, importlib.machinery
_spec = importlib.util.spec_from_file_location("elicit", ROOT / "experiments" / "run_elicitation.py")
_elicit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_elicit)

SLEEP      = _elicit.SLEEP
KEY_VARS   = _elicit.KEY_VARS
SYS_PROMPTS= _elicit.SYSTEM_PROMPTS
DEFAULT_MODELS = _elicit.DEFAULT_MODELS


def sample_once(backend, model, api_key, domain, prompt, temperature=0.3, max_tokens=250, retries=5):
    system = SYS_PROMPTS[domain]
    for attempt in range(retries):
        wait = min(2 ** (attempt + 1), 60)
        try:
            if backend == "gemini":
                from google import genai as _g
                from google.genai import types as _gt
                c = _g.Client(api_key=api_key)
                return c.models.generate_content(
                    model=model,
                    config=_gt.GenerateContentConfig(
                        system_instruction=system, max_output_tokens=max_tokens, temperature=temperature),
                    contents=prompt).text.strip()
            
            import requests
            urls = {"groq":       "https://api.groq.com/openai/v1/chat/completions",
                    "nim":        "https://integrate.api.nvidia.com/v1/chat/completions",
                    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
                    "openai":     "https://api.openai.com/v1/chat/completions"}
            hdrs = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            if backend == "openrouter":
                hdrs["HTTP-Referer"] = "https://github.com/rushhaabhhh/calibrated-oversight"
                hdrs["X-Title"]      = "Calibration Study"

            payload = {"model": model,
                       "messages": [{"role":"system","content":system},
                                    {"role":"user","content":prompt}],
                       "temperature": temperature, "max_tokens": max_tokens}

            # Single non-streaming request for all backends — 120s timeout.
            r = requests.post(urls[backend], json=payload, headers=hdrs, timeout=120)

            if r.status_code != 200:
                try:
                    err_msg = r.json().get("error", {}).get("message", r.text)
                except Exception:
                    err_msg = r.text
                if r.status_code == 429 or r.status_code >= 500:
                    wait_rl = max(wait, 30 if backend == "nim" else 20)
                    print(f"    [HTTP {r.status_code}] Rate limit or overload hit. Waiting {wait_rl}s...")
                    time.sleep(wait_rl)
                    continue
                else:
                    r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            if attempt == retries - 1:
                raise
            err_str = str(e).lower()
            is_rate_limit = (
                "resourceexhausted" in err_str or
                "429" in err_str or
                "quota" in err_str or
                "limit" in err_str or
                "timeout" in err_str or
                "connection" in err_str or
                isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
            )
            cooldown = 45 if backend == "nim" else 20
            wait_time = max(wait, cooldown) if is_rate_limit else wait
            print(f"    Error ({type(e).__name__}): {e} — retry {attempt+1} in {wait_time}s")
            time.sleep(wait_time)


def parse_answer_only(raw, domain):
    ans_m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    if not ans_m:
        return None
    ans = ans_m.group(1).strip()
    if domain == "swebench":
        return "yes" if re.search(r"\byes\b", ans, re.IGNORECASE) else "no"
    elif domain == "gsm8k":
        nums = re.findall(r"-?\d+(?:[.,]\d+)?", ans.replace(",",""))
        return nums[-1] if nums else None
    elif domain in ("mmlu", "truthfulqa"):
        m = re.search(r"\b([A-E])\b", ans)
        return m.group(1).upper() if m else ans[:30].lower()
    return None


def run(args):
    model = args.model or DEFAULT_MODELS[args.backend]
    api_key = os.environ.get(KEY_VARS[args.backend])
    if not api_key:
        sys.exit(f"ERROR: {KEY_VARS[args.backend]} not set.")

    source = json.load(open(args.source_file, encoding="utf-8"))
    todo_items = [r for r in source
                  if not r.get("parse_error") and r.get("domain") in args.domains]

    out_path = Path(args.results_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing, done_ids = [], set()
    if out_path.exists() and not args.fresh:
        try:
            existing = json.load(open(out_path))
            done_ids = {r["instance_id"] for r in existing}
            print(f"Resuming: {len(done_ids)} already done.")
        except Exception:
            pass

    todo = [r for r in todo_items if r["instance_id"] not in done_ids]
    print(f"\n{'='*60}")
    print(f"  Self-Consistency  |  {args.backend.upper()} / {model}")
    print(f"  Domains: {', '.join(args.domains)}  |  Items: {len(todo)}  |  Samples/item: {args.n_samples}")
    print(f"  ETA: ~{len(todo) * SLEEP.get(args.backend,2.0) * args.n_samples / 60:.0f} min")
    print(f"{'='*60}\n")

    results = list(existing)
    for i, src in enumerate(todo):
        domain = src["domain"]
        # Reconstruct keys for build_prompt compatibility (since serialized records use 'question')
        item = {
            "domain": domain,
            "instance_id": src["instance_id"],
            "text": src.get("question", src.get("text", "")),
            "repo": src.get("repo") or src.get("domain_meta", {}).get("repo", ""),
        }
        prompt = _elicit.build_prompt(item)
        answers = []
        for s in range(args.n_samples):
            try:
                raw = sample_once(args.backend, model, api_key, domain, prompt, args.temperature)
                answers.append(parse_answer_only(raw, domain))
            except Exception as e:
                print(f"    sample {s+1} failed: {e}")
                answers.append(None)
            time.sleep(SLEEP.get(args.backend, 2.0))

        valid_ans = [a for a in answers if a is not None]
        sc_conf = (Counter(valid_ans).most_common(1)[0][1] / len(valid_ans)) if valid_ans else None

        rec = {
            "instance_id": src["instance_id"], "domain": domain,
            "model_id": model, "backend": args.backend,
            "n_samples": args.n_samples, "temperature": args.temperature,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "answers": answers,
            "self_consistency_confidence": sc_conf,
            "prediction": src.get("prediction"), "label": src.get("label"),
            "verbal_confidence_in_own": src.get("confidence_in_own_answer"),
            "parse_error": sc_conf is None,
        }
        results.append(rec)
        mode = Counter(valid_ans).most_common(1)[0][0] if valid_ans else "?"
        sc_str = f"{sc_conf:.2f}" if sc_conf is not None else "N/A"
        print(f"  [{i+1:>4}/{len(todo)}] {domain:<12} sc={sc_str}"
              f"  mode={mode}  {src['instance_id'][:35]}", flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone: {len(results)} | Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-file",
        default="experiments/results/results_llama33_70b_multidomain.json")
    ap.add_argument("--domains", nargs="+",
        choices=["swebench","gsm8k","mmlu","truthfulqa"],
        default=["swebench","gsm8k"])
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--backend", choices=list(DEFAULT_MODELS), default="groq")
    ap.add_argument("--model", default="")
    ap.add_argument("--results-file",
        default="experiments/results/results_llama33_70b_selfconsistency.json")
    ap.add_argument("--fresh", action="store_true")
    run(ap.parse_args())

if __name__ == "__main__":
    main()
