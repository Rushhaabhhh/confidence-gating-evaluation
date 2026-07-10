# Research Proposal — Day 1 Draft
**Target:** GovAI Fellowship Application  
**Word count:** ~900 words  
**Title:** *Calibrated Confidence and Signed Transcripts: A Practical, Low-Cost Mechanism for Meaningful Human Oversight of AI Agents*

---

## The Problem

In 2026, three independent governance frameworks — Singapore's Model AI Governance Framework for Agentic AI (IMDA, May 2026), the NIST National Cybersecurity Center of Excellence concept paper on AI Agent Identity and Authorization (February 2026), and the EU AI Act (Regulation (EU) 2024/1689, Articles 14 & 15) — all arrive at the same requirement: AI agents must be subject to *meaningful human oversight* and must maintain *tamper-evident audit trails* of their actions.

The convergence is striking. The EU AI Act (Art. 14(4)(b)) mandates that high-risk AI systems be designed to counter "the possible tendency of automatically relying or over-relying on the output produced by a high-risk AI system" — what it explicitly terms *automation bias*. Singapore's IMDA framework, independently, lists logging and monitoring as a core agent component and specifies that organizations must "define significant checkpoints in the agentic workflow that require human approval, such as high-stakes or irreversible actions." NIST, in direct language, frames the current state as a failure: "AI agents are commonly treated as generic service accounts without dedicated identity, authorization, or accountability controls" and asks: *"How can we ensure that agents log their actions and intent in a tamper-proof and verifiable manner?"*

None of them say how.

---

## The Research Question

**Can a cheap, model-agnostic mechanism — a calibrated confidence score triggering human review, combined with a cryptographically signed action log — actually satisfy what these three governance frameworks are asking for? And how would we know?**

This question has a technical component (does confidence-based escalation actually improve decision accuracy when humans are inserted at the right threshold?) and a governance component (does the mechanism's design map onto the regulatory language in a verifiable way?). Both need to be answered for the mechanism to be credible.

---

## Background: What Calibration Research Has Established

The machine learning calibration literature provides the empirical foundation. Guo et al. (2017) demonstrated that modern neural networks are systematically overconfident and introduced Expected Calibration Error (ECE) and temperature scaling as the standard measurement and correction toolkit. Tian et al. (2023) extended this to RLHF-fine-tuned large language models, showing that verbalized confidence scores — a number the model states in text output — are often *better calibrated* than internal log-probabilities, reducing ECE by up to 50% without any additional training. Damani et al. (2025/ICLR 2026) showed that reinforcement learning with a Brier-score calibration reward (RLCR) can bake calibration directly into reasoning model training.

Together, these papers establish that calibrated confidence is *measurable, improvable, and accessible via API* — even for deployed commercial models. What the calibration literature has not done is connect these measurements to a governance purpose: how does a well-calibrated confidence score enable the kind of human oversight that regulators are asking for?

---

## What I Have Already Built

To de-risk this fellowship project, I have already built three components:

1. **A calibration measurement pipeline**: end-to-end code that prompts a language model for verbalized confidence, computes ECE and Brier score, and plots reliability diagrams. Based on the Tian et al. methodology.

2. **A signed audit-log prototype**: a short Python script (using only standard-library `hmac` and `hashlib`) that signs each agent action, chains it to the previous entry's hash, and includes a `verify()` function. The schema captures: `step_id, timestamp, action, confidence, oversight_flag, prev_hash, signature`. Every field maps to a specific regulatory requirement: `confidence` and `oversight_flag` address EU Art. 14's automation bias and override provisions; `prev_hash + signature` addresses NIST's tamper-proof non-repudiation requirement.

3. **An RL calibration pilot (HONEST-RL)**: a small, preliminary reinforcement learning experiment that trained small models on calibration-reward objectives similar to RLCR. Results are weak and noisy — consistent with Damani et al.'s observation that calibration-via-RL is harder at small scale. I report this honestly as a preliminary signal, not a finding.

---

## What I Would Do During the Fellowship

The fellowship project would extend this proof-of-concept into a rigorous evaluation:

**Calibration study at scale**: Run a systematic evaluation (200–500 questions across difficulty levels) measuring how well confidence-based human handoff improves decision accuracy. The key figure is a *risk-coverage curve*: if an agent flags its output for human review whenever confidence is below X%, how does accuracy change as X varies? This curve directly answers the governance question — it shows, empirically, the accuracy gain per unit of human oversight work.

**Multi-step agent evaluation**: The current prototype tests single-turn Q&A as a proxy. A fellowship-scale project would extend to multi-step agent trajectories (e.g., code execution, web search tasks), where the calibration and logging challenges are qualitatively harder. This addresses the primary limitation of the current work.

**Framework mapping (multi-jurisdiction)**: Produce a formal comparison across IMDA, NIST, and EU AI Act of which specific regulatory requirements the mechanism satisfies, which it partially addresses, and which remain open. This would be the first structured mapping of a concrete technical mechanism onto all three major 2026 frameworks.

**Engagement with regulators and practitioners**: GovAI's connections to policy and industry stakeholders would enable evaluation of whether the mechanism's design choices match practical deployment constraints — something pure academic work cannot assess.

---

## Honest Limitations

This proposal is for a *small, cheap demonstration*, not a production system. The calibration measurements are on single-turn QA tasks, not real agent workflows. The signed-log prototype uses no production-grade key management. The RL pilot is preliminary. EU AI Act implementation timelines are currently being renegotiated (Digital Omnibus proposal) — I state this honestly rather than presenting a fixed date.

These limitations are real, and I do not hedge them. A fellowship project that honestly acknowledges what it cannot show is more credible than one that overstates its scope — and GovAI's explicit evaluation criterion of "comfort in expressing uncertainty" suggests this is the right stance.

---

## Why This Matters

The gap between regulatory language and technical implementation is not abstract. When the EU AI Act requires that a deployed AI system enable a human to "disregard, override or reverse" its output, someone has to decide — concretely — how to build that. When NIST asks how to make agent logs tamper-proof and verifiable, someone has to answer with running code, not a taxonomy.

This project proposes one specific, reproducible, compute-free answer. It is offered as a starting point, not a finished solution — which is precisely what the field needs in 2026, before the norms solidify around designs that are either impractical (requiring blockchain infrastructure) or absent (no mechanism at all).

---

*Proposal drafted: Day 1, July 2026.*  
*Word count: ~900 words.*
