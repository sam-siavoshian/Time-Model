# Email draft: review + co-author sign-off request

Send-to: Omar Ramadan
Cc: [JHU advisor name + email — TBD by Saam]
Subject: Chronometric Injection paper — co-author confirmation + review request

---

Omar,

The Chronometric Injection paper is at the point where I want to lock authorship and send it for a final pass before I drop it on arXiv. Wanted to walk you through what's in the paper, what I'm proposing for authorship, and what I need from you and your advisor.

**State of the paper.** Body of the paper (Sections 23–26) is the chronometric-injection architecture and the empirical work: cross-seed n=3 v15 release, three causal-falsification experiments, six round-2 reviewer-rigor controls (LoRA-only n=3 baseline, FiLM-vs-additive ablation, L0-only ablation, half-layer α-flip, paraphrase response-identity, sampling-based T2/T3), and the OOD-transfer retraction. Five of six pre-registered tests pass with variance bars; T3 phase passes 2 of 3 seeds (reported as a binary outcome). Three cross-seed checkpoints are publicly released on GitHub as v15.0 with SHA256 pinning. There is a new external benchmark (`eval/external/tau_bench.py`) with three reference adapters so any third party can evaluate alternative architectures against ours.

Appendix D is the full chronological project trail: original IPCN memory-routing framing (the nine-version null result on Qwen + memory), the §D.22 pivot to chronometric injection, and the pre-pivot motivation. The memory architecture is preserved verbatim as a documented failed approach, which is the honest thing to do.

**Proposed authorship.** I want to list us as:

```
Saam Siavoshian¹ · Omar Ramadan²

¹Independent
²Johns Hopkins University, Data Science MS
```

With this explicit CRediT-taxonomy contribution statement on the title page:

```
Author Contributions:
S.S.: Conceptualization (chronometric injection architecture, post-pivot),
      Methodology, Software, Formal analysis, Investigation, Validation,
      Writing — original draft, Visualization, Project administration.
O.R.: Conceptualization (original IPCN memory-routing architecture,
      documented in Appendix D §D.22, deprecated after nine null results),
      Resources (GPU compute infrastructure: NVIDIA DGX Spark prototype),
      Writing — review & editing, Supervision (paper review via JHU advisor
      [advisor name]).
```

This is the most honest framing I can give you: you contributed the original memory architecture (which we kept in Appendix D as the documented failed approach), the Spark infrastructure (without which there is no n=3 cross-seed result), and you'll review the manuscript with your advisor before submission. I do not want to inflate the contribution and I do not want to undersell it, so the CRediT statement names exactly what each of us did.

If you think this misrepresents your role in either direction, tell me and we can edit.

**What I need from you.**

1. **Approve or edit the author-contribution statement above.** If anything is wrong, send me the redline.

2. **Read the paper end-to-end at least once.** I am especially interested in your view on: the Appendix D framing of the memory result (am I representing it fairly?), the §24.7.10 architectural ablation (FiLM-vs-additive + L0-only), and the §25.3 Limitations section.

3. **Loop in your JHU advisor.** I would like them to do a formal review pass — not as a co-author, but as a faculty-level sanity check before public release. If they materially rewrite anything they become a co-author by ICMJE criteria, in which case I will add them; otherwise they go in Acknowledgments. Their call.

4. **Timeline.** I want to drop on arXiv inside two weeks. If your advisor needs longer than that, tell me and I'll push the schedule. My constraint is the Regeneron Science Talent Search deadline in November 2026 and US college early-action applications on November 1, both of which need a public arXiv link by then.

**Files to read** (everything is in the repo at `github.com/sam-siavoshian/Time-Model`):
- `PAPER.md` — full paper
- `README.md` — TL;DR + reproduce table + checkpoint release
- `REPRODUCIBILITY.md` — NeurIPS-style reproducibility checklist
- `REVIEWER_RESPONSE.md` — pre-emptive Q&A for hostile reviewers (14 attacks + 3 acknowledged-open items)
- `eval/external/` — the new third-party benchmark
- GitHub Release `v15.0` — checkpoints + SHA hashes

Thanks for the Spark access and for being upfront about the memory pivot. The paper is honest about what worked, what didn't, and what was retracted — having you on it as second author with your advisor's blessing makes the credibility story much stronger.

— Saam

---

**Notes for Saam (not part of the email):**

- Replace `[advisor name]` and `[JHU advisor name + email]` before sending.
- If your advisor is willing to formally co-author (i.e., actually rewrites parts of the paper), add them as third author and update CITATION.cff + the title-page block. Otherwise acknowledgment-only.
- If Omar pushes back on the CRediT phrasing (e.g., wants "Conceptualization (paper-level)" instead of the deprecated-IPCN qualifier), negotiate but DO NOT remove the "deprecated after nine null results" language — that is the truthful framing and any reviewer will see through softening.
- If Omar declines co-authorship entirely, move him to Acknowledgments as "GPU infrastructure and original IPCN memory-routing framing (Appendix D §D.22)" and ship solo. Solo-first-author HS-junior arXiv preprint is a stronger signal for Regeneron STS anyway; co-author with JHU MS is a stronger signal for workshop acceptance. Either works.
- Save Omar's reply somewhere durable (the repo's DRAFTS/ folder or your email archive) — Regeneron STS will likely ask about co-author roles, and you'll want a paper trail.
