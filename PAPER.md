# IPCN: Involuntary Prefix Consolidation Network

A neural network that mixes stored memories and real elapsed time into the start of every computation, and slowly turns frequently used memories into permanent parts of the network itself.

Author: Saam Siavoshian
Date: 2026-05-12
Status: research spec, pre-implementation

---

## Section 1: What is time?

**Newton:** universal, ticks the same everywhere.

**Plain definition:** the continuous progression of events from the past, through the present, and into the future.

**Operational definition for this paper:** real elapsed seconds τ that supports four things:
1. Causal ordering (A before B means A can influence B, not reverse)
2. Duration measurement (Δτ = how much elapsed)
3. Multi-scale phase (same Δτ can mean different things depending on cycle: day, week, year)
4. Persistence under no-input (τ keeps advancing even with no observations)

Sidesteps qualia. Captures what cognition needs.

---

## Section 2: What is this paper?

**One-sentence thesis:**

> A neural network architecture where persistent memory and real elapsed time are injected into the hidden state before the first layer runs, and where heavily-used memories migrate from external slots into the model's weights.

Three things in that sentence:
1. **Pre-computational memory** — memory enters BEFORE the model thinks, not after
2. **Real elapsed time as substrate** — actual seconds wired into the architecture, not just words like "yesterday"
3. **Consolidation** — used memories migrate from scratchpad to network weights

---

## Section 3: Why memories?

If you remember nothing, you can't notice time passed. The "before" must be stored somewhere to compare against "now."

A creature with no memory has no time. Only an eternal present.

Memory and time are one system, not two. To use time, model must compare current state to past states (= memory). To use memory, model must know when things happened (= time).

---

## Section 4: Baseline LLM vs IPCN

**Old approach (fine-tune on 10k task-duration pairs):**
The model does not HAVE time. It has duration estimates trained into its next-token guesses. Pattern matching trick.

**IPCN:**
Time wired into the mechanism. Memory evolves based on real elapsed seconds. Same tokens with different Δτ produce different memory states and different outputs.

| | Baseline LLM | IPCN |
|---|---|---|
| Memory | context window only, dies between sessions | persistent slots + welded weights |
| When memory enters | after thinking starts (retrieval) | before thinking starts (prefix) |
| Time | token positions only | real elapsed seconds (chronometric state) |
| Silent gaps | invisible to model | memory still evolves on Δτ |
| Learning | training set only | online consolidation from usage |

---

## Section 5: What LLMs can do today

| Capability | Status |
|---|---|
| Date arithmetic ("30 days after March 15") | Mostly works, errors at edges |
| Temporal commonsense ("how long to brush teeth?") | Reasonable estimates, from training text |
| Text-event ordering ("did A happen before B in this passage?") | Works if signal is in text |
| Hypothetical duration arithmetic ("noon + 30 min = ?") | Works as symbolic math |
| Facts with training-time timestamps | Frozen at training cutoff |

## Section 6: What LLMs CANNOT do

| Capability | Status |
|---|---|
| Know current real time | No, unless told via context/tool |
| Self-awareness (infra, tokens/sec, training cutoff) | No |
| Detect silent gaps between messages | No, model is not awake in the gap |
| Felt duration / experiential time | No, and probably substrate-locked |

**The fundamental architecture issue:**

LLMs have zero notion of time during inference. Token position is the only "time" they have. Position 5 comes after position 4. That's it.

A model that takes 10 seconds and one that takes 10 minutes have the SAME internal timeline (same token positions). Architecture cannot tell fast from slow.

Their "knowledge" of time = statistical patterns from training text. "After Monday comes Tuesday" because that string appeared a million times in training. Not because they track anything.

Even chain-of-thought reasoning about durations is symbolic arithmetic on tokens. Numbers in, numbers out. No clock anywhere.

---

## Section 7: The Gaps

| Gap | What's missing |
|---|---|
| **No clock** | Model has no τ. Position ≠ time |
| **No silent-gap awareness** | Δτ between inputs invisible to architecture |
| **No self-rate awareness** | Model doesn't know how fast it generates |
| **No behavioral pressure response** | "You have 5 seconds" only changes the prompt, not behavior |

---

## Section 8: How IPCN closes these gaps

### 8.1 Chronometric encoding (the math, plain)

Real seconds τ → an 82-number vector. Formula at each timescale T_b:

```
ψ(τ, T_b) = [log(1+τ), sin(2π·τ/T_b), cos(2π·τ/T_b)]
```

**Number 1: log(1+τ)** — compresses range.
| τ | log(1+τ) |
|---|---|
| 1 sec | 0.69 |
| 60 sec (1 min) | 4.1 |
| 3600 sec (1 hour) | 8.2 |
| 86400 sec (1 day) | 11.4 |
| 31,536,000 sec (1 year) | 17.3 |

Effect: model compares 1-second-ago and 1-year-ago using similar-magnitude inputs. Networks need this; raw seconds explode.

**Numbers 2 and 3: sin and cos** — put time on a clock.
- At each timescale T_b, hand sweeps once per T_b units
- (cos, sin) = unique position on the unit circle for current phase
- sin alone is ambiguous (sin(π/4) = sin(3π/4)); cos breaks the tie

**13 different scales:** T_b ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536}

Each scale = a clock running at different speed. Model reads all 13 at once. Triangulates real time AND detects cycles at any scale.

Same trick as positional encoding in transformers (Vaswani 2017). Proven math. Novelty here = applying it to real time, not token position.

**Full chronometric vector χ_t** (82 numbers):
- Absolute time τ_t (1 number)
- Elapsed Δτ_t since last update (1 number)
- ψ(τ_t) at 13 scales = 39 numbers
- ψ(Δτ_t) at 13 scales = 39 numbers
- Event density ν_t (1 number)
- Silent-gap flag gap_t (1 number)

### 8.2 What's NOT magic, what is hard

**Not magic:** the encoding itself is standard feature engineering. Sinusoidal multi-scale is well-trodden.

**What's hard:**
1. Getting the model to USE χ_t (chronometric loss forces it during training)
2. Training data with non-trivial time structure (synthetic Temporal Latent World)
3. Memory evolution actually load-bearing on Δτ (tuning ε_dyn carefully)
4. Falsifying — built-in test: ablate Δτ, check if accuracy drops on duration tasks

### 8.3 Closes Gap 1 (no clock)

Model now has 82 numbers describing real τ every chunk. Mathematically sufficient to reconstruct real time. Model has the data. That's not magic, that's giving the model the input.

### 8.4 Closes Gap 2 (silent gaps)

Δτ encoded and consumed by three places:
1. **Prep cook attention:** slot temporal signatures compared against current χ_t
2. **Memory evolution:** `v_{t+1} = LN(v_t + ε · Δτ · Δv)` — bigger Δτ drives bigger memory change
3. **Temporal state z:** running vibe vector updated with χ_t

So 5-min silent gap is NOT invisible. Memory drifts, retrieval weights shift, vibe state updates.

### 8.5 Closes Gap 3 (self-rate awareness) — via ablation cell

Token-rate fact injection: add `tokens_per_real_second` as a feature in χ_t. Model learns own throughput. One of original 4 mechanisms, now an ablation cell.

### 8.6 Closes Gap 4 (behavioral pressure response) — via training data

Architecture supports it; needs targeted training scenarios with deadline pressure + simulated Δτ + reward for shorter outputs that hit accuracy.

---

## Section 9: Anatomy of IPCN (5 parts)

| Part | Job | Size |
|---|---|---|
| Brain (main transformer) | does thinking | ~50M params, 8 layers, width 512, 8 heads |
| Prep cook (PFC) | builds prefix from memory | ~5M params, 2 layers |
| Memory bank | 256 slots × (key + value + metadata) | ~200K floats, persistent across chunks |
| LoRA adapters | weight tweaks for welded memories | ~1-2M params, rank 8 |
| Time encoder | real seconds → vector | 0 params, deterministic |

Plus temporal state z (128-dim running vibe).

Five parts run together. Brain NEVER sees neutral input. Always prefixed.

---

## Section 10: Three storage tiers

| Tier | Data | Size | Lives for | Location |
|---|---|---|---|---|
| **Whisper (prefix P_t)** | 32 × 512 floats | 16K numbers | one forward pass | GPU compute, ephemeral |
| **Scratchpad (memory bank M_t)** | 256 slots × (256-dim key + 256-dim value + metadata) | ~200K floats | many chunks, updated | GPU global, persistent |
| **Welded (LoRA adapters Ω_t)** | rank-8 weight tweaks on PFC + brain layers 0-2 | ~1-2M params | permanent (with rollback) | next to base weights |

Zero text. Zero system prompt. All numbers in three different patches of GPU memory.

---

## Section 11: How consolidation works

**Sequence for slot 47:**

1. Slot's usage counter u_47 crosses threshold (κ > 3.0) + stable for 512+ chunks
2. Collect replay buffer = past contexts where slot 47 helped
3. For each context, run two versions:
   - **Teacher** = full model with slot 47 present → P_T distribution
   - **Student** = model with slot 47 REMOVED but LoRA trainable → P_S distribution
4. Train LoRA via KL(P_T || P_S). Student learns to match teacher without slot 47.
5. Validation gates (all must pass):
   - Held-out accuracy drop < 2%
   - Contradiction-handling drop < 1%
   - General perplexity drift < 5%
6. All pass → attenuate slot 47, free its space for new memory
7. Any fail → rollback LoRA, slot stays

What "moved" = the BEHAVIOR. Now lives in LoRA weights, not slot.

**Why this matters:** continual learning with mechanical rollback. Existing consolidation can't undo. IPCN can.

---

## Section 12: Multi-user safety (CRITICAL)

Personal memories CANNOT go into shared base weights.

| Tier | Who owns it | Example |
|---|---|---|
| Base model weights | shared by everyone | grammar, math, world knowledge |
| Per-user LoRA | one user | "Sam likes green, prefers Bun" |
| Per-user memory bank | one user | recent conversation, current project |

When user logs in:
1. Load base model (shared, frozen)
2. Load YOUR LoRA stack (your file)
3. Load YOUR memory bank (your slots)

When another user logs in: same base, their LoRA, their bank.

Base weights never change. Personal stuff lives in user-scoped tiers. Architecture firewall.

Universal patterns only enter base weights via OFFLINE aggregated training. Separate mechanism, out of IPCN scope.

---

## Section 13: Prior work scan (done 2026-05-12)

**4 Nia scans run. All findings verified, real arXiv IDs.**

### 13.1 Direct LLM time-experience work (5 must-cite)

| Paper | arXiv | Year | What they do | Relation to IPCN |
|---|---|---|---|---|
| "Can LLMs Perceive Time?" Garikaparthi (TCS) | 2604.00010 | ICLR 2026 Workshop | 68 tasks, pre-task estimates overshoot 4-7×, multi-step errors 5-10× | **Closest motivation overlap.** Diagnoses our gap. Quote: "models possess propositional knowledge about duration from training but lack experiential grounding in their own inference time." |
| "Discrete Minds in a Continuous World" Wang et al (Monash) | 2506.05790 | EMNLP 2025 Findings | Coins "Token-Time Hypothesis" — LLMs use token count as wall-clock proxy. BombRush task. | **Strongest conceptual antecedent.** Shows proxy works partially. IPCN replaces proxy with substrate. |
| "Your LLM Agents are Temporally Blind" Cheng et al | 2510.23853 | Oct 2025 | Coins "temporal blindness." 76 scenarios. No model >65% alignment. | Direct adjacency. Diagnostic, not architectural. |
| "Timely Machine" Ma et al | 2601.16486 | Jan 2026 | Wall-clock time as test-time scaling. Timely-Eval + Timely-RL. | **HIGHEST CONCURRENT RISK.** Same "wall-clock as first-class" insight. Their scope: single-loop decode budget. IPCN scope: persistent memory + cross-session. |
| "LLMs Report Subjective Experience" Berg et al | 2510.24797 | Oct 2025 | GPT/Claude/Gemini produce first-person experience reports under self-referential prompting. | **Boundary risk.** IPCN must NOT be conflated with consciousness claims. |

### 13.2 Memory-augmented architectures (8 catalogued)

| Architecture | arXiv | Year | Diff from IPCN |
|---|---|---|---|
| Titans (Google, Behrouz/Mirrokni) | 2501.00663 | 2025 | Memory updated in-forward, no chronometric, no LoRA consolidation |
| TTT layers (Stanford/Meta) | 2407.04620 | ICML 2025 | Hidden state IS an ML model, no episodic separation, no time substrate |
| Mamba | 2312.00752 | 2023 | "Time" = learned delta_t scalar, not wall-clock |
| Recurrent Memory Transformer | 2207.06881 | NeurIPS 2022 | Memory tokens in context, no time tags, dies per-run |
| Memorizing Transformers | 2203.08913 | ICLR 2022 | kNN over frozen cache, no learning from memory |
| Memory Networks | 1410.3916 | ICLR 2015 | No time, no duration, foundational |
| Neural Turing Machines | 1410.5401 | 2014 | Controller-step time only, volatile memory |
| Memformer | 2010.06891 | AACL 2022 | Untimed slots, sequence-only |

**Plus bonus time-aware peers:** ChronoFormer (2504.07373), ContiFormer (NeurIPS 2023). Both have time substrate. Neither has LoRA consolidation.

### 13.3 Prefix / pre-computational memory (11 methods)

Three families:
- **Prefix tuning** (Li & Liang 2101.00190, Prompt Tuning 2104.08691, P-Tuning v2 2110.07602) — static task vectors via KV cache
- **Memory tokens** (RMT, Memorizing Transformer, Compressive Transformer 1911.05507, Memformer) — runtime via attention
- **External memory** (NTM, Memory Networks, MemoryBank 2305.10250) — retrieval at attention time

**Critical finding:** NO published method injects memory at the hidden-state level BEFORE layer 0. IPCN sits in an empty design-matrix cell.

**Warning signal:** DKI (LucasMa2025/DKI, GitHub Feb 2026) — closest live analog. Author DEPRECATED their own approach citing capacity limits, OOD shift, factual accuracy loss. Failure modes we must address.

### 13.4 Industry labs

| Lab | Most relevant work | Year | Status |
|---|---|---|---|
| **Anthropic** | Lindsey "Emergent Introspective Awareness" transformer-circuits.pub | Oct 2025 | ~20% reliability on Claude Opus 4/4.1. Must cite. |
| **DeepMind/Google** | Titans + **MIRAS** (Dec 2025 blog) + **Nested Learning** (Nov 2025 blog), Behrouz + Mirrokni | 2025 | DIRECT memory-consolidation overlap. Must read MIRAS + Nested Learning. |
| **METR** | Kwa et al "Measuring AI Ability to Complete Long Tasks" 2503.14499 | Mar 2025 | Canonical external duration measurement. |
| **OpenAI** | Persistent ChatGPT memory (Feb 2024, expanded Apr 2025) | 2024-25 | Product only, no architecture. Market signal. |
| **Meta/FAIR** | Memory Layers at Scale (Berges Dec 2024), Memory Mosaics (Zhang/Bottou Jul 2025) | 2024-25 | Memory, no time, no introspection. |
| **Apollo** | Meinke "In-Context Scheming" 2412.04984 | Dec 2024 | Alignment-side introspection. Tangential. |

---

## Section 14: Novelty bets (what survives)

After 4 scans, 5 distinct claims survive:

| Bet | Survives? | Threat |
|---|---|---|
| 1. Pre-computational memory injection (before layer 0) | ✅ holds | DKI (GitHub, deprecated). Address its 3 failure modes. |
| 2. Real elapsed time as causal substrate | ✅ holds, tightened | Timely Machine (2601.16486) same insight but decode-only. Distinguish via persistent-memory + cross-session scope. |
| 3. LoRA consolidation from episodic to weights | ✅ holds strongly | DeepMind's Titans/MIRAS/Nested Learning is closest. Read both before writing draft. |
| 4. Falsifiable pre-registered 7-prediction multiverse | ✅ holds | Nobody pre-registers in this space. Strong differentiator. |
| 5. Continual learning with mechanical rollback | ✅ holds | No competing approach has rollback. |

**Sharpened thesis:** A neural-network architecture combining real elapsed time as causal substrate for persistent cross-session memory, pre-computational hidden-state injection before the main forward pass, and usage-driven LoRA consolidation of stable memories into weights. Falsifiable via pre-registered multiverse predictions.

---

## Section 15: Defense against KV-injection attack

Reviewers will say: "writing to hidden state h is just writing to KV cache with extra steps, since k = W_k · h and v = W_v · h."

Three counter-arguments:

1. **Information density per token.** Writing 32 prefix vectors directly to h vs 32 memory tokens in context. Memory tokens compete with real tokens for attention probability mass. Hidden-state writes don't.

2. **Multi-layer compositional propagation.** Writing to h at layer 0 means residual stream carries memory through every layer's MLP + attention. KV write only affects attention's view.

3. **Avoids attention-probability competition.** Memory doesn't have to win softmax weight against actual input tokens.

These become a subsection: "Mechanical Defense: Why Hidden-State Injection ≠ KV Injection."

---

## Section 16: Falsifiable predictions (pre-register all 7 on OSF)

Each prediction has a numerical threshold. Failure has a pre-written paper narrative. Publishable regardless.

| # | Prediction | Pass threshold |
|---|---|---|
| 1 | Memory swap changes layer 0 | D₀ > 0.1 (memory matters for early hidden state) |
| 2 | Layer 0/1 probes decode memory-conditioned sense | ≥ 80% accuracy |
| 3 | Ablation order holds: B5 > B4 > B3 > B2 > B1 > B0 on ambiguity | gap(B5, B1) ≥ 0.03 |
| 4 | Consolidation transfers slot to weights | CTI > 0.7 |
| 5 | Evolving memory beats static on silent gaps | gap ≥ 0.15 at 64k context + 512 silent min |
| 6 | Δτ-aware beats Δτ-ablated on duration tasks | gap ≥ 0.10 (and KL ≤ 0.1 on duration-insensitive) |
| 7 | Explicit evidence overrides memory | KL(amb) ≥ 0.5, KL(explicit) ≤ 0.1 |

**Failure narratives (multiverse pre-registration):**

- 1, 2, 3 fail → "pre-computational claim doesn't beat late retrieval"
- 4 fails → "external memory works, doesn't migrate to weights"
- 5 fails → "evolution unnecessary, prune it"
- 6 fails → "no real time substrate, only token-counting"
- 7 fails → "prefix delusional, needs stronger precision gating"

---

## Section 17: What we build

### 17.1 Hardware

- **Mac mini M4** (16 GB unified, MPS) — data prep, eval, small probes
- **NVIDIA DGX Spark** (128 GB unified LPDDR5X, Blackwell) — all 7-9B fine-tunes locally
- **Zero rented cloud needed for v1.**

### 17.2 Build path decision (PENDING)

- Option A: from-scratch 50-150M IPCN core (clean claims, ~6 weeks more dev)
- Option B: graft IPCN onto Pythia-160M or Qwen-0.5B (faster start, messier consolidation story)

### 17.3 Dimensions (v1)

| Item | Value |
|---|---|
| Core layers | 8 |
| Width | 512 |
| Heads | 8 |
| FFN | 2048 |
| Chunk length | 256 tokens |
| Prefix length | 32 vectors |
| Memory slots | 256 |
| Memory dim | 256 |
| Temporal state | 128 |
| LoRA rank | 8 (start), 16 (scale-up) |
| Precision | bf16 |
| Optimizer | AdamW |
| LR base | 3e-4 |
| LR LoRA | 1e-5 to 5e-5 |

### 17.4 Data (synthetic, ~200-300k chunks)

**Memory-Biased Ambiguity Suite (~100k examples)**
Three phases per sample: rules → ambiguous input → question that needs rules.
Example:
```
Memory: mips are forest animals with wings, sleep upside down, hunt at night
Current input: child saw mip hanging above cave
Question: what is mip using? A. wings B. wheels C. keyboard D. spoon
```

**Temporal Latent World (~50k streams)**
Hidden simulator emits text events with timestamps. Decay, delayed transitions, periodic events, silent gaps.
Example:
```
min 421: device_07 moved to vault_11
min 422: charged by 9 units
min 423: rule: if active, decay 1/3min
512 minutes pass, no events
Q: current energy of device_07?
```

**Consolidation Ladder (~20k rules)**
Introduce rule → query across many chunks → consolidate → remove slot → test if model still applies rule.

**Mixed real text (~100k tokens, OpenWebText/C4)**
Keep general language ability during training.

### 17.5 Training phases

| Phase | Duration | Goal |
|---|---|---|
| Phase 0 | 50-100k steps | Adapters frozen, sanity check: prefix affects layer 0? |
| Phase 1 | 50k steps | PFC adapters on, test consolidation in PFC only |
| Phase 2 | 50k steps | Early-core adapters on (layers 1-2) |
| Phase 3 | 100k steps | Mix synthetic + real text, watch perplexity drift |

Total: ~3-4 months on DGX Spark.

### 17.6 Ablation matrix

| Model | Slots | Prefix | Broadcast | Evolution | Consolidation |
|---|---|---|---|---|---|
| A0 | no | no | no | no | no |
| A1 | yes | late read only | no | no | no |
| A2 | yes | yes | no | no | no |
| A3 | yes | yes | yes | no | no |
| A4 | yes | yes | yes | yes | no |
| A5 | yes | yes | yes | yes | PFC only |
| A6 | yes | yes | yes | yes | PFC + layers 1-2 |

Core hypothesis: A3 beats A1 on ambiguity, A4 beats A3 on silent gaps, A5/A6 show positive CTI without contradiction collapse.

### 17.7 Hard pass criteria (Phase 0 first experiment)

| Metric | Threshold |
|---|---|
| Acc(A3) − Acc(A1) on ambiguity | ≥ 0.10 |
| I_pre(A3) when prefix useful | ≥ 0.25 |
| Acc_explicit(A3) − Acc_explicit(A1) | ≥ −0.02 |
| CTI(A5) after consolidation | ≥ 0.70 |
| OLP(A5) / OLP(A3) drift | ≤ 1.05 |

Fail any → architecture doesn't support claimed mechanism. Investigate.

---

## Section 18: Failure modes + fixes

| Mode | Symptoms | Fixes |
|---|---|---|
| Prefix delusion | Model overuses memory vs current input. High memory-swap KL on explicit inputs. | Increase precision loss weight, cap λ_pre, adversarial contradictory-prefix training |
| Slot collapse | Few slots eat all writes. Entropy < 0.5 log N_m. | Diversity loss, usage penalty in assignment, top-2 soft assignment, dead-slot reinit |
| Consolidation overfitting | High CTI on replay, poor generalization | Replay augmentation, paraphrase contexts, stricter held-out validation |
| Runaway self-modification | LM perplexity rises, KL drift, contradiction accuracy degrades | Smaller η_cons, lower LoRA rank, EWC regularization, adapter rollback |

---

## Section 19: Next steps (chronological)

**This week:**
1. Read MIRAS blog (DeepMind Dec 2025)
2. Read Nested Learning blog (DeepMind Nov 2025)
3. Read DKI GitHub deprecation notes
4. Read Wang et al "Discrete Minds in Continuous World" + Garikaparthi "Can LLMs Perceive Time?"

**Next 1-2 weeks:**
5. Decide build path: from-scratch vs graft
6. Draft OSF pre-registration with 7 predictions + thresholds + multiverse narratives
7. Lock dataset generation code

**Then:**
8. Code minimal IPCN prototype (PyTorch skeleton)
9. Generate synthetic datasets
10. Phase 0 sanity training
11. Phase 1-3 training
12. Run ablation matrix
13. Score against predictions
14. Write paper draft

**Venue decision:** deferred until results land. Lock when we know what we've got.

---

## Section 20: Risk map

| Risk | Likelihood | Mitigation |
|---|---|---|
| DeepMind ships time + memory + consolidation paper before we submit | Medium-high | Sprint. Pre-register on OSF NOW. |
| Timely Machine extends to persistent memory in follow-up | Medium | Cite as concurrent, distinguish scope (persistent + cross-session). |
| Reviewers cite DKI deprecation as evidence approach won't work | High | Address all 3 DKI failure modes in experimental design. |
| Berg et al "subjective experience" gets conflated with our claims | Medium | Hard scope wall in intro: "we do not claim phenomenal experience." |
| Garikaparthi's diagnostic eats our motivation | Medium | Cite as diagnostic predecessor. Frame IPCN as substrate fix. |

---

## Appendix A: Full technical spec

For the full LaTeX spec with all math (equations, defaults, loss terms, slot update rules, evolution dynamics, consolidation objective), see the original IPCN spec document. This MD file is the plain-language companion.

---

## Appendix B: Key citations (must-cite list)

**Direct LLM time-experience:**
- Garikaparthi, "Can LLMs Perceive Time?" arXiv 2604.00010, ICLR 2026 Workshop ICBINB
- Wang et al, "Discrete Minds in a Continuous World" arXiv 2506.05790, EMNLP 2025 Findings
- Cheng et al, "Your LLM Agents are Temporally Blind" arXiv 2510.23853
- Ma et al, "Timely Machine" arXiv 2601.16486
- Berg et al, "LLMs Report Subjective Experience" arXiv 2510.24797

**Memory architectures:**
- Behrouz, Zhong, Mirrokni, "Titans" arXiv 2501.00663
- Sun et al, "TTT layers" arXiv 2407.04620
- Gu, Dao, "Mamba" arXiv 2312.00752
- Bulatov et al, "Recurrent Memory Transformer" arXiv 2207.06881
- Wu et al, "Memorizing Transformers" arXiv 2203.08913
- Weston et al, "Memory Networks" arXiv 1410.3916
- Graves et al, "Neural Turing Machines" arXiv 1410.5401

**Introspection / self-knowledge:**
- Binder et al, "Looking Inward" arXiv 2410.13787, ICLR 2025
- Anthropic, "Emergent Introspective Awareness in LLMs" transformer-circuits.pub Oct 2025

**External duration measurement:**
- Kwa et al, "Measuring AI Ability to Complete Long Tasks" arXiv 2503.14499 (METR)

**Prefix methods:**
- Li & Liang, "Prefix-Tuning" arXiv 2101.00190
- Lester et al, "Prompt Tuning" arXiv 2104.08691

**Background:**
- Vaswani et al, "Attention Is All You Need" arXiv 1706.03762 (positional encoding)
- Bennett, "A Mind Cannot Be Smeared Across Time" arXiv 2601.11620

---

*End of paper. Word count: ~3,800. Living document, update as scans deepen and prototype progresses.*
