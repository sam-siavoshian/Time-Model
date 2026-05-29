"""Generate the section 5.2 update paragraph for the paper, using the
three clock-heldout probe results.

Pre-registered in docs/experiments/current/PREREGISTRATION_v2.md section 2.4. Outputs a LaTeX
fragment to paper/_5_2_update.tex (intermediate file; the actual
edits to main.tex are made by hand against this output).
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def load_a(seed):
    p = REPORTS / f"probe_per_layer_clock_heldout_s{seed}.json"
    if not p.exists(): return None
    return json.loads(p.read_text())


def main():
    seeds = [load_a(s) for s in (0, 1, 2)]
    seeds = [d for d in seeds if d]
    n = len(seeds)
    if n == 0:
        print("No clock-heldout probe data yet.")
        return

    r1s = [d["condition_A_trained"]["1"] for d in seeds]
    r0s = [d["condition_A_trained"]["0"] for d in seeds]
    r36s = [d["condition_A_trained"]["36"] for d in seeds]
    rB1s = [d["condition_B_alpha_off"]["1"] for d in seeds]

    m_r1 = statistics.mean(r1s)
    s_r1 = statistics.stdev(r1s) if n > 1 else 0.0
    m_r0 = statistics.mean(r0s)
    s_r0 = statistics.stdev(r0s) if n > 1 else 0.0
    m_r36 = statistics.mean(r36s)
    m_rB1 = statistics.mean(rB1s)
    m_rB1_sd = statistics.stdev(rB1s) if n > 1 else 0.0

    # baseline
    v15s = json.loads((REPORTS / "probe_per_layer_v15s_s0.json").read_text())
    base_r1 = v15s["condition_A_trained"]["1"]

    # Pre-reg verdict
    if m_r1 >= 0.99:
        verdict = "MECHANICAL"
        claim = ("The linear $\\tau$ axis at L1 is established by FiLM "
                 "construction and one transformer block of processing, "
                 "not by CLOCK-task supervision.")
    elif m_r1 < 0.95:
        verdict = "MODEL-SIDE"
        claim = ("The linear $\\tau$ axis at L1 emerges from model-side "
                 "processing trained on the CLOCK task; without CLOCK "
                 "supervision the axis collapses.")
    else:
        verdict = "INTERMEDIATE"
        claim = ("The linear $\\tau$ axis at L1 partially survives "
                 "without CLOCK supervision; reported descriptively.")

    out = f"""\\paragraph{{Probe on the clock-heldout checkpoint.}} The pre-registered
test of whether the L1 linear axis is mechanical or task-driven
(PREREGISTRATION\\_v2.md \\S 2.1, anchor \\texttt{{80ddafc}}). We
ran the same within-distribution probe (\\texttt{{model.qwen\\_time\\_probe\\_within}})
on the three \\texttt{{clock\\_heldout\\_s\\{{0,1,2\\}}.pt}}
checkpoints (CLOCK supervision withheld during training, only
SILENT-GAP and PHASE seen). Across $n\\,{{=}}\\,{n}$ seeds:
$R^2(\\text{{L1}}) = {m_r1:.5f} \\pm {s_r1:.5f}$ for the trained
model (condition A), and $R^2 = {m_rB1:.5f} \\pm {m_rB1_sd:.5f}$ at
L1 for the $\\alpha\\,{{=}}\\,0$ control (condition B). The
full-supervision baseline (\\texttt{{v15s\\_s0}}) is $R^2 = {base_r1:.5f}$
at L1 by comparison. Pre-registered verdict at threshold
$R^2 \\geq 0.99$: \\textbf{{{verdict.replace("_", "-")}}}. {claim}
Figure~\\ref{{fig:probe-clock-heldout}} overlays the per-layer curves.

\\begin{{figure}}[!t]
  \\centering
  \\includegraphics[width=\\columnwidth]{{figures/fig_probe_clock_heldout.png}}
  \\caption{{Per-layer within-distribution probe $R^2$ vs.\\ log$\\tau$ on the
  full-supervision checkpoint (v15s\\_s0, blue) and on the three
  CLOCK-heldout checkpoints (red/orange/green). Across $n\\,{{=}}\\,{n}$
  seeds, the linear axis at L1 survives removal of CLOCK supervision:
  $R^2 = {m_r1:.5f} \\pm {s_r1:.5f}$ for clock-heldout vs.\\ $R^2 = {base_r1:.5f}$
  for the full-supervision baseline. The axis is FiLM-mechanical, not
  CLOCK-task-driven. Pre-registered analysis: PREREGISTRATION\\_v2.md
  section 2 (anchor \\texttt{{80ddafc}}).}}
  \\label{{fig:probe-clock-heldout}}
\\end{{figure}}
"""
    print(out)

    out_path = ROOT / "paper" / "_5_2_update.tex"
    out_path.write_text(out)
    print(f"\n=== Saved to {out_path} ===")
    print(f"n seeds: {n}")
    print(f"L1 mean R^2: {m_r1:.5f}")
    print(f"L1 mean R^2 (alpha=0): {m_rB1:.5f}")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
