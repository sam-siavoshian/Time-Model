"""Pre-registered figure (docs/experiments/current/PREREGISTRATION_v2.md section 2.4):
fig_probe_clock_heldout.png overlays the per-layer probe R^2 curves
for v15s_s0 (full supervision) and clock_heldout_s{0,1,2}.

Input:
  reports/probe_per_layer_v15s_s0.json
  reports/probe_per_layer_clock_heldout_s{0,1,2}.json

Output:
  paper/figures/fig_probe_clock_heldout.png
  figures/fig_probe_clock_heldout.png
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

def load_a(path):
    p = REPORTS / path
    if not p.exists(): return None, None
    d = json.loads(p.read_text())
    A = d.get("condition_A_trained", {})
    layers = sorted(int(k) for k in A.keys())
    return layers, [A[str(k)] for k in layers]

def main():
    out_paper = ROOT / "paper" / "figures" / "fig_probe_clock_heldout.png"
    out_root  = ROOT / "figures" / "fig_probe_clock_heldout.png"
    out_paper.parent.mkdir(parents=True, exist_ok=True)
    out_root.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.0))

    # v15s_s0 (full supervision)
    L, R = load_a("probe_per_layer_v15s_s0.json")
    if L is None:
        print("MISSING v15s baseline")
        return
    ax.plot(L, R, "-o", label="Full-supervision (v15s_s0)", linewidth=2,
            markersize=4, color="#1f77b4")

    # clock_heldout seeds
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    seeds_loaded = 0
    r1_clock = []
    for seed, color in zip((0, 1, 2), colors):
        L, R = load_a(f"probe_per_layer_clock_heldout_s{seed}.json")
        if L is None:
            print(f"MISSING clock_heldout_s{seed}")
            continue
        seeds_loaded += 1
        r1_clock.append(R[1])
        ax.plot(L, R, "--^", label=f"Clock-heldout seed {seed}", linewidth=1.3,
                markersize=3, color=color, alpha=0.8)

    ax.axhline(0.99, color="grey", linestyle=":", linewidth=1.0,
               label="Pre-reg threshold (R^2 = 0.99)")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Within-distribution probe R^2 (trained)")
    ax.set_title("Per-layer linear probe R^2: full-supervision vs CLOCK-heldout")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_paper, dpi=200)
    fig.savefig(out_root, dpi=200)
    print(f"saved -> {out_paper}")
    print(f"saved -> {out_root}")
    print(f"loaded seeds: {seeds_loaded}/3")
    if r1_clock:
        import statistics
        m = statistics.mean(r1_clock)
        s = statistics.stdev(r1_clock) if len(r1_clock) > 1 else 0.0
        print(f"clock_heldout L1 R^2: mean={m:.4f}, sd={s:.4f}, n={len(r1_clock)}")

if __name__ == "__main__":
    main()
