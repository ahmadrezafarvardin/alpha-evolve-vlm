"""Figures for the report, built from the scores files and the evolution checkpoints (the sources of truth).

    python scripts/make_figures.py        # writes report/figures/{manual_steps,evolution,holdout}.pdf

Needs matplotlib (report-only dependency). Offline replays (steps 1 and 2a on reserve, 2a on dev) are not scores
files; their counts come from RESULTS.md and are marked as replays in the figure.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
# Reference categorical palette, slots 1-3 in fixed order (blue, orange, aqua); text in ink tokens, not series colors.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.5, "pdf.fonttype": 42,
})


def plain_log_axis(axis, ticks):
    """Log axis with plain decimal tick labels (0.3, 1, 3, ... instead of 10^x)."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    axis.set_major_formatter(fmt)


def scores(split, cid):
    return json.loads((ROOT / "scores" / split / f"scores_{cid}.json").read_text())


def acc(split, cid):
    return 100 * scores(split, cid)["accuracy"]


def manual_steps():
    steps = ["baseline", "1: extraction", "2a: more rules", "2b: answer-only", "2c: NA check"]
    dev = [acc("dev", "baseline_starting_scripts_3090"), acc("dev", "manual_instruct"), 100 * 73 / 128,
           acc("dev", "manual_instruct_s2b"), acc("dev", "manual_instruct_s2c")]
    res = [acc("reserve", "baseline_starting_scripts"), 100 * 207 / 400, 100 * 207 / 400,
           acc("reserve", "manual_instruct_s2b"), acc("reserve", "manual_instruct_s2c")]
    t_res = [scores("reserve", "baseline_starting_scripts")["avg_time_per_query"]] * 3 + [
        scores("reserve", "manual_instruct_s2b")["avg_time_per_query"],
        scores("reserve", "manual_instruct_s2c")["avg_time_per_query"]]
    x = range(len(steps))
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.1), gridspec_kw={"width_ratios": [1.6, 1]})
    a.plot(x, dev, color=BLUE, marker="o", ms=4, label="dev (design set)")
    a.plot(x, res, color=ORANGE, marker="s", ms=4, label="reserve[:400] (partly tuned)")
    a.plot([1, 2], [res[1], res[2]], marker="s", ms=4, mfc="white", mec=ORANGE, ls="none",
           label="reserve, offline replay")
    a.plot([0, 4], [acc("holdout", "baseline_starting_scripts"), acc("holdout", "manual_instruct")], ls="none",
           marker="D", ms=4.5, color=AQUA, label="holdout (final only)")
    a.set_xticks(list(x), steps, rotation=20, ha="right")
    a.set_ylabel("accuracy (%)")
    a.set_title("Manual Instruct: accuracy per step")
    a.legend(frameon=False, loc="lower right")
    b.plot(x, t_res, color=ORANGE, marker="s", ms=4)
    b.set_yscale("log")
    plain_log_axis(b.yaxis, [0.3, 0.5, 1, 2])
    b.set_ylim(0.3, 3)
    b.set_xticks(list(x), steps, rotation=20, ha="right")
    b.set_ylabel("s / query (reserve, log)")
    b.set_title("Time per query")
    fig.tight_layout()
    fig.savefig(OUT / "manual_steps.pdf")
    fig.savefig(OUT / "manual_steps.png", dpi=160)  # preview


def evolution():
    runs = [("instruct_20261015-1207", "Instruct evolution (evo, 400 queries)", BLUE),
            ("thinking_20261015-1453", "Thinking (evo[:128])", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), gridspec_kw={"width_ratios": [2.2, 1]})
    for ax, (run, title, color) in zip(axes, runs):
        ckpt = json.loads((ROOT / "population" / run / "checkpoint.json").read_text())
        programs = ckpt["programs"]
        full = sorted((p.get("generation", 0), 100 * p["scores"]["full"]["accuracy"])
                      for p in programs.values() if "full" in p.get("scores", {}))
        gens = range(ckpt["generation"] + 1)  # all generations, including ones with no full-stage child
        best, top = [], 0
        for g in gens:
            top = max([top] + [a for gg, a in full if gg == g])
            best.append(top)
        ax.scatter([g for g, _ in full], [a for _, a in full], s=10, color=color, alpha=0.45, lw=0,
                   label="child (full stage)")
        ax.step(list(gens), best, where="post", color=color, label="best so far")
        ax.set_xlabel("generation")
        ax.set_title(title)
        ax.set_xticks(list(gens) if len(gens) < 6 else list(range(0, len(gens), 2)))
    axes[0].set_ylabel("evo accuracy (%)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "evolution.pdf")
    fig.savefig(OUT / "evolution.png", dpi=160)  # preview


def holdout():
    def merged_thinking():
        rs = scores("holdout", "manual_thinking_h000")["responses"] + scores("holdout", "manual_thinking_h400")["responses"]
        return 100 * sum(r["correct"] for r in rs) / len(rs), sum(r["latency"] for r in rs) / len(rs)

    def composed(inst, thk):
        quant = {8, 9, 10, 12, 14, 15, 17, 19}
        rs = [t if i["qid"] in quant else i for i, t in zip(inst, thk)]
        return 100 * sum(r["correct"] for r in rs) / len(rs), sum(r["latency"] for r in rs) / len(rs)

    def at(cid):
        d = scores("holdout", cid)
        return 100 * d["accuracy"], d["avg_time_per_query"]

    mi = scores("holdout", "manual_instruct")["responses"]
    ei = scores("holdout", "evolved_instruct")["responses"]
    mt = scores("holdout", "manual_thinking_h000")["responses"] + scores("holdout", "manual_thinking_h400")["responses"]
    et = scores("holdout", "evolved_thinking")["responses"]
    key = lambda rs: sorted(rs, key=lambda r: r["key"])  # noqa: E731
    mi, ei, mt, et = key(mi), key(ei), key(mt), key(et)
    points = [  # (label, (acc, s/q), color, marker, label offset in points, horizontal alignment)
        ("baseline", at("baseline_starting_scripts"), INK2, "x", (7, 0), "left"),
        ("manual Instruct (best_speed)", at("manual_instruct"), BLUE, "o", (7, -1), "left"),
        ("evolved Instruct (best_overall)", at("evolved_instruct"), BLUE, "^", (7, 1), "left"),
        ("routing, manual", composed(mi, mt), AQUA, "o", (0, -11), "center"),
        ("routing, evolved", composed(ei, et), AQUA, "^", (-7, 0), "right"),
        ("manual Thinking", merged_thinking(), ORANGE, "o", (0, -11), "center"),
        ("evolved Thinking (best_accuracy)", at("evolved_thinking"), ORANGE, "^", (-7, 3), "right"),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    for label, (a, t), color, marker, off, ha in points:
        ax.scatter([t], [a], s=40 if marker != "x" else 30, color=color, marker=marker, zorder=3,
                   edgecolors="white" if marker != "x" else None, linewidths=0.8)
        ax.annotate(label, (t, a), xytext=off, textcoords="offset points", fontsize=7, color=INK, va="center",
                    ha=ha)
    ax.set_xscale("log")
    ax.set_xlabel("seconds per query (log scale, RTX 3090)")
    ax.set_ylabel("holdout accuracy (%)")
    ax.set_title("Holdout (800 queries): accuracy vs. time; circles manual, triangles evolved")
    ax.set_xlim(0.25, 60)
    ax.set_ylim(22, 71)
    plain_log_axis(ax.xaxis, [0.3, 1, 3, 10, 30])
    fig.tight_layout()
    fig.savefig(OUT / "holdout.pdf")
    fig.savefig(OUT / "holdout.png", dpi=160)  # preview


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    manual_steps()
    evolution()
    holdout()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.pdf"))))
