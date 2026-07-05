"""
Generate paper figures and LaTeX tables for the DeST-RS diagnostic study.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "paper_figures"
OUT.mkdir(exist_ok=True)

# ── colour palette ──
C_BASE    = '#b0b0b0'  # grey – text/instruction baselines
C_LORA    = '#4c72b0'  # blue – LoRA variants
C_PREFIX  = '#dd8452'  # orange – Prefix variants
C_JOINT   = '#55a868'  # green – joint / warm-start / staged
C_DIAG    = '#c44e52'  # red – diagnostic variants
C_BEST    = '#000000'  # black outline for B3

# ── data ──
# Each entry: (label, acc, ppl, category)
# Multi-seed: mean acc±std; PPL use mean across seeds

# --- Main Table data ---
main_data_newfix = [
    ("Frozen Base + Text",      27.41, None,  "baseline"),
    ("LoRA + Strategy Text",    42.22, 7.81,   "lora"),
    ("Prefix-only",             55.93, 10.40,  "prefix"),
    ("Prefix+LoRA",             37.17, 7.76,   "joint"),
    ("Warm-start",              41.85, 8.03,   "joint"),
    ("Prefix\u2192LoRA",        38.89, 8.01,   "joint"),
]

# Multi-seed means (for error bars on trade-off plot)
multi_seed = {
    "Prefix-only\n(B3)":        (55.87, 1.28, 10.54, "prefix"),
    "Prefix+LoRA\n(B4)":        (37.20, 3.78, 7.93,  "joint"),
    "Warm-start\n(B7)":         (40.49, 2.19, 8.18,  "joint"),
    "Prefix\u2192LoRA\n(B9)":   (40.00, 2.98, 8.07,  "joint"),
}

# Diagnostic variants (from fix_b6)
diag_data = [
    ("B5 Orth",            36.06, 7.80, "diag"),
    ("B6\u2081ix\n(orth+cls)", 31.85, 7.97, "diag"),
    ("B6 cls=5.0",         39.63, 7.92, "diag"),
    ("B6 orth=1.0",        40.37, 8.14, "diag"),
    ("B6 Gradient\nRouting",  37.78, 7.91, "diag"),
    ("B6 Contrastive",     42.59, 8.06, "diag"),
    ("B6 Contrastive\n(no orth)", 41.11, 7.90, "diag"),
]

# Additional: frozen prefix from fix_b6
frozen_prefix = ("Frozen Prefix\n+LoRA (B9)", 44.44, 8.06, "joint")

CAT_COLOR = {
    "baseline": C_BASE,
    "lora":     C_LORA,
    "prefix":   C_PREFIX,
    "joint":    C_JOINT,
    "diag":     C_DIAG,
}

# ── Figure 1: Strategy Accuracy Bar Chart ──
def plot_tradeoff():
    # ── Build data: ordered from highest to lowest accuracy ──
    # (label, acc, acc_std_or_None, ppl_or_None, category)
    rows = [
        # Core PEFT
        ("Prefix-only (B3)",           55.87, 1.28, 10.54, "prefix"),
        # Baselines
        ("LoRA + Strategy Text",       42.22, None,  7.81, "lora"),
        ("Frozen Base + Strategy Text",27.41, None,  None, "baseline"),
        # Staged / joint
        ("Warm-start (B7)",            40.49, 2.19,  8.18, "joint"),
        ("Prefix→LoRA (B9)",           40.00, 2.98,  8.07, "joint"),
        ("Prefix+LoRA (B4)",           37.20, 3.78,  7.93, "joint"),
        # Diagnostic
        ("+ Orth (B5)",                36.06, None,  7.80, "diag"),
        ("B6 contrastive",             42.59, None,  8.06, "diag"),
        ("B6 contrastive (no orth)",   41.11, None,  7.90, "diag"),
        ("B6 orth=1.0",                40.37, None,  8.14, "diag"),
        ("B6 cls=5.0",                 39.63, None,  7.92, "diag"),
        ("B6 gradient routing",        37.78, None,  7.91, "diag"),
        ("B6 fix (orth+cls)",          31.85, None,  7.97, "diag"),
    ]

    # sort within each category
    cat_order = {"prefix": 0, "baseline": 1, "lora": 2, "joint": 3, "diag": 4}
    rows.sort(key=lambda r: (cat_order.get(r[4], 9), -r[1]))

    labels = [r[0] for r in rows]
    accs   = [r[1] for r in rows]
    stds   = [r[2] for r in rows]
    ppls   = [r[3] for r in rows]
    cats   = [r[4] for r in rows]
    n = len(rows)

    colors = [CAT_COLOR[c] for c in cats]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    y_pos = range(n)
    bars = ax.barh(y_pos, accs, height=0.6, color=colors, zorder=3,
                   edgecolor='white', linewidth=0.5)

    # error bars for multi-seed
    for i, (acc, std) in enumerate(zip(accs, stds)):
        if std is not None:
            ax.errorbar(acc, i, xerr=std, fmt='none', ecolor='#333333',
                        elinewidth=1.5, capsize=3, capthick=1.2, zorder=4)

    # PPL annotation on the right side of bars
    for i, (acc, ppl) in enumerate(zip(accs, ppls)):
        if ppl is not None:
            ax.text(acc + 1.5, i, f"PPL {ppl:.2f}", va='center', fontsize=6.5,
                    color='#555555')
        else:
            ax.text(acc + 1.5, i, "PPL —", va='center', fontsize=6.5,
                    color='#aaaaaa')

    # random baseline line
    ax.axvline(x=11.11, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel('Strategy Accuracy (%)', fontsize=9.5)
    ax.set_xlim(0, 69)
    ax.tick_params(labelsize=8)
    ax.grid(axis='x', alpha=0.18, zorder=0)

    # group separator lines
    group_boundaries = []
    for i in range(1, n):
        if cats[i] != cats[i-1]:
            group_boundaries.append(i - 0.5)
    for gb in group_boundaries:
        ax.axhline(y=gb, color='#cccccc', linewidth=0.6, zorder=2)

    # category labels on right
    cat_labels_y = []
    current = None
    for i, c in enumerate(cats):
        if c != current:
            cat_labels_y.append(i)
            current = c
    cat_names = {"prefix": "Prefix", "baseline": "Text", "lora": "LoRA",
                 "joint": "Joint / Staged", "diag": "Diagnostic"}
    for cy in cat_labels_y:
        ax.text(66, cy, cat_names.get(cats[cy], ''), va='center', fontsize=7,
                color=CAT_COLOR[cats[cy]], fontweight='bold', alpha=0.7)

    # random text
    ax.text(12.5, n - 0.15, 'Random (11.1%)', fontsize=7, color='#888888', va='bottom')

    fig.tight_layout(pad=0.6)
    for fmt in ["png", "pdf"]:
        fig.savefig(OUT / f"fig1_tradeoff.{fmt}", dpi=250, bbox_inches="tight")
    plt.close()
    print("[OK] fig1_tradeoff saved")

# ── Figure 2: Per-class Accuracy Breakdown ──
def plot_perclass():
    strategies = ['elicit-pref','self-need','other-need','no-need',
                  'promote-coordination','showing-empathy','small-talk',
                  'uv-part','vouch-fair']
    # Per-class acc from B3 (main new_fix seed42) and B4
    # We'll load from the strategy_eval_llm.json details
    base = Path(r"C:\Users\tanyang\Desktop\大模型\小论文\dest_rs\output\need")
    
    def get_per_class(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        details = data.get('details', [])
        by_target = {}
        for d in details:
            t = d['target_strategy']
            if t not in by_target:
                by_target[t] = {'correct': 0, 'total': 0}
            by_target[t]['total'] += 1
            if d.get('correct', 0):
                by_target[t]['correct'] += 1
        return [by_target.get(s, {'correct':0,'total':1})['correct']/
                max(by_target.get(s, {'correct':0,'total':1})['total'],1)*100
                for s in strategies]

    b3_path = base / "casino_augmented_new_fix_seed42/b3_prefix_only/strategy_eval_llm.json"
    b4_path = base / "casino_augmented_new_fix_seed42/b4_prefix_lora/strategy_eval_llm.json"
    
    b3_acc = get_per_class(b3_path)
    b4_acc = get_per_class(b4_path)
    
    # Evaluator calibration on human text
    calib = [84.2, 51.4, 30.0, 0.0, 51.7, 40.0, 57.7, 25.0, 35.7]

    x = np.arange(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, b3_acc, width, label='Prefix-only (B3)', color=C_PREFIX, edgecolor='white')
    bars2 = ax.bar(x, b4_acc, width, label='Prefix+LoRA (B4)', color=C_JOINT, edgecolor='white')
    bars3 = ax.bar(x + width, calib, width, label='Evaluator Calibration\n(on human text)', 
                   color=C_BASE, edgecolor='white', hatch='//')

    ax.set_ylabel('Accuracy %', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title('Per-Class Strategy Accuracy: B3 vs B4 vs Evaluator Calibration', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_perclass.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUT / "fig2_perclass.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] fig2_perclass saved")


# ── Bootstrap CI ──
def bootstrap_ci(data, n_bootstrap=10000):
    """95% bootstrap CI for mean of binary outcomes."""
    data = np.array(data)
    means = []
    n = len(data)
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        means.append(sample.mean())
    return np.percentile(means, [2.5, 97.5])


def compute_bootstrap():
    base = Path(r"C:\Users\tanyang\Desktop\大模型\小论文\dest_rs\output\need")
    seeds = ['casino_augmented_new_fix_seed42', 'casino_augmented_new_fix_seed43', 'casino_augmented_new_fix_seed44']
    exps = ['b3_prefix_only', 'b4_prefix_lora', 'b7_dest_rs_warm', 'b9_prefix_then_lora']

    print("\n=== Bootstrap 95% CI (per-seed) ===")
    for exp in exps:
        all_accs = []
        for seed in seeds:
            path = base / seed / exp / "strategy_eval_llm.json"
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            details = data['details']
            accs_seed = [d['correct'] for d in details]
            all_accs.extend(accs_seed)
        ci = bootstrap_ci(all_accs)
        mean = np.mean(all_accs) * 100
        print(f"  {exp:<20s}: {mean:.2f}%  [{ci[0]*100:.2f}, {ci[1]*100:.2f}]")

    print("\n=== Paired bootstrap: B3 vs B4 ===")
    for seed in seeds:
        b3_path = base / seed / "b3_prefix_only/strategy_eval_llm.json"
        b4_path = base / seed / "b4_prefix_lora/strategy_eval_llm.json"
        with open(b3_path, 'r', encoding='utf-8') as f:
            b3 = json.load(f)['details']
        with open(b4_path, 'r', encoding='utf-8') as f:
            b4 = json.load(f)['details']
        # Group by dialogue_id + target_strategy to form pairs
        b3_map = {}
        for d in b3:
            key = (d.get('dialogue_id'), d.get('target_strategy'))
            b3_map[key] = d['correct']
        b4_map = {}
        for d in b4:
            key = (d.get('dialogue_id'), d.get('target_strategy'))
            b4_map[key] = d['correct']
        # Common keys
        common = set(b3_map.keys()) & set(b4_map.keys())
        diffs = [b3_map[k] - b4_map[k] for k in common]
        mean_diff = np.mean(diffs) * 100
        ci = bootstrap_ci(diffs)
        print(f"  {seed}: B3-B4 diff = {mean_diff:.1f}pp [{ci[0]*100:.1f}, {ci[1]*100:.1f}]")


# ── LaTeX Tables ──
def latex_tables():
    latex = []
    
    # Table 1: Main Comparison
    latex.append(r"""
\begin{table}[t]
\centering
\caption{Main results: strategy accuracy and perplexity across different PEFT configurations.
         Multi-seed means and standard deviations over 3 seeds (42/43/44) are reported for
         key experiments.}
\label{tab:main}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Prefix} & \textbf{LoRA} & \textbf{Valid PPL} & \textbf{Strategy Acc (\%)} \\
\midrule
Frozen Base + Strategy Text   & --  & --  & --   & 27.41 \\
LoRA + Strategy Text           & --  & \cmark & 7.81 & 42.22 \\
Prefix-only (B3)               & \cmark & --  & 10.54$\pm$0.12 & 55.87$\pm$1.28 \\
Prefix+LoRA (B4)               & \cmark & \cmark & 7.93$\pm$0.17 & 37.20$\pm$3.78 \\
Warm-start DeST-RS (B7)        & \cmark & \cmark & 8.18$\pm$0.24 & 40.49$\pm$2.19 \\
Prefix$\rightarrow$LoRA (B9)   & \cmark$^\dagger$ & \cmark & 8.07$\pm$0.10 & 40.00$\pm$2.98 \\
\midrule
\multicolumn{5}{l}{\footnotesize $^\dagger$Frozen Prefix from B3; only LoRA is trained.} \\
\multicolumn{5}{l}{\footnotesize Multi-seed: mean$\pm$std over seeds 42/43/44 for PPL and accuracy.} \\
\bottomrule
\end{tabular}
\end{table}
""")

    # Table 2: Diagnostic Variants
    latex.append(r"""
\begin{table}[t]
\centering
\caption{Diagnostic variants: effect of orthogonal regularization, auxiliary classification,
         gradient routing, and contrastive loss.}
\label{tab:diag}
\begin{tabular}{lcc}
\toprule
\textbf{Variant} & \textbf{Valid PPL} & \textbf{Strategy Acc (\%)} \\
\midrule
\multicolumn{3}{c}{\textit{Baselines (reference)}} \\
\quad Prefix-only (B3)        & 10.40 & 55.93 \\
\quad Prefix+LoRA (B4)        & 7.76  & 37.17 \\
\midrule
\multicolumn{3}{c}{\textit{Orthogonal + Classification}} \\
\quad + Orth (B5)             & 7.80  & 36.06 \\
\quad + Orth + Cls (B6$_{\text{fix}}$) & 7.97 & 31.85 \\
\quad + Orth + Cls ($\lambda_{\text{cls}}{=}5.0$) & 7.92 & 39.63 \\
\quad + Orth + Cls ($\lambda_{\text{orth}}{=}1.0$) & 8.14 & 40.37 \\
\midrule
\multicolumn{3}{c}{\textit{Gradient Routing \& Contrastive}} \\
\quad Gradient Routing        & 7.91  & 37.78 \\
\quad Contrastive Loss        & 8.06  & 42.59 \\
\quad Contrastive (no orth)   & 7.90  & 41.11 \\
\midrule
\multicolumn{3}{c}{\textit{Staged Training}} \\
\quad Frozen Prefix + LoRA (B9) & 8.06 & 44.44 \\
\bottomrule
\end{tabular}
\end{table}
""")

    # Table 3: Evaluator Calibration
    latex.append(r"""
\begin{table}[t]
\centering
\caption{LLM evaluator calibration on human-written validation responses.
         The evaluator achieves 52.0\% overall accuracy, revealing substantial
         class ambiguity.}
\label{tab:calib}
\begin{tabular}{lcc}
\toprule
\textbf{Strategy} & \textbf{Calibration Acc (\%)} & \textbf{Notes} \\
\midrule
elicit-pref         & 84.2 & Relatively reliable \\
small-talk          & 57.7 & Often confused with empathy \\
promote-coordination& 51.7 & Confused with self-need \\
self-need           & 51.4 & Evaluator bias toward this class \\
showing-empathy     & 40.0 & Moderate reliability \\
vouch-fair          & 35.7 & Difficult to distinguish \\
other-need          & 30.0 & Unreliable \\
uv-part             & 25.0 & Unreliable \\
no-need             & 0.0  & Shortcut risk: evaluator cannot recognize \\
\midrule
Overall             & 52.0 & Substantial ambiguity \\
\bottomrule
\end{tabular}
\end{table}
""")

    latex_path = OUT / "tables.tex"
    latex_path.write_text("\n".join(latex), encoding='utf-8')
    print(f"[OK] LaTeX tables saved to {latex_path}")


def plot_confusion_matrix():
    """Generate evaluator calibration confusion matrix heatmap."""
    import matplotlib.colors as mcolors

    calib_path = Path(__file__).resolve().parent.parent / "output" / "evaluator_calibration_results.json"
    with open(calib_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    STRATEGIES = [
        "elicit-pref", "small-talk", "promote-coordination", "self-need",
        "showing-empathy", "vouch-fair", "other-need", "uv-part", "no-need",
    ]
    SHORT = ["Elicit", "SmallTalk", "PromCoord", "SelfNeed",
             "Empathy", "VouchFair", "OtherNeed", "UV-Part", "NoNeed"]
    n = len(STRATEGIES)
    s2i = {s: i for i, s in enumerate(STRATEGIES)}

    # build confusion matrix
    cm = np.zeros((n, n), dtype=int)
    for s in samples:
        g, p = s["gold"], s["pred"]
        if p is None:
            continue
        if g in s2i and p in s2i:
            cm[s2i[g], s2i[p]] += 1

    # per-class accuracy and total
    row_sums = cm.sum(axis=1)
    per_class_acc = np.zeros(n)
    for i in range(n):
        per_class_acc[i] = cm[i, i] / row_sums[i] * 100 if row_sums[i] > 0 else 0
    total_acc = np.diag(cm).sum() / cm.sum() * 100

    # normalize rows → recall (what fraction of true class i goes to predicted j)
    cm_norm = np.zeros_like(cm, dtype=float)
    for i in range(n):
        if row_sums[i] > 0:
            cm_norm[i] = cm[i] / row_sums[i]

    # ── plot ──
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    cmap = plt.cm.YlOrRd
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=0.9)

    # colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Fraction of true class", fontsize=8.5)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(SHORT, fontsize=7.5)
    ax.set_xlabel("Predicted", fontsize=9, labelpad=8)
    ax.set_ylabel("True (Gold)", fontsize=9, labelpad=8)

    # annotate: count + percentage
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val == 0:
                continue
            pct = cm_norm[i, j] * 100
            text = f"{val}\n({pct:.0f}%)"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5,
                    color=color, fontweight="bold" if i == j else "normal")

    # per-class accuracy in right margin
    for i in range(n):
        ax.text(n + 0.35, i, f"{per_class_acc[i]:.0f}%", va="center",
                fontsize=7.5, fontweight="bold",
                color=C_DIAG if per_class_acc[i] < 40 else (C_PREFIX if per_class_acc[i] < 70 else C_JOINT))

    ax.set_xlim(-0.5, n + 0.8)
    ax.text(n + 0.35, -1.0, "Acc.", fontsize=7, ha="center", color="grey")

    # overall accuracy in title area
    ax.set_title(f"Evaluator confusion matrix (human-written responses)\nOverall accuracy: {total_acc:.1f}%  |  Total: {int(cm.sum())} samples",
                 fontsize=10, fontweight="bold", pad=12)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(OUT / f"fig4_confusion.{fmt}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Confusion matrix saved to {OUT / 'fig4_confusion.png'}")


if __name__ == "__main__":
    plot_tradeoff()
    plot_perclass()
    compute_bootstrap()
    latex_tables()
    plot_confusion_matrix()
    print("\nDone! All figures and tables saved to", OUT)
