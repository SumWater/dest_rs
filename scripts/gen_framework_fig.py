"""
Generate academic line-drawing Method framework figure for IEEE paper (refined).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "paper_figures"

fig, ax = plt.subplots(figsize=(7.2, 5.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

S = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', lw=1.3)
SG = dict(boxstyle='round,pad=0.3', facecolor='#f2f2f2', edgecolor='black', lw=1.3)
SD = dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='gray', lw=0.7, ls='--')
SDG = dict(boxstyle='round,pad=0.2', facecolor='#fafafa', edgecolor='gray', lw=0.6, ls='--')

def box(ax, x, y, w, h, text, s=S, fs=8.5, bold=False, color='black'):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h, **s)
    ax.add_patch(b)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=color)

def arrow(x1, y1, x2, y2, c='black', lw=1.0, ls='-', rad=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, ls=ls,
                                connectionstyle=f'arc3,rad={rad}'))

def bidir(x1, y1, x2, y2, c='gray', lw=0.7):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='<->', color=c, lw=lw, ls='--'))

# ── Title ──
ax.text(5, 9.7, 'Diagnostic Framework: Prefix-LoRA Hybrid PEFT',
        ha='center', fontsize=11, fontweight='bold')

# ── Top: Input ──
box(ax, 5, 9.0, 4.0, 0.8, 'Dialogue Context $x$ + Strategy ID $s$', SG, 9, True)

# ── Left: Prefix Bank ──
box(ax, 1.8, 7.0, 2.9, 1.05, 'Strategy-Specific\nPrefix Bank\n(9 $\\times$ 20 virtual tokens)', S, 8)

# ── Right: LoRA ──
box(ax, 8.2, 7.0, 2.9, 1.05, 'Shared LoRA\n($r{=}16$, Q/K/V/O proj.)\n$\\Delta W = \\frac{\\alpha}{r}BA$', S, 7.5)

# ── Centre: Frozen base ──
box(ax, 5, 4.5, 4.2, 1.0, 'Frozen Qwen3-8B (4-bit, bfloat16)', SG, 9, True)

# ── Bottom: Output ──
box(ax, 5, 2.2, 3.0, 0.8, 'Generated Utterance $y$', SG, 9, True)

# ── Loss boxes ──
box(ax, 8.5, 2.2, 2.2, 0.65, '$\\mathcal{L}_{\\rm gen}$ (AR loss)', SDG, 7.5, color='gray')

# ── Auxiliary: Orth ──
bidir(3.25, 6.65, 6.75, 6.65)
ax.text(5.0, 6.95, '$\\mathcal{L}_{\\rm orth}$', ha='center', fontsize=7.5, color='gray', style='italic')

# ── Auxiliary: Cls ──
box(ax, 0.7, 5.3, 1.5, 0.65, '$\\mathcal{L}_{\\rm cls}$', SDG, 8, color='gray')
arrow(0.7, 5.62, 0.7, 6.15, c='gray', lw=0.7, ls='--')

# ── Auxiliary: Contrastive ──
box(ax, 6.8, 1.4, 2.0, 0.5, '$\\mathcal{L}_{\\rm contrastive}$', SDG, 7.5, color='gray')
arrow(5.8, 1.65, 5.8, 1.8, c='gray', lw=0.7, ls='--')
# dashed line connecting to main loss path
ax.plot([6.8, 7.4], [1.65, 2.0], color='gray', lw=0.7, ls='--')

# ── Gradient routing annotation ──
ax.text(4.2, 5.85, 'gradient\nrouting', fontsize=6.5, color='gray', ha='center',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='gray', lw=0.5, ls='--'))

# ── Main arrows ──
arrow(3.2, 8.6, 1.8, 7.5)   # input → prefix
arrow(6.8, 8.6, 8.2, 7.5)   # input → lora
arrow(1.8, 6.45, 3.3, 5.0)  # prefix → base
arrow(8.2, 6.45, 6.7, 5.0)  # lora → base
arrow(5, 4.0, 5, 2.6)       # base → output
arrow(6.5, 2.2, 7.4, 2.2)   # output → gen_loss

# ── Legend (bottom-left) ──
legend = [
    ('$\\mathcal{L}_{\\rm orth}$:', 'Orthogonal constraint (B5, B6 variants)'),
    ('$\\mathcal{L}_{\\rm cls}$:', 'Auxiliary strategy classification (B6)'),
    ('Gradient routing:', 'gen $\\to$ LoRA, cls $\\to$ Prefix (B6$_{\\rm grad}$)'),
    ('$\\mathcal{L}_{\\rm contrastive}$:', 'Token-level strategy sensitivity (B6$_{\\rm contrast}$)'),
]
for i, (label, desc) in enumerate(legend):
    y = 0.55 - i * 0.32
    ax.text(0.3, y, label, fontsize=6.5, color='dimgray', fontweight='bold')
    ax.text(2.8, y, desc, fontsize=6.5, color='dimgray')

fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig3_framework.pdf", dpi=300, bbox_inches='tight')
fig.savefig(OUT / "fig3_framework.png", dpi=200, bbox_inches='tight')
plt.close()
print("[OK] fig3_framework saved")
