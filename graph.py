import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open("contradiction_results.json", "r") as f:
    data = json.load(f)

intervals = ["8-16", "16-24", "24-32", "32-40", "40-52", "52-64", "64-76", "76-92", "92-108", "108-128"]
models = [
    ("vqvae", "VQ-VAE Baseline (Ground-Truth Codes)", "#2b5c8f", "--", "s"),
    ("checkpoint-iter-10000.pt", "NCM Generator @ 10k Steps", "#74a9cf", "-", "o"),
    ("checkpoint-iter-25000.pt", "NCM Generator @ 25k Steps", "#8c96c6", "-", "^"),
    ("checkpoint-iter-50000.pt", "NCM Generator @ 50k Steps", "#8856a7", "-", "D"),
    ("checkpoint-iter-75000.pt", "NCM Generator @ 75k Steps", "#d94801", "-", "o"),
]

# Export clean CSV table
rows = []
for interval in intervals:
    row = {"Distance_Interval": interval}
    for key, _, _, _, _ in models:
        short = key.replace("checkpoint-iter-", "").replace(".pt", "")
        row[f"{short}_contradiction_%"] = round(data[key][interval]["contradiction_prob"] * 100, 2)
        row[f"{short}_same_word_%"] = round(data[key][interval]["same_word_prob"] * 100, 2)
        row[f"{short}_entropy"] = round(data[key][interval]["entropy_spread"], 4)
    rows.append(row)
pd.DataFrame(rows).to_csv("ncm_intra_chunk_coherence_table.csv", index=False)

# Plot 2-panel diagnostic figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=300)
x = np.arange(len(intervals))

for key, label, color, ls, marker in models:
    y = [data[key][iv]["contradiction_prob"] * 100 for iv in intervals]
    lw = 2.6 if ("75000" in key or "vqvae" in key) else 1.6
    alpha = 1.0 if ("75000" in key or "vqvae" in key) else 0.78
    ax1.plot(x, y, label=label, color=color, linestyle=ls, marker=marker, linewidth=lw, markersize=6, alpha=alpha)

ax1.axhline(83.33, color="#555555", linestyle=":", linewidth=1.3, label="Uniform 6-Choice Baseline (83.3%)")
ax1.set_xticks(x)
ax1.set_xticklabels(intervals, rotation=30, ha="right", fontsize=9.5)
ax1.set_ylabel("Intra-Chunk Contradiction Probability (%)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Inter-Slot Token Distance Bucket (VQ-VAE Hierarchy Ladder)", fontsize=10.5, fontweight="bold")
ax1.set_title("A. Intra-Chunk Attribute Contradiction Rate vs. Token Distance", fontsize=11.5, fontweight="bold", pad=10)
ax1.set_ylim(-4, 93)
ax1.grid(True, linestyle="--", alpha=0.35)
ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5, loc="lower left", bbox_to_anchor=(0.02, 0.08))

ckpt_labels = ["VQ-VAE\n(Oracle)", "Gen\n10k", "Gen\n25k", "Gen\n50k", "Gen\n75k"]
ckpt_keys = [m[0] for m in models]
mean_contra = [np.mean([data[k][iv]["contradiction_prob"] * 100 for iv in intervals]) for k in ckpt_keys]
std_contra = [np.std([data[k][iv]["contradiction_prob"] * 100 for iv in intervals]) for k in ckpt_keys]
mean_ent = [np.mean([data[k][iv]["entropy_spread"] for iv in intervals]) for k in ckpt_keys]

x_c = np.arange(len(ckpt_labels))
width = 0.38
bars1 = ax2.bar(x_c - width/2, mean_contra, width, yerr=std_contra, capsize=4,
                error_kw=dict(lw=1.1, capthick=1.1, ecolor="#333333"),
                color=["#2b5c8f", "#74a9cf", "#8c96c6", "#8856a7", "#d94801"],
                edgecolor="#222222", linewidth=0.8, label="Mean Contradiction % (±1 SD)")
ax2.set_ylabel("Mean Contradiction Probability (%)", fontsize=11, fontweight="bold", color="#222222")
ax2.set_xticks(x_c)
ax2.set_xticklabels(ckpt_labels, fontsize=10, fontweight="bold")
ax2.set_ylim(0, 92)
ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

for bar, val, sd in zip(bars1, mean_contra, std_contra):
    top_y = val + sd + 1.8 if val > 1 else val + 1.8
    ax2.text(bar.get_x() + bar.get_width()/2, top_y, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax2_twin = ax2.twinx()
ax2_twin.plot(x_c + width/2, mean_ent, color="#1b7837", marker="s", linewidth=2.2, markersize=7, label="Mean Marginal Entropy (nats)")
ax2_twin.set_ylabel("Mean Candidate Marginal Entropy (nats)", fontsize=11, fontweight="bold", color="#1b7837")
ax2_twin.tick_params(axis="y", labelcolor="#1b7837")
ax2_twin.set_ylim(0, 1.38)

for xc_val, ent_val in zip(x_c + width/2, mean_ent):
    ax2_twin.text(xc_val + 0.05, ent_val + 0.04, f"{ent_val:.2f}", ha="center", va="bottom", fontsize=8.5, color="#1b7837", fontweight="bold")

ax2.set_title("B. Training Trajectory: Contradiction Rate vs. Logit Entropy", fontsize=11.5, fontweight="bold", pad=10)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

plt.suptitle("Intra-Chunk Attribute Permanence Diagnostic (TinyStories NCM, Context = 128, 12-Level VQ-VAE)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("ncm_intra_chunk_coherence_diagnostic.png", bbox_inches="tight", dpi=300)