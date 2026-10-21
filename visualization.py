"""
Visualization module for Fuzzy Inference System.
Creates clean, publication-grade Matplotlib plots in an artistic Light Theme:
1. Input 1 (CIBIL Score) Membership Functions with live operating point
2. Input 2 (Monthly Income) Membership Functions with live operating point
3. Aggregated Output (Loan Approval Chance) with Centroid defuzzification
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for web servers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any


# Clean, Architectural Light Theme Palette
STYLE_CONFIG = {
    "bg_color": "#FFFFFF",         # Card pure white
    "plot_bg": "#FAFAF9",          # Off-white warm paper
    "grid_color": "#E4E4E7",       # Delicate grid hairline
    "text_color": "#09090B",       # Architectural ink
    "muted_text": "#71717A",       # Subtle zinc labels
    "spine_color": "#D4D4D8",      # Crisp hairline spine
    "cibil_poor": "#DC2626",       # Crimson
    "cibil_avg": "#D97706",        # Ochre / Amber
    "cibil_good": "#059669",       # Deep Emerald
    "income_low": "#EA580C",       # Terracotta
    "income_med": "#2563EB",       # Bauhaus Cobalt
    "income_high": "#7C3AED",      # Violet
    "output_reject": "#DC2626",    # Crimson
    "output_hold": "#D97706",      # Ochre
    "output_approve": "#059669",   # Deep Emerald
    "accent_dark": "#09090B",      # Ink black
    "accent_blue": "#2563EB",      # Cobalt
    "centroid": "#C2410C",         # Burnt sienna / terracotta
}


def _apply_artistic_light_theme(fig, ax):
    """Applies high-end, gallery/architectural light aesthetics to a matplotlib figure."""
    fig.patch.set_facecolor(STYLE_CONFIG["bg_color"])
    ax.set_facecolor(STYLE_CONFIG["plot_bg"])
    ax.tick_params(colors=STYLE_CONFIG["muted_text"], labelsize=8.5, width=0.8, length=3)
    
    # Hairline architectural borders
    for spine in ax.spines.values():
        spine.set_color(STYLE_CONFIG["spine_color"])
        spine.set_linewidth(0.8)
    
    # Subtle dot-dash grid
    ax.grid(True, linestyle=":", alpha=0.6, color=STYLE_CONFIG["grid_color"], linewidth=0.7)


def plot_cibil_mfs(fis, current_cibil: float, cibil_degs: Dict[str, float]) -> plt.Figure:
    """Plots CIBIL Score Membership Functions and marks current input in artistic light mode."""
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=140)
    _apply_artistic_light_theme(fig, ax)

    x = fis.cibil_universe
    poor_y = fis._compute_mf(x, *fis.cibil_params["Poor"])
    avg_y = fis._compute_mf(x, *fis.cibil_params["Average"])
    good_y = fis._compute_mf(x, *fis.cibil_params["Good"])

    # Base curves
    ax.plot(x, poor_y, label="Poor (≤640)", color=STYLE_CONFIG["cibil_poor"], lw=1.8)
    ax.plot(x, avg_y, label="Average (580-770)", color=STYLE_CONFIG["cibil_avg"], lw=1.8)
    ax.plot(x, good_y, label="Good (≥720)", color=STYLE_CONFIG["cibil_good"], lw=1.8)

    # Architectural translucent fills
    ax.fill_between(x, 0, poor_y, color=STYLE_CONFIG["cibil_poor"], alpha=0.07)
    ax.fill_between(x, 0, avg_y, color=STYLE_CONFIG["cibil_avg"], alpha=0.07)
    ax.fill_between(x, 0, good_y, color=STYLE_CONFIG["cibil_good"], alpha=0.07)

    # Current Operating Point: Precision caliper line
    ax.axvline(x=current_cibil, color=STYLE_CONFIG["accent_dark"], linestyle="--",
               lw=1.3, alpha=0.85, label=f"Input: {int(current_cibil)}")

    # Active membership markers
    for term, color in [("Poor", STYLE_CONFIG["cibil_poor"]),
                        ("Average", STYLE_CONFIG["cibil_avg"]),
                        ("Good", STYLE_CONFIG["cibil_good"])]:
        deg = cibil_degs[term]
        if deg > 0.01:
            ax.plot(current_cibil, deg, "o", color=color, markersize=5.5,
                    markeredgecolor="#FFFFFF", markeredgewidth=1.2)
            ax.annotate(f"μ={deg:.2f}", (current_cibil, deg),
                        textcoords="offset points", xytext=(8, 4),
                        color=color, fontsize=8, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec=color, lw=0.9))

    ax.set_title("CIBIL Score // Linguistic Partitions & Operating Point",
                 fontsize=9.5, fontweight="600", color=STYLE_CONFIG["text_color"], pad=10)
    ax.set_xlabel("CIBIL Score (300 — 900)", fontsize=8.5, color=STYLE_CONFIG["muted_text"])
    ax.set_ylabel("Membership Degree μ(x)", fontsize=8.5, color=STYLE_CONFIG["muted_text"])
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(300, 900)
    ax.legend(loc="upper right", fontsize=7.5,
              facecolor=STYLE_CONFIG["bg_color"],
              edgecolor=STYLE_CONFIG["spine_color"],
              labelcolor=STYLE_CONFIG["text_color"])
    fig.tight_layout()
    return fig


def plot_income_mfs(fis, current_income: float, income_degs: Dict[str, float]) -> plt.Figure:
    """Plots Monthly Income Membership Functions and marks current input in artistic light mode."""
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=140)
    _apply_artistic_light_theme(fig, ax)

    x = fis.income_universe
    low_y = fis._compute_mf(x, *fis.income_params["Low"])
    med_y = fis._compute_mf(x, *fis.income_params["Medium"])
    high_y = fis._compute_mf(x, *fis.income_params["High"])

    # X axis in thousands
    x_k = x / 1000.0
    current_k = current_income / 1000.0

    ax.plot(x_k, low_y, label="Low (≤₹60k)", color=STYLE_CONFIG["income_low"], lw=1.8)
    ax.plot(x_k, med_y, label="Medium (₹45k-135k)", color=STYLE_CONFIG["income_med"], lw=1.8)
    ax.plot(x_k, high_y, label="High (≥₹110k)", color=STYLE_CONFIG["income_high"], lw=1.8)

    ax.fill_between(x_k, 0, low_y, color=STYLE_CONFIG["income_low"], alpha=0.07)
    ax.fill_between(x_k, 0, med_y, color=STYLE_CONFIG["income_med"], alpha=0.07)
    ax.fill_between(x_k, 0, high_y, color=STYLE_CONFIG["income_high"], alpha=0.07)

    # Current Operating Point
    ax.axvline(x=current_k, color=STYLE_CONFIG["accent_blue"], linestyle="--",
               lw=1.3, alpha=0.85, label=f"Input: ₹{current_k:.1f}k")

    for term, color in [("Low", STYLE_CONFIG["income_low"]),
                        ("Medium", STYLE_CONFIG["income_med"]),
                        ("High", STYLE_CONFIG["income_high"])]:
        deg = income_degs[term]
        if deg > 0.01:
            ax.plot(current_k, deg, "o", color=color, markersize=5.5,
                    markeredgecolor="#FFFFFF", markeredgewidth=1.2)
            ax.annotate(f"μ={deg:.2f}", (current_k, deg),
                        textcoords="offset points", xytext=(8, 4),
                        color=color, fontsize=8, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec=color, lw=0.9))

    ax.set_title("Monthly Net Income // Linguistic Partitions & Operating Point",
                 fontsize=9.5, fontweight="600", color=STYLE_CONFIG["text_color"], pad=10)
    ax.set_xlabel("Monthly Net Income (₹ in Thousands)", fontsize=8.5, color=STYLE_CONFIG["muted_text"])
    ax.set_ylabel("Membership Degree μ(y)", fontsize=8.5, color=STYLE_CONFIG["muted_text"])
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(15, 250)
    ax.legend(loc="upper right", fontsize=7.5,
              facecolor=STYLE_CONFIG["bg_color"],
              edgecolor=STYLE_CONFIG["spine_color"],
              labelcolor=STYLE_CONFIG["text_color"])
    fig.tight_layout()
    return fig


def plot_defuzzification_results(fis, inference_result: Dict[str, Any]) -> plt.Figure:
    """
    Plots the Mamdani Aggregated Output Surface and Defuzzified Centroid (Center of Gravity).
    Artistic light mode with architectural clarity.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)
    _apply_artistic_light_theme(fig, ax)

    y = fis.approval_universe
    agg = inference_result["aggregated_curve"]
    centroid = inference_result["crisp_score"]
    term_curves = inference_result["term_curves"]

    # Base linguistic curves (faint dotted lines)
    ax.plot(y, fis.output_mfs["Reject"], linestyle=":", color=STYLE_CONFIG["output_reject"],
            lw=1.1, alpha=0.35, label="Reject Base")
    ax.plot(y, fis.output_mfs["Hold"], linestyle=":", color=STYLE_CONFIG["output_hold"],
            lw=1.1, alpha=0.35, label="Hold Base")
    ax.plot(y, fis.output_mfs["Approve"], linestyle=":", color=STYLE_CONFIG["output_approve"],
            lw=1.1, alpha=0.35, label="Approve Base")

    # Implied truncated consequent curves
    ax.plot(y, term_curves["Reject"], color=STYLE_CONFIG["output_reject"], lw=1.6, label="Implied Reject")
    ax.plot(y, term_curves["Hold"], color=STYLE_CONFIG["output_hold"], lw=1.6, label="Implied Hold")
    ax.plot(y, term_curves["Approve"], color=STYLE_CONFIG["output_approve"], lw=1.6, label="Implied Approve")

    # Aggregated Fuzzy Set (Mamdani MAX S-Norm): Translucent cobalt wash
    ax.fill_between(y, 0, agg, color=STYLE_CONFIG["accent_blue"], alpha=0.14, label="Aggregated Fuzzy Set μ_agg(z)")
    ax.plot(y, agg, color=STYLE_CONFIG["accent_blue"], lw=2.0)

    # Defuzzified Centroid (Center of Gravity)
    ax.axvline(x=centroid, color=STYLE_CONFIG["centroid"], linestyle="-", lw=2.2, alpha=0.95,
               label=f"Centroid (CoG) = {centroid:.1f}%")
    ax.plot(centroid, 0.02, "^", color=STYLE_CONFIG["centroid"], markersize=9,
            markeredgecolor="#FFFFFF", markeredgewidth=1.2)

    # Editorial annotation box
    y_centroid_h = float(np.interp(centroid, y, agg))
    ax.annotate(
        f"Centroid: {centroid:.1f}%\nDecision: {inference_result['decision']}",
        xy=(centroid, max(0.25, y_centroid_h)),
        xytext=(centroid + (10 if centroid < 60 else -38), 0.74),
        arrowprops=dict(facecolor=STYLE_CONFIG["centroid"], edgecolor=STYLE_CONFIG["centroid"],
                        arrowstyle="->", lw=1.2),
        fontsize=8.5, fontweight="600", color=STYLE_CONFIG["text_color"],
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec=STYLE_CONFIG["centroid"], lw=1.2)
    )

    ax.set_title("Mamdani Output Universe // Aggregated Surface & Centroid Defuzzification",
                 fontsize=9.5, fontweight="600", color=STYLE_CONFIG["text_color"], pad=10)
    ax.set_xlabel("Loan Approval Chance Universe z (%)", fontsize=8.5, color=STYLE_CONFIG["muted_text"])
    ax.set_ylabel("Aggregated Membership μ_agg(z)", fontsize=8.5, color=STYLE_CONFIG["muted_text"])
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", fontsize=7.2,
              facecolor=STYLE_CONFIG["bg_color"],
              edgecolor=STYLE_CONFIG["spine_color"],
              labelcolor=STYLE_CONFIG["text_color"], ncol=2)
    fig.tight_layout()
    return fig
