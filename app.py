"""
=============================================================================
CrediFuzz AI — Architectural Credit Underwriting System
Mamdani Fuzzy Inference System for Epistemic Loan Assessment
Artistic Light Theme Edition // Minimalist Gallery & Editorial Aesthetic
=============================================================================
"""

import textwrap
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt

# Import self-made modular fuzzy logic engine and light-mode visualizer
from fuzzy_engine import LoanFuzzyInferenceSystem
from visualization import plot_cibil_mfs, plot_income_mfs, plot_defuzzification_results


# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CrediFuzz AI — Epistemic Underwriting System",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_html(html_content: str):
    """Safely renders raw HTML directly into the page DOM."""
    st.html(textwrap.dedent(html_content).strip())


# -----------------------------------------------------------------------------
# 2. EDITORIAL SVG VECTOR ICONS (Zero Emojis)
# -----------------------------------------------------------------------------
def _icon(name: str, size: int = 14, color: str = "#71717A") -> str:
    """Returns an ultra-clean, minimal inline Lucide SVG vector icon."""
    _paths = {
        "compass": (
            '<circle cx="12" cy="12" r="10"/>'
            '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>'
        ),
        "sliders": (
            '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/>'
            '<line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/>'
            '<line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/>'
            '<line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/>'
            '<line x1="18" x2="22" y1="16" y2="16"/>'
        ),
        "layers": (
            '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 '
            '1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 '
            '0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>'
        ),
        "activity": (
            '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 '
            '0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>'
        ),
        "grid": (
            '<rect width="18" height="18" x="3" y="3" rx="2"/>'
            '<path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/>'
        ),
        "bar-chart": (
            '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M7 16h8"/><path d="M7 11h12"/><path d="M7 6h3"/>'
        ),
        "list": (
            '<line x1="8" x2="21" y1="6" y2="6"/><line x1="8" x2="21" y1="12" y2="12"/>'
            '<line x1="8" x2="21" y1="18" y2="18"/><line x1="3" x2="3.01" y1="6" y2="6"/>'
            '<line x1="3" x2="3.01" y1="12" y2="12"/><line x1="3" x2="3.01" y1="18" y2="18"/>'
        ),
        "cpu": (
            '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/>'
            '<path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/>'
            '<path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>'
        ),
        "math": '<path d="M18 7V4H6l6 8-6 8h12v-3"/>',
        "check": '<path d="M20 6 9 17l-5-5"/>',
    }
    inner = _paths.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:middle;flex-shrink:0;">{inner}</svg>'
    )


# -----------------------------------------------------------------------------
# 3. ARTISTIC LIGHT THEME CSS — Architectural Gallery & Swiss Design System
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Global Font & Text Tone */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    color: #09090B !important;
}

/* Light Architectural Drafting Canvas */
.stApp {
    background-color: #FAFAF9 !important;
    background-image: 
        linear-gradient(to right, rgba(0, 0, 0, 0.035) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(0, 0, 0, 0.035) 1px, transparent 1px) !important;
    background-size: 32px 32px !important;
    color: #09090B !important;
}

/* Transparent header with no background, border, or shadow */
header[data-testid="stHeader"],
.stAppHeader {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    height: auto !important;
    pointer-events: none !important;
}

/* Toolbar container inside header */
[data-testid="stToolbar"],
.stAppToolbar {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    pointer-events: none !important;
}

/* Hide rainbow decoration line at the top */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Hide Deploy button, Status, Menu, and header right actions */
[data-testid="stHeaderActionElements"],
[data-testid="stToolbarActions"],
.stAppToolbar > div > div:last-child,
div.e3g0k5y6,
#MainMenu,
footer {
    display: none !important;
    visibility: hidden !important;
}

/* Ensure the reopen/expand sidebar button (chevron/burger) is always visible and clickable */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
button[aria-label="Expand sidebar"] {
    pointer-events: auto !important;
    display: inline-flex !important;
    visibility: visible !important;
    background: #FFFFFF !important;
    border: 1px solid #E4E4E7 !important;
    border-radius: 6px !important;
    padding: 6px !important;
    margin: 8px 0 0 10px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 4px 12px rgba(0, 0, 0, 0.04) !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover,
button[aria-label="Expand sidebar"]:hover {
    background: #F4F4F5 !important;
    border-color: #09090B !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08) !important;
}

[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
button[aria-label="Expand sidebar"] svg {
    color: #09090B !important;
    fill: #09090B !important;
    stroke: #09090B !important;
}

/* Collapse button inside sidebar */
[data-testid="stSidebarCollapseButton"] {
    pointer-events: auto !important;
    color: #71717A !important;
}

[data-testid="stSidebarCollapseButton"]:hover {
    color: #09090B !important;
}

/* Container Width & Spacing */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1280px !important;
}

/* Editorial Minimalist White Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid #E4E4E7 !important;
    border-radius: 10px !important;
    padding: 18px 22px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02), 0 6px 16px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #A1A1AA !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03), 0 12px 24px rgba(0, 0, 0, 0.03) !important;
}

/* Sidebar: Architectural Drafting Panel */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E4E4E7 !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.015) !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FAFAF9 !important;
    border: 1px solid #E4E4E7 !important;
}

/* Scenario Preset Buttons: Monograph Museum Index Style */
div.stButton > button {
    background: #FFFFFF !important;
    color: #09090B !important;
    border: 1px solid #E4E4E7 !important;
    border-radius: 6px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    padding: 8px 12px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    transition: all 0.15s ease !important;
    width: 100%;
}

div.stButton > button:hover {
    background: #F4F4F5 !important;
    border-color: #09090B !important;
    color: #09090B !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06) !important;
}

/* Native Sliders Customization */
div[data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #09090B !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* Precision Tabs */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #71717A !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #09090B !important;
    border-bottom-color: #09090B !important;
    font-weight: 600 !important;
}

/* Minimalist Expander */
details {
    background: #FFFFFF !important;
    border: 1px solid #E4E4E7 !important;
    border-radius: 8px !important;
    margin-bottom: 12px !important;
}

summary {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #09090B !important;
    padding: 12px 16px !important;
}

/* Key Quantitative Metric Tiles */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 14px 0 0 0;
}

.metric-tile {
    background: #FAFAF9;
    border: 1px solid #E4E4E7;
    border-radius: 8px;
    padding: 12px 14px;
    text-align: left;
    position: relative;
    overflow: hidden;
}

.metric-tile::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;
    background: #E4E4E7;
}

.metric-tile:hover::before {
    background: #09090B;
}

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    color: #71717A;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #09090B;
}

/* Architectural Section Tags */
.section-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    color: #71717A;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
}

/* Fuzzification Horizontal Meters */
.fuzz-track {
    background: #F4F4F5;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 4px;
}

.fuzz-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

/* 3x3 Mamdani Decision Matrix Tensor */
.matrix-grid {
    display: grid;
    grid-template-columns: 80px repeat(3, 1fr);
    gap: 5px;
    margin-top: 10px;
    font-family: 'Space Grotesk', sans-serif;
}

.matrix-header {
    background: #FAFAF9;
    border: 1px solid #E4E4E7;
    padding: 8px 4px;
    text-align: center;
    border-radius: 5px;
    font-weight: 600;
    font-size: 0.72rem;
    color: #52525B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.matrix-cell {
    background: #FFFFFF;
    border: 1px solid #E4E4E7;
    padding: 8px 6px;
    text-align: center;
    border-radius: 5px;
    transition: all 0.15s ease;
}

.matrix-cell-active {
    background: rgba(37, 99, 235, 0.04);
    border: 1px solid #2563EB;
    border-left: 3px solid #2563EB;
}

/* Active Rule Ledger Row */
.rule-row {
    background: #FAFAF9;
    border: 1px solid #E4E4E7;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.conseq-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
</style>
"""

render_html(CUSTOM_CSS)


# -----------------------------------------------------------------------------
# 4. ABSTRACT ARTISTIC RADIAL INSTRUMENT (SVG Gauge in Light Mode)
# -----------------------------------------------------------------------------
def render_artistic_radial_gauge(score: float, decision: str, decision_color: str, risk_badge: str):
    """
    Renders an abstract architectural dial/compass gauge inspired by Swiss typography
    and precision drafting instruments.
    """
    # Trigonometry for needle
    rotation_deg = -90.0 + (score / 100.0) * 180.0
    rad = np.radians(rotation_deg)

    # Needle tip and counter-weight tail
    cx, cy = 160.0, 140.0
    tip_len = 82.0
    tail_len = 18.0

    nx = cx + tip_len * np.cos(rad)
    ny = cy + tip_len * np.sin(rad)
    bx = cx - tail_len * np.cos(rad)
    by = cy - tail_len * np.sin(rad)

    # Architectural tick marks every 15 degrees (-90 to +90)
    tick_marks = []
    for deg in range(-90, 91, 15):
        r_angle = np.radians(deg)
        t_inner = 106.0
        t_outer = 114.0 if deg in (-90, -45, 0, 45, 90) else 110.0
        x1 = cx + t_inner * np.cos(r_angle)
        y1 = cy + t_inner * np.sin(r_angle)
        x2 = cx + t_outer * np.cos(r_angle)
        y2 = cy + t_outer * np.sin(r_angle)
        t_stroke = "#A1A1AA" if deg in (-90, -45, 0, 45, 90) else "#E4E4E7"
        t_width = "1.5" if deg in (-90, -45, 0, 45, 90) else "0.9"
        tick_marks.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{t_stroke}" stroke-width="{t_width}"/>')

    ticks_svg = "".join(tick_marks)

    gauge_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600&family=Space+Grotesk:wght@600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');
body {{
    margin: 0;
    padding: 0;
    background: transparent;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #09090B;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}}
.instrument-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
}}
.score-readout {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #09090B;
    margin-top: -6px;
    line-height: 1;
    letter-spacing: -0.02em;
}}
.decision-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 12px;
    border-radius: 999px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: {decision_color}0E;
    color: {decision_color};
    border: 1px solid {decision_color}33;
    margin-top: 6px;
}}
.risk-sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: #334155;
    margin-top: 4px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
</style>
</head>
<body>
<div class="instrument-wrap">
    <svg viewBox="0 0 320 175" style="width: 100%; max-width: 270px;" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="artArcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#DC2626"/>
                <stop offset="38%" stop-color="#D97706"/>
                <stop offset="70%" stop-color="#059669"/>
                <stop offset="100%" stop-color="#047857"/>
            </linearGradient>
        </defs>

        <!-- Delicate outer concentric reference line -->
        <path d="M 40 140 A 120 120 0 0 1 280 140" fill="none" stroke="#F4F4F5" stroke-width="1" stroke-dasharray="2 3"/>

        <!-- Tick marks -->
        {ticks_svg}

        <!-- Scale Track Background -->
        <path d="M 52 140 A 108 108 0 0 1 268 140" fill="none" stroke="#F4F4F5" stroke-width="12" stroke-linecap="round"/>

        <!-- Artistic Gradient Value Arc -->
        <path d="M 52 140 A 108 108 0 0 1 268 140" fill="none" stroke="url(#artArcGrad)" stroke-width="9" stroke-linecap="round" opacity="0.85"/>

        <!-- Caliper/Compass Needle Line -->
        <line x1="{bx:.1f}" y1="{by:.1f}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#09090B" stroke-width="1.8" stroke-linecap="round"/>

        <!-- Instrument Center Hub (Architectural ring) -->
        <circle cx="160" cy="140" r="5.5" fill="#09090B"/>
        <circle cx="160" cy="140" r="2.2" fill="#FFFFFF"/>

        <!-- Cardinal Scale Indicators -->
        <text x="36" y="162" font-family="'JetBrains Mono', monospace" font-size="7.5" font-weight="600" fill="#DC2626">0%</text>
        <text x="150" y="24" font-family="'JetBrains Mono', monospace" font-size="7.5" font-weight="600" fill="#D97706">50%</text>
        <text x="272" y="162" font-family="'JetBrains Mono', monospace" font-size="7.5" font-weight="600" fill="#059669">100%</text>
    </svg>

    <div class="score-readout">
        {score:.1f}<span style="font-size: 1.05rem; font-weight: 500; color: #71717A;">%</span>
    </div>
    <div class="decision-pill">
        ● {decision}
    </div>
    <div class="risk-sub">
        RISK PROFILE // <span style="color: {decision_color}; font-weight: 800;">{risk_badge}</span>
    </div>
</div>
</body>
</html>
"""
    components.html(gauge_html, height=240, scrolling=False)


# -----------------------------------------------------------------------------
# 5. HARMONIZED ARTISTIC DECISION PALETTE
# -----------------------------------------------------------------------------
MUTED_DECISION_COLORS = {
    "#00E676": "#047857",  # Deep emerald
    "#FF9100": "#B45309",  # Warm ochre amber
    "#FF1744": "#B91C1C",  # Deep crimson
}


# -----------------------------------------------------------------------------
# 6. FUZZY INFERENCE SYSTEM INITIALIZATION
# -----------------------------------------------------------------------------
if "fis" not in st.session_state:
    st.session_state.fis = LoanFuzzyInferenceSystem()

fis = st.session_state.fis

if "cibil_val" not in st.session_state:
    st.session_state.cibil_val = 745
if "income_val" not in st.session_state:
    st.session_state.income_val = 85000


def set_scenario(cibil: int, income: int):
    st.session_state.cibil_val = cibil
    st.session_state.income_val = income


# -----------------------------------------------------------------------------
# 7. SIDEBAR — Controls, Scenarios & Engine Architecture
# -----------------------------------------------------------------------------
with st.sidebar:
    # Editorial Brand Identity
    st.markdown(f"""
    <div style="padding: 6px 0 18px 0;">
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.28rem; font-weight: 700; color: #09090B; letter-spacing: -0.02em;">
            CrediFuzz <span style="font-weight: 400; color: #71717A;">// AI</span>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #71717A; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;">
            Epistemic Underwriting System
        </div>
    </div>
    <hr style="border: none; border-top: 1px solid #E4E4E7; margin: 0 0 16px 0;">
    """, unsafe_allow_html=True)

    # Parameter Calibration Section
    st.markdown(f"""
    <div class="section-tag">
        {_icon("sliders", 14, "#09090B")} [01] PARAMETERS
    </div>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
        <span style="font-size:0.82rem; font-weight:600; color:#09090B;">CIBIL Credit Score</span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#71717A;">300 — 900</span>
    </div>
    """, unsafe_allow_html=True)

    cibil_input = st.slider(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=st.session_state.cibil_val,
        step=5,
        label_visibility="collapsed",
        key="cibil_slider",
        on_change=lambda: st.session_state.update(cibil_val=st.session_state.cibil_slider)
    )

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px; margin-bottom:2px;">
        <span style="font-size:0.82rem; font-weight:600; color:#09090B;">Monthly Net Income</span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#2563EB; font-weight:600;">
            ₹{st.session_state.income_val:,}
        </span>
    </div>
    """, unsafe_allow_html=True)

    income_input = st.slider(
        "Monthly Income",
        min_value=15000,
        max_value=250000,
        value=st.session_state.income_val,
        step=2500,
        label_visibility="collapsed",
        key="income_slider",
        on_change=lambda: st.session_state.update(income_val=st.session_state.income_slider)
    )

    st.markdown('<hr style="border:none; border-top:1px solid #E4E4E7; margin:16px 0;">', unsafe_allow_html=True)

    # Reference Archetypes (Presets)
    st.markdown(f"""
    <div class="section-tag">
        {_icon("layers", 14, "#09090B")} [02] ARCHETYPES
    </div>
    """, unsafe_allow_html=True)

    if st.button("I · Prime Profile (790, ₹1.5L)"):
        set_scenario(790, 150000)
        st.rerun()
    if st.button("II · Borderline Case (660, ₹70k)"):
        set_scenario(660, 70000)
        st.rerun()
    if st.button("III · Subprime Risk (480, ₹28k)"):
        set_scenario(480, 28000)
        st.rerun()
    if st.button("IV · High Liquidity (540, ₹2.1L)"):
        set_scenario(540, 210000)
        st.rerun()

    st.markdown('<hr style="border:none; border-top:1px solid #E4E4E7; margin:16px 0;">', unsafe_allow_html=True)

    # Architectural Specifications
    st.markdown(f"""
    <div class="section-tag">
        {_icon("cpu", 14, "#09090B")} [03] SPECIFICATIONS
    </div>
    <div style="font-size:0.78rem; margin-bottom:6px; display:flex; justify-content:space-between;">
        <span style="color:#71717A;">Inference Model</span>
        <span style="font-family:'JetBrains Mono',monospace; font-weight:600; color:#09090B;">Mamdani MIN-MAX</span>
    </div>
    <div style="font-size:0.78rem; margin-bottom:6px; display:flex; justify-content:space-between;">
        <span style="color:#71717A;">Defuzzification</span>
        <span style="font-family:'JetBrains Mono',monospace; font-weight:600; color:#09090B;">Centroid (CoG)</span>
    </div>
    <div style="font-size:0.78rem; display:flex; justify-content:space-between;">
        <span style="color:#71717A;">Integration Mesh</span>
        <span style="font-family:'JetBrains Mono',monospace; font-weight:600; color:#09090B;">N = 1001 discrete</span>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. RUN THE INFERENCE ENGINE
# -----------------------------------------------------------------------------
results = fis.infer(cibil_input, income_input)
decision_color = MUTED_DECISION_COLORS.get(results["decision_color"], results["decision_color"])
cibil_degs = results["cibil_degrees"]
income_degs = results["income_degrees"]
max_firing_strength = max(r['weight'] for r in results['rule_results']) if results['rule_results'] else 0.0


# =============================================================================
# 9. MAIN CONTENT — EDITORIAL TOP HEADER & DECISION PROFILE
# =============================================================================
render_html(f"""
<div style="display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid #E4E4E7; padding-bottom:12px; margin-bottom:16px;">
    <div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#71717A; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:3px;">
            // MONOGRAPH ED. 2026.1 — REASONING UNDER EPISTEMIC UNCERTAINTY
        </div>
        <h1 style="font-family:'Space Grotesk',sans-serif; font-size:1.75rem; font-weight:700; color:#09090B; margin:0; letter-spacing:-0.03em;">
            CrediFuzz <span style="font-weight:400; color:#71717A;">Underwriting Portal</span>
        </h1>
    </div>
    <div style="text-align:right;">
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; background:#FFFFFF; border:1px solid #E4E4E7; padding:5px 12px; border-radius:4px; color:#52525B;">
            COGNITIVE MAMDANI ENGINE • FIRST PRINCIPLES NUMPY
        </span>
    </div>
</div>
""")

# Primary Decision Card
with st.container(border=True):
    card_left, card_right = st.columns([1.15, 0.85], gap="large")

    with card_left:
        render_html(f"""
        <div style="padding-top:2px;">
            <div class="section-tag" style="margin-bottom:8px;">
                {_icon("compass", 14, "#09090B")} [01] SYNTHESIS // LENDING PROFILE
            </div>
            <h2 style="font-family:'Space Grotesk',sans-serif; font-size:1.95rem; font-weight:700;
                       color:{decision_color}; margin:0 0 10px 0; letter-spacing:-0.03em;">
                {results['decision']}
            </h2>
            <div style="background:#FAFAF9; border-left:3px solid {decision_color};
                        padding:12px 16px; border-radius:0 6px 6px 0; margin-bottom:12px; border-top:1px solid #E4E4E7; border-right:1px solid #E4E4E7; border-bottom:1px solid #E4E4E7;">
                <div style="font-size:0.84rem; color:#334155; line-height:1.55;">
                    {results['recommendation']}
                </div>
            </div>
            <div style="font-size:0.78rem; color:#71717A;">
                Credit Classification: <span style="color:#09090B; font-weight:600;">{results['risk_tier']}</span>
            </div>
        </div>
        """)

    with card_right:
        render_artistic_radial_gauge(
            results["crisp_score"],
            results["decision"],
            decision_color,
            results["risk_badge"]
        )

    # Key Quantitative Architectural Metrics
    render_html(f"""
    <div class="metric-grid">
        <div class="metric-tile">
            <div class="metric-label">01 // Centroid Valuation</div>
            <div class="metric-value">{results['crisp_score']:.1f}%</div>
        </div>
        <div class="metric-tile">
            <div class="metric-label">02 // Active Mamdani Premises</div>
            <div class="metric-value">{len(results['active_rules'])} <span style="font-size:0.85rem; font-weight:500; color:#71717A;">/ 9</span></div>
        </div>
        <div class="metric-tile">
            <div class="metric-label">03 // Peak Implication Strength</div>
            <div class="metric-value">{max_firing_strength:.2f}</div>
        </div>
    </div>
    """)


# =============================================================================
# 10. DUAL COLUMN: FUZZIFICATION & DECISION MATRIX
# =============================================================================
col_fuzz, col_matrix = st.columns(2, gap="medium")

with col_fuzz:
    with st.container(border=True):
        fuzz_html = f"""
        <div class="section-tag">
            {_icon("activity", 14, "#09090B")} [02] CONTINUOUS MEMBERSHIP PARTITIONS
        </div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:0.74rem; font-weight:600; color:#52525B;
                    text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">
            CIBIL Score Memberships
        </div>
        """
        for term, color in [("Poor", "#DC2626"), ("Average", "#D97706"), ("Good", "#059669")]:
            deg = cibil_degs[term]
            fuzz_html += f"""
            <div style="margin-top:6px;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem;">
                    <span style="color:{color}; font-weight:600;">{term}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#52525B;">
                        μ = {deg:.3f}
                    </span>
                </div>
                <div class="fuzz-track">
                    <div class="fuzz-fill" style="width:{deg*100:.1f}%; background:{color};"></div>
                </div>
            </div>
            """

        fuzz_html += """
        <hr style="border:none; border-top:1px solid #E4E4E7; margin:14px 0;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:0.74rem; font-weight:600; color:#52525B;
                    text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">
            Monthly Income Memberships
        </div>
        """
        for term, color in [("Low", "#EA580C"), ("Medium", "#2563EB"), ("High", "#7C3AED")]:
            deg = income_degs[term]
            fuzz_html += f"""
            <div style="margin-top:6px;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem;">
                    <span style="color:{color}; font-weight:600;">{term}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#52525B;">
                        μ = {deg:.3f}
                    </span>
                </div>
                <div class="fuzz-track">
                    <div class="fuzz-fill" style="width:{deg*100:.1f}%; background:{color};"></div>
                </div>
            </div>
            """
        render_html(fuzz_html)

with col_matrix:
    with st.container(border=True):
        terms_income = ["Low", "Medium", "High"]
        terms_cibil = ["Poor", "Average", "Good"]
        conseq_map = {
            ("Poor", "Low"): ("Reject", 1), ("Poor", "Medium"): ("Reject", 2),
            ("Poor", "High"): ("Hold", 3), ("Average", "Low"): ("Reject", 4),
            ("Average", "Medium"): ("Hold", 5), ("Average", "High"): ("Approve", 6),
            ("Good", "Low"): ("Hold", 7), ("Good", "Medium"): ("Approve", 8),
            ("Good", "High"): ("Approve", 9),
        }

        matrix_html = f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div class="section-tag" style="margin-bottom:0;">
                {_icon("grid", 14, "#09090B")} [03] 3×3 RULE IMPLICATION TENSOR
            </div>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#71717A;">T-Norm: MIN</span>
        </div>
        <div class="matrix-grid">
            <div class="matrix-header">CIBIL \\ Inc</div>
            <div class="matrix-header">Low</div>
            <div class="matrix-header">Medium</div>
            <div class="matrix-header">High</div>
        """

        for c_term in terms_cibil:
            matrix_html += f'<div class="matrix-header">{c_term}</div>'
            for i_term in terms_income:
                conseq, r_id = conseq_map[(c_term, i_term)]
                pw = min(cibil_degs[c_term], income_degs[i_term])
                is_active = pw > 0.001
                active_cls = "matrix-cell-active" if is_active else ""
                c_color = "#059669" if conseq == "Approve" else "#D97706" if conseq == "Hold" else "#DC2626"
                w_text = (
                    f'<div style="font-family:JetBrains Mono,monospace; font-size:0.65rem;'
                    f'color:#2563EB; font-weight:600; margin-top:2px;">w={pw:.2f}</div>'
                    if is_active else ""
                )
                matrix_html += f"""
                <div class="matrix-cell {active_cls}">
                    <div style="font-weight:600; font-size:0.78rem; color:{c_color};">{conseq}</div>
                    {w_text}
                </div>
                """

        matrix_html += "</div>"
        render_html(matrix_html)


# =============================================================================
# 11. CONTINUOUS SURFACES & MAPPINGS (Matplotlib in Light Theme)
# =============================================================================
with st.container(border=True):
    render_html(f"""
    <div class="section-tag">
        {_icon("bar-chart", 14, "#09090B")} [04] CONTINUOUS MEMBERSHIP SURFACES
    </div>
    """)

    tab1, tab2, tab3 = st.tabs([
        "[01] Defuzzified Aggregation Surface",
        "[02] CIBIL Partitions Universe",
        "[03] Income Partitions Universe"
    ])

    with tab1:
        render_html("""<div style="font-size:0.78rem; color:#71717A; margin-bottom:8px;">
            Pointwise MAX aggregation of implied fuzzy sets. The vertical line marks the numerical Center of Gravity (Centroid).
        </div>""")
        fig_output = plot_defuzzification_results(fis, results)
        st.pyplot(fig_output, use_container_width=True)
        plt.close(fig_output)

    with tab2:
        render_html("""<div style="font-size:0.78rem; color:#71717A; margin-bottom:8px;">
            CIBIL domain universe [300, 900]: Poor (Trapezoid), Average (Triangle), Good (Trapezoid).
        </div>""")
        fig_cibil = plot_cibil_mfs(fis, cibil_input, cibil_degs)
        st.pyplot(fig_cibil, use_container_width=True)
        plt.close(fig_cibil)

    with tab3:
        render_html("""<div style="font-size:0.78rem; color:#71717A; margin-bottom:8px;">
            Monthly Income domain [₹15,000, ₹250,000]: Low (Trapezoid), Medium (Triangle), High (Trapezoid).
        </div>""")
        fig_income = plot_income_mfs(fis, income_input, income_degs)
        st.pyplot(fig_income, use_container_width=True)
        plt.close(fig_income)


# =============================================================================
# 12. ACTIVE RULE FIRE TRACE (Audit Ledger)
# =============================================================================
with st.expander("[05] ACTIVE INFERENCE TRACE // AUDIT LEDGER"):
    if results["active_rules"]:
        rule_html = ""
        for r in results["active_rules"]:
            c_color = (
                "#059669" if r['consequent'] == "Approve"
                else "#D97706" if r['consequent'] == "Hold"
                else "#DC2626"
            )
            rule_html += f"""
            <div class="rule-row">
                <div>
                    <span style="font-family:'JetBrains Mono',monospace; font-weight:700;
                                 color:#09090B; font-size:0.8rem;">RULE // 0{r['id']}</span>
                    <span style="font-size:0.8rem; color:#334155; margin-left:10px;">
                        {r['description']}
                    </span>
                </div>
                <div style="text-align:right; min-width:90px;">
                    <span class="conseq-badge"
                          style="background:{c_color}12; color:{c_color}; border:1px solid {c_color}33;">
                        {r['consequent']}
                    </span>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem;
                                color:#71717A; margin-top:2px;">
                        w = {r['weight']:.3f}
                    </div>
                </div>
            </div>
            """
        render_html(rule_html)
    else:
        st.markdown("*No active rules detected at this operating coordinate.*")


# =============================================================================
# 13. MATHEMATICAL SPECIFICATION (Academic Monograph)
# =============================================================================
with st.expander("[06] FORMAL SYSTEM SPECIFICATION // NUMPY EQUATIONS"):
    st.markdown("""
    ### 1. Triangular Membership Function
    $$\\mu_{\\text{tri}}(x; a, b, c) = \\max\\left(0, \\min\\left(\\frac{x - a}{b - a}, \\frac{c - x}{c - b}\\right)\\right)$$
    Used for:
    - **CIBIL Average**: $a=580, b=680, c=770$
    - **Income Medium**: $a=45\\text{k}, b=85\\text{k}, c=135\\text{k}$
    - **Approval Hold**: $a=35\\%, b=55\\%, c=75\\%$

    ### 2. Trapezoidal Membership Function
    $$\\mu_{\\text{trap}}(x; a, b, c, d) = \\max\\left(0, \\min\\left(\\frac{x - a}{b - a}, 1, \\frac{d - x}{d - c}\\right)\\right)$$
    Used for:
    - **CIBIL Poor**: $a=300, b=300, c=520, d=640$
    - **CIBIL Good**: $a=720, b=800, c=900, d=900$
    - **Income Low**: $a=15\\text{k}, b=15\\text{k}, c=35\\text{k}, d=60\\text{k}$
    - **Income High**: $a=110\\text{k}, b=160\\text{k}, c=250\\text{k}, d=250\\text{k}$
    - **Approval Reject**: $a=0, b=0, c=20, d=45$
    - **Approval Approve**: $a=65, b=82, c=100, d=100$

    ### 3. Mamdani Rule Evaluation & Implication
    For each rule $i$ with premise $A_i \\text{ AND } B_i \\implies C_i$:
    $$w_i = \\min(\\mu_{A_i}(x_{\\text{cibil}}), \\mu_{B_i}(y_{\\text{income}}))$$
    $$\\mu_{\\text{rule } i}(z) = \\min(w_i, \\mu_{C_i}(z))$$

    ### 4. Fuzzy Aggregation (S-Norm MAX)
    $$\\mu_{\\text{aggregated}}(z) = \\max_{i=1}^9 \\mu_{\\text{rule } i}(z)$$

    ### 5. Centroid (Center of Gravity) Defuzzification
    $$\\text{Centroid} = \\frac{\\int z \\cdot \\mu_{\\text{aggregated}}(z) \\, dz}{\\int \\mu_{\\text{aggregated}}(z) \\, dz} \\approx \\frac{\\sum_{k} z_k \\cdot \\mu_{\\text{aggregated}}(z_k)}{\\sum_{k} \\mu_{\\text{aggregated}}(z_k)}$$
    Discretized over $N=1001$ uniformly spaced points in the domain $[0, 100]$.
    """)


# =============================================================================
# 14. FOOTER // COLOPHON
# =============================================================================
render_html("""
<div style="text-align:center; color:#A1A1AA; font-family:'JetBrains Mono',monospace; font-size:0.72rem;
            margin-top:28px; padding-top:16px; border-top:1px solid #E4E4E7; letter-spacing:0.04em;">
    CREDIFUZZ AI • FORMAL FUZZY LOGIC UNDERWRITING ENGINE • NUMPY INTEGRATION • EDITION 2026.1
</div>
""")
