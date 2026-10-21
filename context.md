# CrediFuzz AI — Project Context & Technical Reference

> **Document Purpose**: Comprehensive architectural, mathematical, and implementation reference for developers and future AI models working on or extending the **CrediFuzz AI** Fuzzy Logic Loan Approval system.

---

## 1. Executive Summary & Philosophy

**CrediFuzz AI** is an intelligent, explainable credit underwriting system built using a **Mamdani Fuzzy Inference System (FIS)**. Unlike opaque black-box machine learning models (e.g., Deep Neural Networks, XGBoost), fuzzy inference models human underwriting logic with mathematical rigor, transparent boundary handling, and complete explainability.

### Key Architectural Pillars
- **Zero Heavy FIS Dependencies**: Implemented strictly from first principles using **NumPy**—no `scikit-fuzzy`, `fuzzywuzzy`, or `simpful`. This ensures zero deprecation risk, minimal installation footprint, and high execution speed.
- **Epistemic Uncertainty Handling**: Captures real-world ambiguity in credit criteria (e.g., whether an income of ₹62,000 is "Medium" or "Low", or a 675 CIBIL score is "Average" or "Good").
- **Deterministic & Auditable**: Every credit decision produces an audit trace showing active rules, firing strengths ($w_i$), membership degrees ($\mu$), and defuzzification curves.
- **Fintech Dashboard**: Interactive dark-mode user interface powered by **Streamlit**, bespoke CSS glassmorphism, dynamic SVG gauge visualizer, and dark-theme **Matplotlib** membership curves.

---

## 2. Codebase Architecture & File Map

```
Fuzzy-Logic-Loan-Approval/
│
├── app.py                  # Main Streamlit dashboard application & presentation layer
├── fuzzy_engine.py         # Pure NumPy Mamdani Fuzzy Inference Engine (Core Logic)
├── visualization.py        # Dark-mode Matplotlib visualizers for MFs & Defuzzification
├── requirements.txt        # Minimal Python dependencies (numpy, matplotlib, streamlit)
├── banner.jpg              # High-tech dashboard banner visual
├── context.md              # Technical specification & context for AI models/developers
└── README.md               # User-facing overview & quickstart documentation
```

### File Responsibilities & Modules

| File | Primary Responsibilities | Key Symbols / Classes |
| :--- | :--- | :--- |
| [`fuzzy_engine.py`](file:///c:/Users/hp/Desktop/Fuzzy-Logic-Loan-Approval/fuzzy_engine.py) | Mathematical definition of membership functions, universes of discourse, 9-rule Mamdani base, MIN-MAX implication/aggregation, and Centroid defuzzification. | `triangular_mf`, `trapezoidal_mf`, `LoanFuzzyInferenceSystem` (`fuzzify`, `evaluate_rules`, `defuzzify_centroid`, `infer`) |
| [`visualization.py`](file:///c:/Users/hp/Desktop/Fuzzy-Logic-Loan-Approval/visualization.py) | Generates dark-theme, publication-grade Matplotlib plots with headless backend (`Agg`). | `plot_cibil_mfs`, `plot_income_mfs`, `plot_defuzzification_results`, `STYLE_CONFIG` |
| [`app.py`](file:///c:/Users/hp/Desktop/Fuzzy-Logic-Loan-Approval/app.py) | Streamlit layout, session state management, input sliders, presets, dynamic SVG gauge rendering via `st.components.v1.html`, rule matrix, and audit trail. | `render_html`, `render_modern_gauge`, `set_scenario`, main UI loop |
| [`requirements.txt`](file:///c:/Users/hp/Desktop/Fuzzy-Logic-Loan-Approval/requirements.txt) | Dependency pin specifications (`numpy>=1.24.0`, `matplotlib>=3.7.0`, `streamlit>=1.30.0`). | — |

---

## 3. Mathematical Specifications

### 3.1 Universes of Discourse ($U$)
- **Input 1: CIBIL Credit Score** ($x \in [300, 900]$), discretized with $N = 601$ points ($\Delta = 1.0$).
- **Input 2: Monthly Net Income** ($y \in [15\,000, 250\,000]$ INR), discretized with $N = 1000$ points.
- **Output: Loan Approval Chance** ($z \in [0, 100]\%$), discretized with $N = 1001$ points ($\Delta = 0.1\%$).

### 3.2 Membership Function Formulations

#### Triangular Membership Function $\mu_{\text{tri}}(x; a, b, c)$
$$\mu_{\text{tri}}(x; a, b, c) = \max\left(0, \min\left(\frac{x - a}{b - a}, \frac{c - x}{c - b}\right)\right)$$
- Peak at $x = b$ where $\mu = 1.0$.
- Zero outside $[a, c]$.

#### Trapezoidal Membership Function $\mu_{\text{trap}}(x; a, b, c, d)$
$$\mu_{\text{trap}}(x; a, b, c, d) = \max\left(0, \min\left(\frac{x - a}{b - a}, 1, \frac{d - x}{d - c}\right)\right)$$
- Shoulders with boundary-safe clamping:
  - If $a = b$: Left boundary shoulder (flat $\mu=1$ for $x \le a$).
  - If $c = d$: Right boundary shoulder (flat $\mu=1$ for $x \ge d$).
  - Plateau at $\mu = 1.0$ for $x \in [b, c]$.

### 3.3 Linguistic Parameter Definitions

#### Input 1: CIBIL Score Partitions
- **Poor**: Trapezoidal $[300, 300, 520, 640]$ *(Left shoulder)*
- **Average**: Triangular $[580, 680, 770]$
- **Good**: Trapezoidal $[720, 800, 900, 900]$ *(Right shoulder)*

#### Input 2: Monthly Income Partitions (INR ₹)
- **Low**: Trapezoidal $[15\,000, 15\,000, 35\,000, 60\,000]$ *(Left shoulder)*
- **Medium**: Triangular $[45\,000, 85\,000, 135\,000]$
- **High**: Trapezoidal $[110\,000, 160\,000, 250\,000, 250\,000]$ *(Right shoulder)*

#### Output: Loan Approval Chance Partitions (%)
- **Reject**: Trapezoidal $[0, 0, 20, 45]$ *(Left shoulder)*
- **Hold**: Triangular $[35, 55, 75]$
- **Approve**: Trapezoidal $[65, 82, 100, 100]$ *(Right shoulder)*

---

## 4. Mamdani Rule Base (3 × 3 Grid)

The system defines 9 exhaustive rules covering all input combinations:

| Rule ID | CIBIL Premise | Income Premise | Consequent Output | Business / Underwriting Rationale |
| :---: | :---: | :---: | :---: | :--- |
| **1** | Poor | Low | **Reject** | Severe default risk; low repayment ability and poor credit history. |
| **2** | Poor | Medium | **Reject** | Average income insufficient to offset poor repayment track record. |
| **3** | Poor | High | **Hold** | High income provides cashflow, but bad credit requires collateral/scrutiny. |
| **4** | Average | Low | **Reject** | Moderate credit history impaired by constrained debt servicing capability. |
| **5** | Average | Medium | **Hold** | Standard profile requiring standard manual verification and bank statements. |
| **6** | Average | High | **Approve** | Substantial income cushion compensates for average credit history. |
| **7** | Good | Low | **Hold** | Excellent credit habits, but low income limits loan sizing / capacity. |
| **8** | Good | Medium | **Approve** | Low risk profile; strong repayment integrity and sustainable income. |
| **9** | Good | High | **Approve** | Prime borrower; priority processing and maximum approval limit. |

### Inference Logic:
1. **Conjunction (AND Operator)**:
   $$w_i = \min\left(\mu_{A_i}(x_{\text{cibil}}), \mu_{B_i}(y_{\text{income}})\right)$$
2. **Mamdani Implication (MIN Truncation)**:
   $$\mu_{\text{rule } i}(z) = \min\left(w_i, \mu_{C_i}(z)\right)$$
3. **Aggregation (S-Norm MAX Operator)**:
   $$\mu_{\text{aggregated}}(z) = \max_{i=1}^9 \mu_{\text{rule } i}(z)$$
4. **Centroid Defuzzification (Center of Gravity - CoG)**:
   $$z^* = \frac{\int z \cdot \mu_{\text{aggregated}}(z) \, dz}{\int \mu_{\text{aggregated}}(z) \, dz} \approx \frac{\sum_{k=1}^N z_k \cdot \mu_{\text{aggregated}}(z_k)}{\sum_{k=1}^N \mu_{\text{aggregated}}(z_k)}$$
   *Note: If no rules fire ($\sum \mu = 0$), fallback neutral value is $50.0\%$.*

---

## 5. Decision Thresholds & Risk Categorization

| Defuzzified Score ($z^*$) | Decision Output | Risk Tier | Status Badge | Hex Color | Underwriting Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$\ge 68.0\%$** | `APPROVED` | Tier 1: Prime / Low Credit Risk | `LOW RISK` | `#00E676` | Qualifies for standard & preferential interest rates with expedited disbursement. |
| **$42.0\% \le z^* < 68.0\%$** | `ON HOLD / MANUAL REVIEW` | Tier 2: Moderate Credit Risk | `MODERATE RISK` | `#FF9100` | Requires manual underwriting, bank statement verification, or co-applicant/collateral. |
| **$< 42.0\%$** | `REJECTED` | Tier 3: Subprime / High Default Risk | `HIGH RISK` | `#FF1744` | Does not meet minimum credit-to-income threshold. Recommend credit rebuilding. |

---

## 6. Data Contract (`LoanFuzzyInferenceSystem.infer`)

Calling `fis.infer(cibil, income)` returns a dictionary adhering to this schema:

```python
{
    "cibil_input": float,                      # Raw CIBIL input (e.g. 745.0)
    "income_input": float,                     # Raw Income input (e.g. 85000.0)
    "cibil_degrees": {                         # Fuzzified membership degrees
        "Poor": float,
        "Average": float,
        "Good": float
    },
    "income_degrees": {                        # Fuzzified membership degrees
        "Low": float,
        "Medium": float,
        "High": float
    },
    "rule_results": [                          # All 9 rules with individual evaluations
        {
            "id": int,                         # 1 to 9
            "cibil_term": str,
            "income_term": str,
            "consequent": str,
            "description": str,
            "weight": float,                   # Premise fire strength w_i in [0, 1]
            "implied_curve": np.ndarray        # Array of length 1001 (truncated MF)
        },
        ...
    ],
    "active_rules": list,                      # Filtered rules where weight > 0, sorted descending
    "approval_universe": np.ndarray,           # 1001 points from 0.0 to 100.0
    "aggregated_curve": np.ndarray,            # 1001 points of combined fuzzy set
    "term_curves": {                           # Truncated curves grouped by consequent
        "Reject": np.ndarray,
        "Hold": np.ndarray,
        "Approve": np.ndarray
    },
    "crisp_score": float,                      # Defuzzified score (e.g. 72.68)
    "decision": str,                           # "APPROVED" | "ON HOLD / MANUAL REVIEW" | "REJECTED"
    "decision_code": str,                      # "APPROVE" | "HOLD" | "REJECT"
    "decision_color": str,                     # "#00E676" | "#FF9100" | "#FF1744"
    "risk_tier": str,                          # Human-readable risk tier string
    "risk_badge": str,                         # "LOW RISK" | "MODERATE RISK" | "HIGH RISK"
    "recommendation": str                      # Practical underwriting guidance text
}
```

---

## 7. Operational & Execution Guide

### Local Environment Setup
- **Python Version**: Python 3.9+ (Tested and verified on Python 3.13)
- **Install Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Application
- Run with Streamlit:
  ```bash
  streamlit run app.py --server.headless true --server.port 8501
  ```
- **Access URL**: `http://localhost:8501`

### Headless Verification
To verify code execution without a browser:
```python
from fuzzy_engine import LoanFuzzyInferenceSystem

fis = LoanFuzzyInferenceSystem()
result = fis.infer(cibil=750, income=85000)
print(f"Centroid Score: {result['crisp_score']}% | Decision: {result['decision']}")
```

---

## 8. Extension Roadmap for Future Models

When extending or refactoring this system, consider the following prioritized architectural enhancements:

### 1. Multi-Dimensional Input Expansion
Currently, the system uses 2 input variables. The Mamdani engine is modular and easily expandable to $N$ inputs:
- **Debt-to-Income (DTI) Ratio** ($0\% - 100\%$): Linguistic terms `Low (<30%)`, `Moderate (30-50%)`, `Critical (>50%)`.
- **Employment Stability / Experience** ($0 - 25$ years): Linguistic terms `Novice (<2y)`, `Stable (2-7y)`, `Senior (>7y)`.
- **Collateral / Loan-to-Value (LTV)**: `Low LTV (<60%)`, `Standard (60-80%)`, `High LTV (>80%)`.

### 2. Sugeno (TSK) FIS Alternative
Implement a Takagi-Sugeno-Kang engine alongside Mamdani:
- Sugeno consequents are crisp linear functions: $z_i = k_{i0} + k_{i1} \cdot x_{\text{cibil}} + k_{i2} \cdot y_{\text{income}}$.
- Advantage: Computationally faster defuzzification via weighted average without discrete numerical integration over universes.

### 3. REST / Microservice API Layer
Wrap `fuzzy_engine.py` into a **FastAPI** service:
- `POST /api/v1/underwrite` accepting `{ "cibil": int, "monthly_income": float }`
- Returning the JSON contract defined in Section 6.
- Allows decoupled frontends (React, Vue, mobile apps) or backend banking integrations.

### 4. Automated Testing Suite
Create a `tests/` directory with `pytest`:
- **Boundary Tests**: Verify $x < 300$, $x > 900$, $y < 15\,000$, and $y > 250\,000$ clamp gracefully.
- **Monotonicity Tests**: Verify that increasing CIBIL with constant income never decreases the crisp score.
- **Defuzzification Consistency**: Confirm centroid integration accurately returns expected midpoints for symmetric inputs.

### 5. ANFIS (Adaptive Neuro-Fuzzy Inference System) Tuning
Use historical bank approval/default datasets to auto-tune membership parameters ($a, b, c, d$) via backpropagation or genetic algorithms, while preserving the interpretable rule matrix.
