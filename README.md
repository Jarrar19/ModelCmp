# Dual-Dataset XAI Benchmark: Interpretable vs. Black-Box

An interactive clinical decision-support and Explainable AI (XAI) comparative study developed for **TAE 1: Problem-Based Learning in Explainable AI**.

**Topic:** *Black-Box vs. Interpretable Model Comparison* — Compare a black-box model and an interpretable model across small clinical and large-scale complex datasets to analyze performance versus explainability.

---

## 1. Executive Summary & The Core Research Question

A fundamental question in machine learning is:
> *Does data scale and non-linear complexity always justify using an uninterpretable black-box model?*

To answer this empirically, this platform implements and compares **Logistic Regression (Inherently Interpretable Glass-Box)** against **XGBoost (Black-Box Ensemble + SHAP)** across **two contrasting dataset scales and domains**:

1. **Small Clinical Benchmark (297 patients)**: **UCI Cleveland Heart Disease** (13 clinical indicators)
2. **Complex Large-Scale Benchmark (32,561 records)**: **Adult Census Income** (12 demographic & financial attributes)

---

## 2. Empirical Findings Across Dataset Scales

| Dimension | Dataset 1: Cleveland Heart Disease | Dataset 2: Adult Census Income |
| :--- | :---: | :---: |
| **Domain** | High-Stakes Cardiology | Socioeconomic / Financial |
| **Cohort Size** | **297 patients** (Small Clinical) | **32,561 records** (Large Complex) |
| **Features** | 13 clinical indicators | 12 demographic/work features |
| **Logistic Regression Accuracy** | 83.33% | 84.63% |
| **XGBoost Accuracy** | **85.00%** (+1.67%) | **87.69% (+3.06% lead)** |
| **Logistic Regression ROC-AUC** | **0.9498 (Superior by +0.0235)** | 0.8916 |
| **XGBoost ROC-AUC** | 0.9263 | **0.9305 (Superior by +0.0389)** |
| **Key Takeaway** | **Interpretable Model Wins on AUC**: On clean, small clinical data, linear regularization resists overfitting. Zero reason to sacrifice glass-box auditability. | **Black-Box Wins on Non-Linearity**: With 32k records, trees learn complex non-linear interactions (e.g. capital gain spikes, education thresholds), pulling ahead by +3.06% accuracy. |

---

## 3. Explainability Comparison

| Dimension | Logistic Regression (Glass-Box) | XGBoost (Black-Box) |
| :--- | :--- | :--- |
| **Decision Mechanism** | Exact linear weighted sum: $z = \beta_0 + \sum \beta_i x_i$ | 100+ boosted decision trees with multi-branch splits |
| **Explanation Method** | Direct linear attribution: $w_i \cdot x_i$ | Game-theoretic **SHAP (Shapley Additive exPlanations)** |
| **Explanation Fidelity** | **100% Ground-Truth Faithful** | **Post-hoc Approximation** |
| **Computational Latency** | **Microseconds ($O(D)$)** | Milliseconds to seconds ($O(T \cdot L \cdot D^2)$) |

---

## 4. Quick Start & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train both datasets and generate model artifacts (already completed)
python backend/train.py

# 3. Launch the interactive FastAPI server
cd backend
uvicorn app:app --port 8000 --reload
```

Open **`http://localhost:8000/`** in your browser.

Use the **Dataset Switcher** in the left sidebar to flip between:
- 🏥 **Cleveland Heart Disease (297 cases)**
- 📊 **Adult Census Income (32,561 cases)**

---

## 5. Viva / Evaluation Defense Guide

**Q1: What did your cross-dataset comparison prove?**  
*Answer:* It proved that the "Accuracy vs. Interpretability trade-off" is strongly dependent on **data scale and interaction complexity**. On the small clinical dataset (297 cases), Logistic Regression matched XGBoost and achieved a higher ROC-AUC (0.9498 vs. 0.9263), demonstrating that complex models are not universally superior. However, when scaling to the 32,561-record Adult Census dataset, XGBoost pulled ahead by +3.06% accuracy and +0.039 AUC because it could learn higher-order non-linear combinations that a linear hyperplane cannot represent.

**Q2: What is the downside of using XGBoost on the large dataset even though it had higher accuracy?**  
*Answer:* The computational cost of explainability. Computing SHAP Shapley values across 150 boosted trees on large datasets introduces significant inference latency ($O(T \cdot L \cdot D^2)$), whereas Logistic Regression explanations are instantaneous ($O(D)$) and 100% ground-truth faithful.

**Q3: If we want both high accuracy and 100% interpretability on large datasets, what is the modern solution?**  
*Answer:* **Explainable Boosting Machines (EBMs)** based on Generalized Additive Models (GAMs). EBMs use boosting on individual features and pairwise interactions ($g(y) = \beta_0 + \sum f_i(x_i) + \sum f_{ij}(x_i, x_j)$), capturing non-linear curves while remaining completely glass-box interpretable without requiring SHAP approximations.
