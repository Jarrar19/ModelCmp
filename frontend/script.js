const API = ""; // same-origin FastAPI server

let CURRENT_DATASET = "heart";
let FEATURES = [];
let METADATA = {};
let DATASET_CONFIG = {};

// ---------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    const target = document.getElementById(`view-${btn.dataset.view}`);
    if (target) target.classList.add("active");
  });
});

// ---------------------------------------------------------------
// Dataset Switcher
// ---------------------------------------------------------------
document.querySelectorAll(".dataset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const dName = btn.dataset.dataset;
    if (dName !== CURRENT_DATASET) {
      switchDataset(dName);
    }
  });
});

async function switchDataset(dName) {
  document.querySelectorAll(".dataset-btn").forEach((b) => b.classList.remove("active"));
  const activeBtn = document.getElementById(`side-btn-${dName}`);
  if (activeBtn) activeBtn.classList.add("active");

  CURRENT_DATASET = dName;

  // Clear results on dataset change
  const resultsEl = document.getElementById("results-wrap");
  if (resultsEl) resultsEl.innerHTML = "";

  const submitStatus = document.getElementById("submit-status");
  if (submitStatus) submitStatus.textContent = "";

  // Load new dataset details
  await loadDatasetData(dName);
}

// ---------------------------------------------------------------
// Helper API fetcher
// ---------------------------------------------------------------
async function fetchJSON(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------
// Load & Update for Active Dataset
// ---------------------------------------------------------------
async function loadDatasetData(dName) {
  const [featData, metricsData] = await Promise.all([
    fetchJSON(`/api/features?dataset=${dName}`),
    fetchJSON(`/api/metrics?dataset=${dName}`),
  ]);

  FEATURES = featData.features;
  METADATA = featData.metadata;
  DATASET_CONFIG = featData;

  // Update Banner & Headers
  const bannerTag = document.getElementById("banner-tag");
  const bannerTitle = document.getElementById("banner-title");
  const bannerThesis = document.getElementById("banner-thesis");
  const diagnoseHeading = document.getElementById("diagnose-heading");
  const diagnoseDesc = document.getElementById("diagnose-desc");
  const formTitle = document.getElementById("form-card-title");
  const formSub = document.getElementById("form-card-sub");
  const metricsTitle = document.getElementById("metrics-card-title");
  const rocImg = document.getElementById("roc-image");
  const rocSub = document.getElementById("roc-sub");

  if (dName === "heart") {
    if (bannerTag) bannerTag.textContent = "Clinical Benchmark";
    if (bannerTag) bannerTag.className = "banner-tag badge-clinical";
    if (bannerTitle) bannerTitle.textContent = "UCI Cleveland Heart Disease (297 patients)";
    if (bannerThesis) bannerThesis.textContent = "Small Clinical Cohort: Interpretable model matches or beats black-box (AUC 0.9498 vs 0.9263).";
    if (diagnoseHeading) diagnoseHeading.textContent = "Predict & Explain: Cleveland Heart Disease";
    if (diagnoseDesc) diagnoseDesc.textContent = "Enter patient clinical indicators to evaluate disease risk side-by-side using both models.";
    if (formTitle) formTitle.textContent = "Patient Clinical Measurements (13 Indicators)";
    if (formSub) formSub.textContent = "Standardized cardiology indicators from the Cleveland Clinic dataset.";
    if (metricsTitle) metricsTitle.textContent = "Benchmark Table: Cleveland Heart Disease (297 records)";
    if (rocImg) rocImg.src = "roc_curve_heart.png";
    if (rocSub) rocSub.textContent = "Heart Disease: Logistic Regression achieves AUC 0.9498 vs XGBoost 0.9263.";
  } else {
    if (bannerTag) bannerTag.textContent = "Complex Benchmark";
    if (bannerTag) bannerTag.className = "banner-tag badge-complex";
    if (bannerTitle) bannerTitle.textContent = "Adult Census Income (32,561 records)";
    if (bannerThesis) bannerThesis.textContent = "Large-Scale Data: XGBoost pulls ahead by +3.06% accuracy & +0.039 AUC via non-linear interactions.";
    if (diagnoseHeading) diagnoseHeading.textContent = "Predict & Explain: Adult Census Income (> $50K)";
    if (diagnoseDesc) diagnoseDesc.textContent = "Enter individual demographic & work profile to evaluate high-income (> $50K) likelihood.";
    if (formTitle) formTitle.textContent = "Demographic & Financial Indicators (12 Features)";
    if (formSub) formSub.textContent = "Standardized census attributes from the landmark SHAP / LIME benchmark dataset.";
    if (metricsTitle) metricsTitle.textContent = "Benchmark Table: Adult Census Income (32,561 records)";
    if (rocImg) rocImg.src = "roc_curve_adult.png";
    if (rocSub) rocSub.textContent = "Adult Census: XGBoost pulls ahead with AUC 0.9305 vs Logistic Regression 0.8916.";
  }

  buildForm();
  renderMetrics(metricsData);
}

// ---------------------------------------------------------------
// Build Form with Clinical / Attribute Groups & Dropdowns
// ---------------------------------------------------------------
function buildForm() {
  const form = document.getElementById("patient-form");
  const container = document.getElementById("form-fields") || form;

  // Derive unique categories in order
  const categories = [];
  FEATURES.forEach((fname) => {
    const meta = METADATA[fname] || {};
    const cat = meta.category || "Attributes";
    if (!categories.includes(cat)) categories.push(cat);
  });

  const grouped = {};
  categories.forEach((c) => (grouped[c] = []));

  FEATURES.forEach((fname) => {
    const meta = METADATA[fname] || { label: fname, category: "Attributes", type: "number" };
    const cat = meta.category || "Attributes";
    grouped[cat].push({ name: fname, ...meta });
  });

  let html = "";
  for (const cat of categories) {
    const fields = grouped[cat];
    if (!fields || fields.length === 0) continue;

    html += `<div class="feature-group-label">${cat}</div><div class="field-grid">`;

    fields.forEach((f) => {
      html += `<div class="field">
        <label for="f_${f.name}">
          ${f.label} ${f.unit ? `<span class="unit">(${f.unit})</span>` : ""}
        </label>`;

      if (f.type === "select" && f.options) {
        html += `<select id="f_${f.name}" data-feature="${f.name}" required class="form-select">`;
        f.options.forEach((opt) => {
          const isSel = (f.default !== undefined && opt.value === f.default) ? "selected" : "";
          html += `<option value="${opt.value}" ${isSel}>${opt.text}</option>`;
        });
        html += `</select>`;
      } else {
        const defVal = f.default !== undefined ? f.default : "";
        html += `<input 
          type="number" 
          id="f_${f.name}" 
          data-feature="${f.name}" 
          min="${f.min !== undefined ? f.min : ''}" 
          max="${f.max !== undefined ? f.max : ''}" 
          step="${f.step || 'any'}" 
          value="${defVal}"
          placeholder="${defVal ? 'e.g. ' + defVal : ''}"
          required 
          class="form-input"
        />`;
      }

      if (f.description) {
        html += `<span class="field-hint">${f.description}</span>`;
      }
      html += `</div>`;
    });

    html += `</div>`;
  }

  container.innerHTML = html;

  form.removeEventListener("submit", onSubmit);
  form.addEventListener("submit", onSubmit);

  const submitBtn = document.getElementById("submit-btn");
  if (submitBtn) {
    submitBtn.onclick = (e) => {
      e.preventDefault();
      onSubmit(e);
    };
  }
}

function readForm() {
  const features = {};
  FEATURES.forEach((fname) => {
    const meta = METADATA[fname] || {};
    const fallback = meta.default !== undefined ? meta.default : 0.0;
    const el = document.getElementById(`f_${fname}`);
    if (!el) {
      features[fname] = fallback;
      return;
    }
    const val = parseFloat(el.value);
    features[fname] = isNaN(val) ? fallback : val;
  });
  return features;
}

// ---------------------------------------------------------------
// Submit -> Predict + Explain
// ---------------------------------------------------------------
async function onSubmit(e) {
  if (e) e.preventDefault();
  const submitBtn = document.getElementById("submit-btn");
  const status = document.getElementById("submit-status");
  const features = readForm();

  if (submitBtn) submitBtn.disabled = true;
  if (status) {
    status.className = "status-note";
    status.textContent = `Running ${DATASET_CONFIG.title || 'dataset'} models…`;
  }

  try {
    const [pred, exp] = await Promise.all([
      fetchJSON(`/api/predict?dataset=${CURRENT_DATASET}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      }),
      fetchJSON(`/api/explain?dataset=${CURRENT_DATASET}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      }),
    ]);
    renderResults(pred, exp);
    if (status) status.textContent = "";

    const resultsEl = document.getElementById("results-wrap");
    if (resultsEl) resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    if (status) {
      status.className = "status-note error-note";
      status.textContent = "Error running analysis: " + err.message;
    }
    console.error("Diagnostic error:", err);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

// ---------------------------------------------------------------
// Render Results (Diagnosis, Confidence, & Explanations)
// ---------------------------------------------------------------
function renderResults(pred, exp) {
  const wrap = document.getElementById("results-wrap");
  const lrPred = pred.interpretable;
  const xgbPred = pred.blackbox;

  const isHeart = CURRENT_DATASET === "heart";
  const targetLabel = isHeart ? "Estimated Disease Probability" : "Estimated High-Income (> $50K) Likelihood";
  const posColorClass = isHeart ? "card-disease" : "card-income-high";
  const negColorClass = isHeart ? "card-healthy" : "card-income-low";

  const lrClass = lrPred.prediction_code === 1 ? posColorClass : negColorClass;
  const xgbClass = xgbPred.prediction_code === 1 ? posColorClass : negColorClass;

  const posFill = isHeart ? "fill-disease" : "fill-high-income";

  wrap.innerHTML = `
    <div class="card results-container">
      <div class="results-header">
        <h2>Model Assessment & Agreement</h2>
        <div class="agree-badge ${pred.agree ? "badge-agree" : "badge-disagree"}">
          ${
            pred.agree
              ? "✓ Both Models Agree on Classification"
              : "⚠️ Models Disagree — Conflicting Decisions!"
          }
        </div>
      </div>

      <!-- XAI INSIGHT CALLOUT -->
      <div class="insight-box">
        <div class="insight-title">💡 XAI Attribution Insight: What Drove Each Decision</div>
        <p class="insight-text">${exp.clinical_insight}</p>
        ${
          !pred.agree
            ? `<div class="insight-subnote"><strong>Viva Insight:</strong> On this borderline case, the models diverged. Logistic Regression reached its decision through fixed linear weights, while XGBoost branched through non-linear tree interactions estimated by SHAP.</div>`
            : ""
        }
      </div>

      <!-- SIDE-BY-SIDE DIAGNOSTIC CARDS -->
      <div class="result-grid">
        <div class="result-card ${lrClass}">
          <div class="model-badge">Logistic Regression · Glass-Box</div>
          <div class="pred-label">${lrPred.prediction}</div>
          <div class="prob-row">
            <span>${targetLabel}:</span>
            <strong>${(lrPred.probability * 100).toFixed(1)}%</strong>
          </div>
          <div class="prob-track">
            <div class="prob-fill ${posFill}" style="width: ${(lrPred.probability * 100).toFixed(1)}%"></div>
          </div>
          <div class="conf-note">${(lrPred.confidence * 100).toFixed(1)}% certainty in this class</div>
        </div>

        <div class="result-card ${xgbClass}">
          <div class="model-badge">XGBoost · Black-Box Ensemble</div>
          <div class="pred-label">${xgbPred.prediction}</div>
          <div class="prob-row">
            <span>${targetLabel}:</span>
            <strong>${(xgbPred.probability * 100).toFixed(1)}%</strong>
          </div>
          <div class="prob-track">
            <div class="prob-fill ${posFill}" style="width: ${(xgbPred.probability * 100).toFixed(1)}%"></div>
          </div>
          <div class="conf-note">${(xgbPred.confidence * 100).toFixed(1)}% certainty in this class</div>
        </div>
      </div>

      <!-- EXPLANATION CHARTS -->
      <div class="explanations-header">
        <h3>Feature Contributions: Glass-Box vs. Black-Box</h3>
        <p>Compare which indicators pushed each model toward or away from the target class.</p>
        <div class="legend-bar">
          <span class="legend-item"><span class="legend-dot dot-risk"></span> <strong>Red/Purple:</strong> ${DATASET_CONFIG.pos_direction_label || 'Pushes toward Class 1'}</span>
          <span class="legend-item"><span class="legend-dot dot-protective"></span> <strong>Green:</strong> ${DATASET_CONFIG.neg_direction_label || 'Pushes toward Class 0'}</span>
        </div>
      </div>

      <div class="grid-2">
        <div class="explanation-col">
          <div class="col-title">
            <span>XGBoost — SHAP Values</span>
            <span class="col-badge">Post-Hoc Tree Attribution</span>
          </div>
          <div class="bars-container">
            ${renderBars(exp.blackbox_top)}
          </div>
        </div>

        <div class="explanation-col">
          <div class="col-title">
            <span>Logistic Regression — Weights × Value</span>
            <span class="col-badge">Exact Linear Equation</span>
          </div>
          <div class="bars-container">
            ${renderBars(exp.interpretable_top)}
          </div>
        </div>
      </div>

    </div>
  `;
}

// ---------------------------------------------------------------
// Render Horizontal Contribution Bars
// ---------------------------------------------------------------
function renderBars(items) {
  const maxAbs = Math.max(...items.map((d) => Math.abs(d.contribution)), 0.001);

  return items
    .map((d) => {
      const isPositive = d.contribution > 0;
      const pct = Math.min((Math.abs(d.contribution) / maxAbs) * 100, 100);

      // Pretty formatted patient value
      let valDisplay = d.patient_value;
      const meta = METADATA[d.feature];
      if (meta && meta.options) {
        const found = meta.options.find((o) => o.value === d.patient_value);
        if (found) valDisplay = found.text;
      } else if (meta && meta.unit) {
        valDisplay = `${d.patient_value} ${meta.unit}`;
      }

      return `
        <div class="bar-row">
          <div class="bar-info">
            <span class="bar-feature-name" title="${d.label}">${d.label}</span>
            <span class="bar-patient-val">${valDisplay}</span>
          </div>
          <div class="bar-track-wrap">
            <div class="bar-track">
              <div class="bar-fill ${isPositive ? "fill-risk" : "fill-prot"}" style="width: ${pct}%;"></div>
            </div>
            <span class="bar-score ${isPositive ? "score-risk" : "score-prot"}">
              ${d.contribution > 0 ? "+" : ""}${d.contribution.toFixed(3)}
            </span>
          </div>
        </div>`;
    })
    .join("");
}

// ---------------------------------------------------------------
// Metrics Rendering
// ---------------------------------------------------------------
function renderMetrics(metricsData) {
  const current = metricsData.current || metricsData;
  const mInterp = current.interpretable;
  const mBlack = current.blackbox;

  // Overview stats
  const statName = document.getElementById("stat-dataset-name");
  const statTotal = document.getElementById("stat-total-records");
  const statLr = document.getElementById("stat-lr-auc");
  const statXgb = document.getElementById("stat-xgb-auc");

  if (statName) statName.textContent = current.dataset_name || (CURRENT_DATASET === "adult" ? "Adult Census" : "Heart Disease");
  if (statTotal) statTotal.textContent = (current.n_total || current.n_train + current.n_test).toLocaleString();
  if (statLr) statLr.textContent = mInterp.roc_auc.toFixed(3);
  if (statXgb) statXgb.textContent = mBlack.roc_auc.toFixed(3);

  // Performance Table
  const rows = [
    [
      "Accuracy",
      `${(mInterp.accuracy * 100).toFixed(1)}%`,
      `${(mBlack.accuracy * 100).toFixed(1)}%`,
      mBlack.accuracy > mInterp.accuracy 
        ? `XGBoost leads by +${((mBlack.accuracy - mInterp.accuracy) * 100).toFixed(2)}%`
        : `Logistic Regression matches within ${((mBlack.accuracy - mInterp.accuracy) * 100).toFixed(2)}%`
    ],
    [
      "ROC-AUC",
      mInterp.roc_auc.toFixed(4),
      mBlack.roc_auc.toFixed(4),
      mInterp.roc_auc > mBlack.roc_auc
        ? `Logistic Regression superior (+${(mInterp.roc_auc - mBlack.roc_auc).toFixed(4)})`
        : `XGBoost superior (+${(mBlack.roc_auc - mInterp.roc_auc).toFixed(4)})`
    ],
    [
      "Precision",
      `${(mInterp.precision * 100).toFixed(1)}%`,
      `${(mBlack.precision * 100).toFixed(1)}%`,
      "Positive predictive accuracy across threshold."
    ],
    [
      "Recall (Sensitivity)",
      `${(mInterp.recall * 100).toFixed(1)}%`,
      `${(mBlack.recall * 100).toFixed(1)}%`,
      "Ability to identify positive instances."
    ],
    [
      "F1-Score",
      mInterp.f1.toFixed(4),
      mBlack.f1.toFixed(4),
      "Harmonic balance between precision and recall."
    ],
  ];

  const tbody = document.getElementById("metrics-tbody");
  if (tbody) {
    tbody.innerHTML = rows
      .map(
        ([label, lr, xgb, note]) => `
      <tr>
        <td class="metric-name"><strong>${label}</strong></td>
        <td><span class="metric-val">${lr}</span></td>
        <td><span class="metric-val">${xgb}</span></td>
        <td class="metric-note">${note}</td>
      </tr>`
      )
      .join("");
  }
}

// ---------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------
async function init() {
  await loadDatasetData(CURRENT_DATASET);
}

init().catch((err) => {
  console.error("Initialization error:", err);
  const statusEl = document.getElementById("submit-status");
  if (statusEl) {
    statusEl.innerHTML =
      '<span class="status-note error-note">Could not connect to FastAPI server on port 8000. Start backend with `uvicorn app:app --port 8000`.</span>';
  }
});
