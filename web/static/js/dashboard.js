/**
 * CardioVision 3D - Clinical Dashboard Controller
 * Bridges Three.js 3D Heart Visualizer, FastAPI Backend, SHAP Explanations & Charts
 */

class CardioDashboard {
  constructor() {
    this.visualizer = null;
    this.presets = [];
    this.currentData = {};
    this.selectedTarget = "Cath";
    this.shapChart = null;
    this.rocChart = null;
    this.debounceTimer = null;

    this.init();
  }

  async init() {
    // 1. Initialize 3D Visualizer
    this.visualizer = new Heart3DVisualizer("heart3d-canvas-container");
    window.CardioApp = this;

    // 2. Setup UI Listeners
    this.setupNavigation();
    this.setupCameraControls();
    this.setupInputStudioListeners();

    // 3. Load Presets & Initial Patient
    await this.loadPresets();
    await this.loadMetrics();

    // 4. Default run with Preset 0 (Triple Vessel CAD)
    if (this.presets.length > 0) {
      this.selectPreset(0);
    }
  }

  setupNavigation() {
    // Top Tabs
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

        btn.classList.add("active");
        const tabId = btn.getAttribute("data-tab");
        const targetContent = document.getElementById(tabId);
        if (targetContent) targetContent.classList.add("active");

        if (tabId === "tab-metrics" && !this.rocChart) {
          this.renderRocChart();
        }
      });
    });

    // Vessel card selection
    ["LAD", "LCX", "RCA"].forEach((key) => {
      const card = document.getElementById(`card-vessel-${key.toLowerCase()}`);
      if (card) {
        card.addEventListener("click", () => {
          this.selectVesselCard(key);
          this.visualizer.focusVessel(key);
        });
      }
    });
  }

  selectVesselCard(key) {
    document.querySelectorAll(".vessel-card").forEach((c) => c.classList.remove("active-vessel"));
    const card = document.getElementById(`card-vessel-${key.toLowerCase()}`);
    if (card) card.classList.add("active-vessel");
    this.selectedTarget = key;
    this.renderShapChart();
  }

  setupCameraControls() {
    // Focus buttons
    const btnLAD = document.getElementById("btn-focus-lad");
    const btnLCX = document.getElementById("btn-focus-lcx");
    const btnRCA = document.getElementById("btn-focus-rca");
    const btnReset = document.getElementById("btn-focus-reset");
    const btnAnt = document.getElementById("btn-view-ant");
    const btnPost = document.getElementById("btn-view-post");

    if (btnLAD) btnLAD.addEventListener("click", () => { this.visualizer.focusVessel("LAD"); this.selectVesselCard("LAD"); });
    if (btnLCX) btnLCX.addEventListener("click", () => { this.visualizer.focusVessel("LCX"); this.selectVesselCard("LCX"); });
    if (btnRCA) btnRCA.addEventListener("click", () => { this.visualizer.focusVessel("RCA"); this.selectVesselCard("RCA"); });
    if (btnReset) btnReset.addEventListener("click", () => { this.visualizer.resetView(); this.selectedTarget = "Cath"; this.renderShapChart(); });
    if (btnAnt) btnAnt.addEventListener("click", () => this.visualizer.focusAnterior());
    if (btnPost) btnPost.addEventListener("click", () => this.visualizer.focusPosterior());

    // Toggles
    const toggleBeat = document.getElementById("toggle-heartbeat");
    const toggleTorso = document.getElementById("toggle-torso");
    const toggleFlow = document.getElementById("toggle-flow");

    if (toggleBeat) {
      toggleBeat.addEventListener("click", () => {
        const active = this.visualizer.toggleHeartbeat();
        toggleBeat.classList.toggle("active", active);
      });
    }
    if (toggleTorso) {
      toggleTorso.addEventListener("click", () => {
        const active = this.visualizer.toggleTorso();
        toggleTorso.classList.toggle("active", active);
      });
    }
    if (toggleFlow) {
      toggleFlow.addEventListener("click", () => {
        const active = this.visualizer.toggleBloodFlow();
        toggleFlow.classList.toggle("active", active);
      });
    }
  }

  async loadPresets() {
    try {
      const res = await fetch("/api/presets");
      const json = await res.json();
      this.presets = json.presets || [];

      const select = document.getElementById("preset-select");
      if (!select) return;
      select.innerHTML = "";

      this.presets.forEach((p, idx) => {
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = p.name;
        select.appendChild(opt);
      });

      select.addEventListener("change", (e) => {
        this.selectPreset(parseInt(e.target.value));
      });
    } catch (e) {
      console.error("Failed to load presets:", e);
    }
  }

  selectPreset(index) {
    if (!this.presets[index]) return;
    const preset = this.presets[index];
    this.currentData = { ...preset.data };

    // Update preset description banner
    const descEl = document.getElementById("preset-description");
    if (descEl) descEl.textContent = preset.description;

    // Sync input sliders & form controls
    this.populateInputsFromData(this.currentData);

    // Run prediction
    this.triggerPrediction();
  }

  populateInputsFromData(data) {
    // Numeric inputs & sliders
    const numericFields = [
      "Age", "Weight", "Length", "BMI", "BP", "PR", "FBS", "CR",
      "TG", "LDL", "HDL", "BUN", "ESR", "HB", "EF-TTE", "Region RWMA"
    ];

    numericFields.forEach((field) => {
      const input = document.getElementById(`input-${field.toLowerCase().replace(/[^a-z0-9]/g, "-")}`);
      const badge = document.getElementById(`val-${field.toLowerCase().replace(/[^a-z0-9]/g, "-")}`);
      if (input && data[field] !== undefined) {
        input.value = Math.round(Number(data[field]));
        if (badge) badge.textContent = input.value;
      }
    });

    // Checkboxes / Binaries
    const checkboxMap = {
      "DM": "cb-dm",
      "HTN": "cb-htn",
      "Current Smoker": "cb-smoker",
      "FH": "cb-fh",
      "Typical Chest Pain": "cb-typical-cp",
      "Dyspnea": "cb-dyspnea",
      "St Elevation": "cb-st-elevation",
      "St Depression": "cb-st-depression",
      "Tinversion": "cb-t-inversion",
      "Q Wave": "cb-q-wave",
      "LVH": "cb-lvh"
    };

    Object.entries(checkboxMap).forEach(([field, id]) => {
      const cb = document.getElementById(id);
      if (cb && data[field] !== undefined) {
        const val = data[field];
        cb.checked = val === 1 || val === "1" || String(val).toUpperCase() === "Y";
      }
    });
  }

  setupInputStudioListeners() {
    // Listen to range sliders
    const sliders = document.querySelectorAll(".input-studio-grid input[type='range']");
    sliders.forEach((slider) => {
      slider.addEventListener("input", (e) => {
        const badge = e.target.parentElement.querySelector(".slider-val-badge");
        if (badge) badge.textContent = e.target.value;
        this.collectDataFromInputs();
        this.debouncePrediction();
      });
    });

    // Listen to checkboxes
    const cbs = document.querySelectorAll(".input-studio-grid input[type='checkbox']");
    cbs.forEach((cb) => {
      cb.addEventListener("change", () => {
        this.collectDataFromInputs();
        this.debouncePrediction();
      });
    });

    // Action button
    const btnCalc = document.getElementById("btn-recalculate");
    if (btnCalc) {
      btnCalc.addEventListener("click", () => {
        this.collectDataFromInputs();
        this.triggerPrediction();
      });
    }
  }

  collectDataFromInputs() {
    const data = { ...this.currentData };

    // Numerical
    const numMappings = [
      { id: "input-age", field: "Age" },
      { id: "input-weight", field: "Weight" },
      { id: "input-length", field: "Length" },
      { id: "input-bmi", field: "BMI" },
      { id: "input-bp", field: "BP" },
      { id: "input-pr", field: "PR" },
      { id: "input-fbs", field: "FBS" },
      { id: "input-cr", field: "CR" },
      { id: "input-tg", field: "TG" },
      { id: "input-ldl", field: "LDL" },
      { id: "input-hdl", field: "HDL" },
      { id: "input-bun", field: "BUN" },
      { id: "input-esr", field: "ESR" },
      { id: "input-hb", field: "HB" },
      { id: "input-ef-tte", field: "EF-TTE" },
      { id: "input-region-rwma", field: "Region RWMA" }
    ];

    numMappings.forEach((m) => {
      const el = document.getElementById(m.id);
      if (el) data[m.field] = parseFloat(el.value);
    });

    // Binary Checkboxes
    const cbMappings = [
      { id: "cb-dm", field: "DM" },
      { id: "cb-htn", field: "HTN" },
      { id: "cb-smoker", field: "Current Smoker" },
      { id: "cb-fh", field: "FH" },
      { id: "cb-typical-cp", field: "Typical Chest Pain" },
      { id: "cb-dyspnea", field: "Dyspnea" },
      { id: "cb-st-elevation", field: "St Elevation" },
      { id: "cb-st-depression", field: "St Depression" },
      { id: "cb-t-inversion", field: "Tinversion" },
      { id: "cb-q-wave", field: "Q Wave" },
      { id: "cb-lvh", field: "LVH" }
    ];

    cbMappings.forEach((m) => {
      const el = document.getElementById(m.id);
      if (el) data[m.field] = el.checked ? 1 : 0;
    });

    this.currentData = data;
  }

  debouncePrediction() {
    clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.triggerPrediction();
    }, 180);
  }

  async triggerPrediction() {
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: this.currentData })
      });

      if (!response.ok) throw new Error("API call failed");
      const result = await response.json();
      this.lastResult = result;

      // Update 3D Canvas
      this.visualizer.updateVesselColors(result.vessels);

      // Update UI Widgets
      this.renderOverallCad(result.overall_cad);
      this.renderVesselCards(result.vessels);
      this.renderShapChart();
      this.renderCategoryBreakdown();
    } catch (e) {
      console.error("Prediction failed:", e);
    }
  }

  renderOverallCad(cad) {
    const probPct = Math.round(cad.probability * 100);
    const gaugeText = document.getElementById("cad-gauge-percentage");
    const gaugeCircle = document.getElementById("cad-gauge-svg-circle");
    const badgeRisk = document.getElementById("cad-risk-badge");
    const diagTitle = document.getElementById("cad-diagnosis-title");
    const summaryText = document.getElementById("cad-summary-text");

    if (gaugeText) gaugeText.textContent = `${probPct}%`;
    if (diagTitle) diagTitle.textContent = cad.severity_classification;
    if (summaryText) summaryText.textContent = cad.summary;

    if (gaugeCircle) {
      const circumference = 251.2; // 2 * PI * 40
      const offset = circumference - (probPct / 100) * circumference;
      gaugeCircle.style.strokeDashoffset = offset;
      const color = probPct >= 65 ? "#ff1744" : probPct >= 35 ? "#ffb300" : "#00e676";
      gaugeCircle.style.stroke = color;
    }

    if (badgeRisk) {
      badgeRisk.textContent = `${cad.risk_tier} Risk`;
      badgeRisk.className = `badge-risk badge-${cad.risk_tier.toLowerCase()}`;
    }
  }

  renderVesselCards(vessels) {
    ["LAD", "LCX", "RCA"].forEach((key) => {
      const v = vessels[key];
      if (!v) return;

      const probPct = Math.round(v.probability * 100);
      const probEl = document.getElementById(`vessel-${key.toLowerCase()}-prob`);
      const fillEl = document.getElementById(`vessel-${key.toLowerCase()}-fill`);
      const badgeEl = document.getElementById(`vessel-${key.toLowerCase()}-badge`);
      const findingEl = document.getElementById(`vessel-${key.toLowerCase()}-finding`);

      if (probEl) probEl.textContent = `${probPct}%`;
      if (fillEl) {
        fillEl.style.width = `${probPct}%`;
        fillEl.style.backgroundColor = v.color_code;
      }
      if (badgeEl) {
        badgeEl.textContent = v.risk_tier;
        badgeEl.className = `badge-risk badge-${v.risk_tier.toLowerCase()}`;
      }
      if (findingEl) findingEl.textContent = v.clinical_finding;
    });
  }

  renderShapChart() {
    if (!this.lastResult || !this.lastResult.explanations) return;
    const targetKey = this.selectedTarget || "Cath";
    const exp = this.lastResult.explanations[targetKey];
    if (!exp) return;

    const canvas = document.getElementById("shap-chart-canvas");
    if (!canvas) return;

    const topFeatures = exp.top_features || [];
    const labels = topFeatures.map((f) => f.feature);
    const dataValues = topFeatures.map((f) => f.shap_value);
    const bgColors = dataValues.map((v) => (v > 0 ? "rgba(255, 23, 68, 0.75)" : "rgba(0, 230, 118, 0.75)"));

    if (this.shapChart) {
      this.shapChart.destroy();
    }

    const titleTarget = targetKey === "Cath" ? "Overall CAD" : `${targetKey} Stenosis`;
    document.getElementById("shap-chart-title").textContent = `SHAP Feature Attribution: ${titleTarget}`;

    this.shapChart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "SHAP Impact on Risk",
            data: dataValues,
            backgroundColor: bgColors,
            borderRadius: 4
          }
        ]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const item = topFeatures[ctx.dataIndex];
                const direction = item.shap_value > 0 ? "Increases Risk" : "Reduces Risk";
                return `${direction} by ${Math.abs(item.shap_value).toFixed(4)} (Value: ${item.patient_value})`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.08)" },
            ticks: { color: "#94a3b8", font: { size: 10 } }
          },
          y: {
            grid: { display: false },
            ticks: { color: "#f1f5f9", font: { size: 11, weight: "bold" } }
          }
        }
      }
    });
  }

  renderCategoryBreakdown() {
    if (!this.lastResult || !this.lastResult.explanations) return;
    const targetKey = this.selectedTarget || "Cath";
    const exp = this.lastResult.explanations[targetKey];
    if (!exp || !exp.category_contributions) return;

    const container = document.getElementById("category-breakdown-container");
    if (!container) return;
    container.innerHTML = "";

    Object.entries(exp.category_contributions).forEach(([cat, val]) => {
      const row = document.createElement("div");
      row.className = "category-row";
      const color = val > 0 ? "var(--risk-high)" : val < 0 ? "var(--risk-low)" : "var(--text-muted)";
      const sign = val > 0 ? "+" : "";

      row.innerHTML = `
        <span class="cat-name">${cat}</span>
        <span class="cat-impact" style="color: ${color}">${sign}${val.toFixed(3)}</span>
      `;
      container.appendChild(row);
    });
  }

  async loadMetrics() {
    try {
      const res = await fetch("/api/metrics");
      const json = await res.json();
      this.metricsData = json;
      this.renderMetricsTable(json.metrics);
    } catch (e) {
      console.error("Failed to load metrics:", e);
    }
  }

  renderMetricsTable(metrics) {
    const tbody = document.getElementById("metrics-table-body");
    if (!tbody || !metrics) return;
    tbody.innerHTML = "";

    const names = {
      Cath: "Overall CAD (Cath)",
      LAD: "LAD Stenosis",
      LCX: "LCX Stenosis",
      RCA: "RCA Stenosis"
    };

    Object.entries(metrics).forEach(([target, m]) => {
      const te = m.test_evaluation;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><b>${names[target] || target}</b></td>
        <td><span class="badge-track">${m.best_model}</span></td>
        <td class="val-bold" style="color: var(--accent-cyan)">${(te.roc_auc * 100).toFixed(1)}%</td>
        <td class="val-bold">${(te.f1_score * 100).toFixed(1)}%</td>
        <td>${(te.accuracy * 100).toFixed(1)}%</td>
        <td>${(te.recall * 100).toFixed(1)}%</td>
        <td>${(te.precision * 100).toFixed(1)}%</td>
      `;
      tbody.appendChild(tr);
    });
  }

  renderRocChart() {
    if (!this.metricsData || !this.metricsData.roc_curves) return;
    const canvas = document.getElementById("roc-chart-canvas");
    if (!canvas) return;

    const rocCurves = this.metricsData.roc_curves;
    const colors = {
      Cath: "#ff1744",
      LAD: "#ffb300",
      LCX: "#00e5ff",
      RCA: "#00e676"
    };

    const datasets = Object.entries(rocCurves).map(([key, data]) => {
      const xyPoints = data.fpr.map((fpr, i) => ({ x: fpr, y: data.tpr[i] }));
      return {
        label: `${key} (AUC = ${data.auc})`,
        data: xyPoints,
        borderColor: colors[key] || "#fff",
        backgroundColor: "transparent",
        borderWidth: 2.5,
        tension: 0.1,
        pointRadius: 0
      };
    });

    // Chance line
    datasets.push({
      label: "Chance (AUC = 0.50)",
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: "#64748b",
      borderWidth: 1.5,
      borderDash: [5, 5],
      pointRadius: 0
    });

    this.rocChart = new Chart(canvas, {
      type: "line",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: 1,
            title: { display: true, text: "False Positive Rate (1 - Specificity)", color: "#94a3b8" },
            grid: { color: "rgba(255, 255, 255, 0.08)" }
          },
          y: {
            min: 0,
            max: 1,
            title: { display: true, text: "True Positive Rate (Sensitivity)", color: "#94a3b8" },
            grid: { color: "rgba(255, 255, 255, 0.08)" }
          }
        },
        plugins: {
          legend: { position: "bottom", labels: { color: "#f1f5f9", boxWidth: 12 } }
        }
      }
    });
  }
}

// Start app on DOMContentLoaded
window.addEventListener("DOMContentLoaded", () => {
  new CardioDashboard();
});
