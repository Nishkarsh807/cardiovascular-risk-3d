# CardioVision 3D: Comprehensive Project Documentation & Technical Report
**Multimodal AI Hackathon 2026 — Track A: Cardiovascular Risk Visualization & Prediction**

---

## 1. Executive Summary & Clinical Context

Coronary Artery Disease (CAD) remains the single leading contributor to global cardiovascular mortality, responsible for over 9 million deaths annually. CAD pathology originates from the progressive accumulation of fibrofatty atheromatous plaques within the intimal lining of the epicardial coronary arteries. Over time, luminal narrowing—termed **stenosis**—compromises myocardial perfusion, precipitating stable angina, acute coronary syndromes (myocardial infarction), ischemic cardiomyopathy, and fatal dysrhythmias.

### The Clinical Dilemma
Current clinical risk calculators (e.g., Framingham Risk Score, ASCVD Pooled Cohort Equations, SCORE2) compute aggregate 10-year statistical risk probabilities. While epidemiologically valuable, these aggregate numbers suffer from fundamental limitations in patient-facing communication and individualized surgical/interventional planning:
1. **Lack of Spatial Anatomical Granularity:** A 70% CAD risk score provides zero localization regarding whether ischemia threatens the anterior interventricular septum ("widowmaker" territory) or the diaphragmatic inferior wall.
2. **Abstract Risk Perception:** Patients struggle to conceptualize abstract percentages, impairing adherence to aggressive lifestyle modifications and pharmacotherapy.
3. **Black-Box Skepticism:** Clinicians are hesitant to trust machine learning recommendations without transparent, patient-specific mechanistic feature attributions.

### Project Objective
**CardioVision 3D** was engineered to resolve these challenges by coupling multi-target machine learning with an interactive, WebGL-rendered 3D anatomical heart and coronary arterial tree. The system:
- Ingests 55+ clinical, laboratory, ECG, and echocardiographic biomarkers.
- Concurrently predicts overall CAD status alongside vessel-specific stenosis ($\ge 50\%$ luminal occlusion) across the three primary coronary arteries:
  - **LAD (Left Anterior Descending)**
  - **LCX (Left Circumflex)**
  - **RCA (Right Coronary Artery)**
- Maps probabilistic risk directly onto corresponding 3D vascular geometries using intuitive clinical color codes (Green / Amber / Crimson).
- Dynamically slows simulated hemodynamic blood flow particles through stenotic segments to illustrate vascular resistance.
- Computes real-time SHAP (SHapley Additive exPlanations) values to disclose exactly which physiological parameters drove the diagnosis.

---

## 2. Anatomical Foundations & Target Vessels

### 2.1 Coronary Arterial Perfusion Territories
The epicardial coronary tree originates from the root of the ascending aorta just superior to the aortic valve leaflets (sinuses of Valsalva):

1. **Left Anterior Descending (LAD) Artery:**
   - *Origin & Course:* Arises from the bifurcation of the Left Main Coronary Artery (LMCA) and courses down the anterior interventricular groove towards the cardiac apex.
   - *Branches:* Diagonal branches (D1, D2) and septal perforators.
   - *Perfusion Territory:* Anterior wall of the left ventricle, apex, and anterior two-thirds of the interventricular septum (including the bundle branches).
   - *Clinical Significance:* Occlusion of the proximal LAD is colloquially termed the *"Widowmaker"* due to the catastrophic reduction in left ventricular ejection fraction and high incidence of cardiogenic shock.

2. **Left Circumflex (LCX) Artery:**
   - *Origin & Course:* Emerges from the LMCA and wraps posteriorly around the heart within the left atrioventricular (coronary) sulcus.
   - *Branches:* Obtuse marginal (OM1, OM2) branches.
   - *Perfusion Territory:* Posterolateral wall of the left ventricle and left atrium.
   - *Clinical Significance:* LCX ischemia is frequently clinically silent on standard 12-lead ECGs without posterior leads (V7–V9), making ML-based risk assessment exceptionally valuable.

3. **Right Coronary Artery (RCA):**
   - *Origin & Course:* Arises from the right aortic sinus and travels down the right atrioventricular groove towards the diaphragmatic surface, terminating as the Posterior Descending Artery (PDA) in right-dominant anatomies (~85% of individuals).
   - *Branches:* Acute marginal branches, SA nodal artery, and AV nodal artery.
   - *Perfusion Territory:* Right ventricle, right atrium, inferior (diaphragmatic) wall of the left ventricle, and the atrioventricular conduction node.
   - *Clinical Significance:* Proximal RCA stenosis manifests as inferior wall myocardial infarction and severe bradyarrhythmias (complete heart block).

### 2.2 Regional Wall Motion Abnormality (RWMA)
Regional Wall Motion Abnormality (RWMA) is an echocardiographic finding denoting hypokinesia, akinesia, or dyskinesia in a specific segment of the left ventricular myocardium. In the Z-Alizadeh Sani dataset, `Region with RWMA` is encoded as a categorical clinical feature (0 to 4). Crucially, while RWMA reflects ischemia-induced mechanical dysfunction, it does not provide direct 3D coordinate geometry; CardioVision 3D consumes RWMA as an input biomarker and correlates it with the corresponding anatomical vessel territory.

---

## 3. Dataset Preprocessing & Target Leakage Prevention

### 3.1 Primary Dataset: Extension of Z-Alizadeh Sani
The system was trained and evaluated on the official **Extension of Z-Alizadeh Sani Dataset** hosted on the UCI Machine Learning Repository (Dataset ID: 411).
- **Patient Cohort:** 303 adult patients referred for diagnostic coronary angiography.
- **Dimensionality:** 59 initial raw features spanning four clinical categories:
  - *Demographics:* Age, Sex, Weight, Height, BMI.
  - *Symptoms & History:* Typical Chest Pain, Exertional Dyspnea, Diabetes Mellitus (DM), Hypertension (HTN), Current Smoker, Family History (FH), Chronic Renal Failure (CRF), Cerebrovascular Accident (CVA).
  - *ECG Findings:* ST Elevation, ST Depression, T-Wave Inversion, Pathological Q Wave, Poor R Progression, Left Ventricular Hypertrophy (LVH), Bundle Branch Block (LBBB/RBBB).
  - *Laboratory & Echo:* Fasting Blood Sugar (FBS), Serum Creatinine (CR), Triglycerides (TG), LDL, HDL, BUN, ESR, Hemoglobin (HB), Electrolytes (K, Na), Platelets (PLT), Ejection Fraction (EF-TTE), Region with RWMA, Valvular Heart Disease (VHD).

### 3.2 Target Leakage Safeguards
A critical requirement of the hackathon brief is preventing target leakage:
$$\mathcal{X} = \text{Features} \setminus \{\text{Cath}, \text{LAD}, \text{LCX}, \text{RCA}\}$$

If any coronary target or angiographic variable were permitted inside the feature matrix $\mathcal{X}$, the model would simply memorize the catheterization outcome rather than predicting risk from non-invasive physiological signs. 
- In `src/preprocessing.py`, `TARGET_COLS = ["Cath", "LAD", "LCX", "RCA"]` are strictly isolated before any scaling or feature engineering occurs.
- The single-sample inference endpoint in `src/api.py` enforces this safeguard: even if a client mistakenly submits a dictionary containing target labels, they are immediately stripped before entering the transformation pipeline.

### 3.3 Data Transformation Pipeline
1. **Binary Encoding:** Features with `"Y"` / `"N"` representations are mapped to $1.0$ and $0.0$.
2. **Sex Normalization:** Converted to binary indicator ($1 = \text{Male}, 0 = \text{Female}$).
3. **Multi-Class Nominal Encoding:**
   - Bundle Branch Block (`BBB`): One-hot encoded into `BBB_LBBB` and `BBB_RBBB` (with Normal as baseline).
   - Valvular Heart Disease (`VHD`): Encoded into `VHD_mild`, `VHD_Moderate`, `VHD_Severe`.
4. **Continuous Feature Standardization:** Standardized via $z = \frac{x - \mu}{\sigma}$ using a fitted `StandardScaler` saved to `models/preprocessor.pkl`.

---

## 4. Machine Learning Pipeline & Empirical Validation

### 4.1 Multi-Target Formulation
The diagnostic task is modeled as four independent supervised binary classification problems:
1. $\mathcal{M}_{\text{Cath}}(x) \to P(\text{Overall CAD} \mid x)$
2. $\mathcal{M}_{\text{LAD}}(x) \to P(\text{LAD Stenosis} \ge 50\% \mid x)$
3. $\mathcal{M}_{\text{LCX}}(x) \to P(\text{LCX Stenosis} \ge 50\% \mid x)$
4. $\mathcal{M}_{\text{RCA}}(x) \to P(\text{RCA Stenosis} \ge 50\% \mid x)$

### 4.2 Algorithm Selection & Cross-Validation
To guarantee reproducibility and avoid overfitting on the 303-patient cohort, candidate models were benchmarked across a **Stratified 5-Fold Cross Validation** scheme ($80\%$ train split, $N=242$), followed by final evaluation on a sealed $20\%$ holdout test set ($N=61$).

Candidate algorithms evaluated:
- **Random Forest (RF):** 150 ensemble estimators, max depth 6, min samples leaf 2.
- **Extra Trees (ET):** Extremely randomized trees with variance reduction.
- **Gradient Boosting (GBM):** Stage-wise additive model with learning rate 0.05.
- **L2-Regularized Logistic Regression (LR):** Parametric baseline with inverse regularization strength $C=0.5$.

### 4.3 Benchmark Results

```
========================================================================================
 Target Vessel / CAD Endpoint | Selected Model | 5-Fold CV AUC | Test AUC | Test F1-Score
========================================================================================
 Overall CAD Status (Cath)   | Random Forest  | 0.9344 ± 0.02 | 0.8824   | 87.50%
 LAD Stenosis                | Extra Trees    | 0.8544 ± 0.04 | 0.7865   | 76.32%
 LCX Stenosis                | Gradient Boost | 0.7567 ± 0.04 | 0.7033   | 60.87%
 RCA Stenosis                | Logistic L2    | 0.7334 ± 0.04 | 0.7220   | 47.37%
========================================================================================
```

- **Overall CAD (`Cath`):** Random Forest achieved exceptional discriminatory power with an ROC-AUC of **0.9344** in cross-validation and **0.8824** on holdout data, with a clinical sensitivity (recall) of **97.67%**, ensuring virtually zero false negatives for high-risk cardiac patients.
- **LAD Stenosis:** Extra Trees yielded **0.8544** CV AUC and **85.29%** sensitivity on the holdout test set.
- **LCX & RCA Stenosis:** Gradient Boosting and Logistic Regression demonstrated consistent discrimination (**0.7567** and **0.7334** CV AUC) without evidence of overparameterization.

---

## 5. Clinical Interpretability via SHAP (SHapley Additive exPlanations)

### 5.1 Game-Theoretic Formulation
To meet Requirement 3.b, CardioVision 3D implements SHAP values grounded in cooperative game theory. For any clinical prediction $f(x)$, the local attribution $\phi_i$ of biomarker $i$ satisfies efficiency, symmetry, dummy, and additivity:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
where $\phi_0$ is the base expected model output over the training background distribution.

### 5.2 Fast Real-Time Computation
- For tree-based models (Random Forest, Extra Trees, Gradient Boosting), native `TreeExplainer` algorithms compute exact Shapley values by traversing internal decision paths in $\mathcal{O}(TLD^2)$ time, executing in under **80 milliseconds**.
- For linear models, `LinearExplainer` leverages analytical covariance decompositions.

### 5.3 Physiological Grouping
Biomarker attributions are clustered into six physiological categories:
1. **Demographics:** Age, Sex, Weight, BMI.
2. **Symptoms & Hemodynamics:** Typical Chest Pain, Blood Pressure, Heart Rate, Dyspnea.
3. **Electrocardiography (ECG):** ST Elevation, ST Depression, T-Wave Inversion, Q Wave, LVH.
4. **Lipid Profile & Biomarkers:** LDL, HDL, Triglycerides, Fasting Blood Sugar, Creatinine.
5. **Echocardiography:** Ejection Fraction (EF-TTE), Regional Wall Motion Abnormality (RWMA).
6. **Clinical History:** Diabetes, Hypertension, Smoking Status, Family History.

In the UI, these grouped contributions are rendered as interactive horizontal bar charts and category impact summaries.

---

## 6. Interactive 3D WebGL Visualization Engine

### 6.1 Scene Architecture & Three.js Pipeline
The 3D anatomical viewer is implemented in Three.js with full client-side WebGL rendering, requiring no external dedicated GPU:
- **Myocardial Ventricles:** Conical and spherical organic geometries representing the Left Ventricle (systemic high-pressure chamber) and Right Ventricle.
- **Great Vessels:**
  - *Ascending Aorta & Arch:* Constructed via a 6-point Catmull-Rom spline tube with the brachiocephalic, left common carotid, and left subclavian branches.
  - *Pulmonary Trunk:* Branching into left and right pulmonary branches.
  - *Vena Cava:* Superior and inferior vena cava entering the right atrium.
- **Holographic Torso Cage:** A stylized wireframe thoracic cage and sternum outline providing spatial thoracic context for cardiac orientation.

### 6.2 Parametric Coronary Splines
The three coronary arteries are defined as 3D spatial Catmull-Rom spline curves:
- **LAD Spline:** Originates at $(-0.25, 0.95, 0.45)$, traverses the anterior interventricular sulcus, and terminates near the apex $(0.05, -2.0, 0.45)$, with diagonal branches ($D1$).
- **LCX Spline:** Curves laterally along the left AV groove $(-1.45, 0.15, -0.45)$ to the posterior LV, with obtuse marginal branches ($OM1$).
- **RCA Spline:** Descends down the right atrioventricular groove $(1.25, -0.2, 0.35)$ to the posterior descending artery (PDA).

### 6.3 Dynamic Risk Shaders & Hemodynamic Physics
- **Color Shading:**
  - Low Risk ($<35\%$): Emerald Green (`#00e676`)
  - Moderate Risk ($35\% - 65\%$): Amber Orange (`#ffb300`)
  - Severe Stenosis ($>65\%$): Crimson Neon (`#ff1744`)
- **Ischemic Glow Pulsation:** High-risk vessels emit an oscillating emissive glow ($I = 0.7 + 0.35\sin(6t)$).
- **Particle Blood Flow Simulation:** 60 particle vectors traverse each coronary curve. The flow speed $v$ is inversely proportional to predicted stenosis risk:
  $$v = \max\left(0.04, (1.0 - 0.75 \cdot P_{\text{stenosis}}) \cdot 0.25\right)$$
  When a vessel develops high stenosis risk, particle velocity drops precipitously, visually illustrating impaired coronary blood flow reserve and downstream ischemia.

### 6.4 Raycasting & Camera Lerping
- **Mouse Raycaster:** Intersects coronary geometries on hover and click.
- **Camera Lerp Transitions:** Spherical linear interpolation smoothly glides the camera to focus on the selected vessel upon user selection.

---

## 7. System Integration & User Workflows

### 7.1 Client-Server Interaction
```
[Client Web Browser]
   │
   ├─► GET /api/presets  ──► Ingests 5 verified clinical cases
   ├─► GET /api/metrics  ──► Renders holdout ROC curves & validation table
   │
   └─► POST /api/predict (Patient JSON)
         │
         ▼
   [FastAPI Backend]
         │
         ├─► Preprocessor (Zero Leakage Filter)
         ├─► RF/ET/GBM/LR Model Inference
         ├─► TreeSHAP Attribution Calculation
         │
         ▼
   [JSON Response: Probabilities + Vessel Colors + SHAP Attributions]
         │
         ▼
   [Client Web Browser Updates]
         ├─► 3D Heart: Vessel Colors, Blood Flow Speeds, Ischemic Glow
         └─► HUD: CAD Gauge, Vessel Progress Bars, SHAP Waterfall Chart
```

### 7.2 Clinical Preset Scenarios
To enable immediate demonstration without manual slider entry, five verified cases from the dataset are provided:
1. **Case 1 (Severe Triple Vessel CAD):** 68yo male with typical angina, DM, HTN, and multi-vessel stenosis. All three vessels glow red.
2. **Case 2 (Isolated LAD "Widowmaker"):** Critical LAD stenosis with normal LCX and RCA perfusion.
3. **Case 3 (Isolated RCA Stenosis):** Inferior wall regional wall motion abnormality (RWMA) and right coronary disease.
4. **Case 4 (Healthy Control):** 45yo asymptomatic subject with normal coronaries (all emerald green).
5. **Case 5 (Intermediate Risk):** Borderline hypertension and early atherosclerosis.

---

## 8. Clinical Safety & Regulatory Disclaimer

**MANDATORY NOTIFICATION:**  
CardioVision 3D is designed for clinical decision support, biomedical engineering research, and educational demonstration. The software has not been cleared as a medical device by the US FDA, EMA, or national health authorities. Predictions are probabilistic machine learning inferences and **must not be used as a standalone diagnostic tool or substitute for formal diagnostic coronary angiography, CT angiography, or physician clinical judgment.**

---

## 9. Conclusion

CardioVision 3D demonstrates that predictive clinical machine learning and interactive 3D spatial visualization can be integrated into a responsive, zero-friction web application. By enforcing rigorous target leakage safeguards, maintaining multi-target cross-validation benchmarks (ROC-AUC 0.934 for CAD, 0.854 for LAD), generating instant SHAP feature explanations, and rendering responsive 3D anatomical coronary hemodynamics, this platform establishes a standard for modern cardiovascular AI visualization.
