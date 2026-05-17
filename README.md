# Critical Oxygen Saturation Estimation

A binary classification pipeline predicting whether a patient's **oxygen saturation** (`oximetry`) is critical or normal, built from multimodal sensor data. Implemented in Python using pandas, scikit-learn, scipy, and statsmodels across a structured three-phase data science workflow.

**Authors:** Mykhailo Chepara, Bei Veronika

## Dataset

Three source files merged on station ID and timestamp: patient records (demographics, registration time, station assignment), observation readings (sensor values per visit), and station metadata (QoS ratings). The target variable `oximetry` is binary. Input features include SpO₂, HR, RR, EtCO₂, FiO₂, PRV, BP, Skin Temperature, PVI, Hb level, SV, CO, Blood Flow Index, O₂ extraction ratio, SNR, and others. `patient.csv` had a corrupted structure and was reparsed with a custom line-by-line reformatter before loading.

## Phase 1 — EDA & Data Cleaning

- Per-attribute descriptive statistics and distribution analysis for 10+ variables; out-of-range values flagged against known physiological bounds (e.g. SpO₂ 95–100%, HR 60–100 bpm)
- Outlier detection via IQR (±1.5×IQR) and 3-SD methods; outliers representing >1.5% of a column replaced by median or capped at the 5th/95th percentile, smaller fractions dropped outright
- Mixed date format unification across 5 format patterns using a custom parser
- QoS encoded ordinally: `maintenance=0`, `acceptable=1`, `good=2`, `excellent=3`
- Pairwise correlation analysis and dependency analysis against `oximetry`
- Two statistical hypotheses formulated and tested with appropriate tests (t-test / Mann-Whitney); statistical power verified

## Phase 2 — Preprocessing & Feature Selection

- 75/25 train/test split; all fitting done on training set only
- Power transforms applied per column group: Box-Cox (`PVI`, `Hb level`, `Skin Temperature`, `SpO₂`, `RR`, `FiO₂`, `Blood Flow Index`, `Respiratory effort`, `O₂ extraction ratio`), Yeo-Johnson (`longitude`, `latitude`), standard scaling (`BP`)
- Feature selection with 5 methods: Chi-squared statistic, F-statistic (`SelectKBest`), RFE with linear SVC, L1-penalized `LinearSVC` with `SelectFromModel`, and Lasso regression — consensus top features: **RR, Skin Temperature, SpO₂, Hb level**
- Both a manual pipeline and an `sklearn.Pipeline` implemented; the pipeline wraps all preprocessing steps for reproducible application to unseen data

## Phase 3 — Modelling & Evaluation

**Models trained:**
- Custom **ID3 dichotomizer** — entropy-based recursive splits with build-time pruning (minimum node size threshold + child subset balance constraint to prevent degenerate splits)
- **Random Forest** (`n_estimators=50`, `criterion=gini`, `min_samples_split=10`, `min_samples_leaf=2`)
- **Logistic Regression** (L1 penalty, `C=0.5`, `solver=liblinear`, balanced class weights)
- **Quadratic Discriminant Analysis** (`reg_param=0.01`)
- **MLP Classifier** (layers: 40→30→20, tanh activation, `alpha=0.0001`, `batch_size=40`, Adam)
- **Ensemble** — hard-voting `VotingClassifier` over Random Forest, MLP, and Logistic Regression

**Tuning & validation:**
- `GridSearchCV` with 5–7-fold cross-validation per model
- Logistic Regression calibrated post-tuning using `CalibratedClassifierCV` (sigmoid, cv=5); Brier score checked for RF and MLP (calibration adequate without correction)
- 5-fold cross-validation on the combined train+validation set for final stability check

**Best model: Ensemble (hard voting)** — training F1 ~92%, validation F1 ~90%, Δ≤3% across folds indicating no significant overfitting.

## Requirements

```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn pingouin
```

## Usage

Place source CSVs under `Datasets/` and run `project.ipynb` top to bottom. The `preclean_data()` function can be used to prepare any new raw data before passing it into the fitted pipeline.
