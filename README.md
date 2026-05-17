# Critical Oxygen Saturation Estimation

A data science project for predicting **oxygen saturation** (`oximetry`) from multimodal patient sensor data. The dataset contains physiological readings such as SpO₂, heart rate, respiratory rate, hemoglobin levels, and more. The task is a binary classification problem: predicting whether a patient's oxygen saturation is in a critical or normal range.

**Authors:** Mykhailo Chepara, Bei Veronika

## Dataset

The dataset consists of three related files — patient records, observation readings, and station metadata — merged across a shared time and station identifier. Sensor variables include SpO₂, HR, RR, EtCO₂, FiO₂, PRV, Skin Temperature, PVI, Hb level, SV, CO, Blood Flow Index, O₂ extraction ratio, SNR, and others.

## Project Structure

The project is a single Jupyter notebook (`project.ipynb`) organized into three phases:

**Phase 1 — Exploratory Data Analysis**
- Structural analysis of all three source files
- Per-attribute distribution and descriptive statistics for 10+ variables
- Pairwise correlation analysis and dependency analysis against the target variable
- Outlier detection using IQR and 3-SD methods; outliers handled by removal or capping/median substitution depending on their proportion
- Inconsistency detection: mixed date formats, corrupted file structure (`patient.csv`), out-of-range sensor values
- Statistical hypothesis testing with power analysis

**Phase 2 — Preprocessing & Feature Selection**
- Train/test split (75%/25%)
- Date format unification, missing value imputation, categorical encoding
- Feature transformations: Box-Cox and Yeo-Johnson power transforms, standard scaling
- Feature selection using 5 methods: Chi-squared, F-statistic, RFE with SVC, L1-penalized LinearSVC, and Lasso regression — top features identified as RR, Skin Temperature, SpO₂, and Hb level
- Reproducible `sklearn.Pipeline` wrapping all preprocessing steps

**Phase 3 — Machine Learning**
- Custom ID3 dichotomizer with build-time pruning (minimum node size, subset balance constraint)
- Random Forest (scikit-learn), Logistic Regression, Quadratic Discriminant Analysis, MLP Classifier
- Hyperparameter tuning via `GridSearchCV` with 5–7-fold cross-validation for all models
- Soft/hard voting Ensemble Model combining Random Forest, MLP, and Logistic Regression
- Model calibration check (Brier score); Logistic Regression calibrated with `CalibratedClassifierCV`
- Final model: **Ensemble (hard voting)** — training F1 ~92%, validation F1 ~90%, difference ≤3% indicating no significant overfitting

## Requirements

```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn pingouin
```

## Usage

Open `project.ipynb` in Jupyter and run all cells. Place the source CSV files under a `Datasets/` directory before running.
