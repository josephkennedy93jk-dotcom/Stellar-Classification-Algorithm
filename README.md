# Stellar Classification with Machine Learning

Classifying Sloan Digital Sky Survey (SDSS) objects as **galaxies, stars, or quasars** from photometric measurements — an end-to-end analytics project built to demonstrate an *honest* machine-learning workflow, not just a high accuracy number.

The centrepiece is a controlled **data-leakage experiment**: each model is trained with and without redshift, a feature that in a realistic image-only pipeline would not be available. The gap between those two scores is the story.

---

## Table of Contents

- [The Problem](#the-problem)
- [The Dataset](#the-dataset)
- [Methodology](#methodology)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Selection](#3-model-selection)
  - [4. The Leakage Experiment](#4-the-leakage-experiment)
  - [5. Cross-Validated Bake-Off](#5-cross-validated-bake-off)
- [Results](#results)
- [Interpretation](#interpretation)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## The Problem

Modern sky surveys photograph millions of objects every night — far more than any team of astronomers could classify by hand. The task is therefore operational, not just scientific:

> Can we automatically sort each observed object into **galaxy, star, or quasar** using only the measurements a telescope already captures?

This is a **multi-class classification** problem. The practical value is triage: an automated classifier lets scientists focus expensive follow-up observations on the objects that actually warrant them.

A subtle but important distinction runs through the whole project: classifying an object *before* spectroscopic follow-up (from images alone) is a genuinely harder and more useful task than classifying it *after*, because the most predictive feature — redshift — is only obtained through that expensive follow-up. Most of the methodology below exists to respect this distinction.

---

## The Dataset

The SDSS stellar classification dataset contains roughly **100,000 rows**, each describing one observed sky object. The columns fall into four groups:

| Group | Columns | Role |
|---|---|---|
| **Identifiers** | `obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `spec_obj_ID`, `plate`, `MJD`, `fiber_ID` | Bookkeeping — which telescope run captured the object. No predictive value. **Dropped.** |
| **Sky position** | `alpha` (right ascension), `delta` (declination) | Where the object sits on the celestial sphere. Should be irrelevant to type. |
| **Photometric bands** | `u`, `g`, `r`, `i`, `z` | Brightness through five filters, ultraviolet to infrared. **The core signal.** |
| **Spectroscopic** | `redshift`, `class` | `class` is the target label (GALAXY / STAR / QSO). `redshift` is highly predictive but only available post-spectroscopy. |

The five photometric bands are **magnitudes** — a logarithmic brightness scale where *lower numbers mean brighter*. Because they all measure the same object through different filters, they are strongly correlated with one another (a fact that directly motivates the feature engineering below).

---

## Methodology

### 1. Exploratory Data Analysis

Before modelling, the data was interrogated with four blunt questions:

1. **Is the data even okay?** — checked shape, missing values, duplicate rows, and sentinel outliers (SDSS uses values like `-9999` to flag failed measurements).
2. **What am I predicting?** — examined class balance. Galaxies dominate at roughly **60%**, which sets the baseline: any model must beat 60%, the score achievable by blindly guessing "galaxy" every time.
3. **Are any features cheating?** — plotted redshift by class and confirmed it almost perfectly separates the three types, flagging it as a leakage risk early.
4. **Are inputs redundant?** — a correlation heatmap showed the five bands moving together almost identically, signalling that feeding all five raw magnitudes into a model would be wasteful.

That final finding — redundancy among the bands — is what pointed directly toward feature engineering.

### 2. Feature Engineering

Absolute brightness is a poor discriminator because it confounds an object's intrinsic nature with its distance: a dim nearby star and a bright faraway galaxy can register similar raw magnitudes.

The fix — a convention astronomers have used for over a century — is the **colour index**: the difference in brightness between two adjacent bands. Colour captures the *shape* of an object's light spectrum, independent of distance or overall brightness. Four colour features were engineered:

```python
df['u_g'] = df['u'] - df['g']
df['g_r'] = df['g'] - df['r']
df['r_i'] = df['r'] - df['i']
df['i_z'] = df['i'] - df['z']
```

This is **borrowed domain knowledge, then empirically validated**. The choice of colour indices was not derived from first principles; it was sourced from the field's established practice, then confirmed by the modelling results (colours consistently outrank the raw bands they are built from — see [Results](#results)).

### 3. Model Selection

Two model families were used deliberately — not to crown a winner, but to diagnose the *shape* of the problem:

- **Logistic Regression** — a linear baseline. It draws straight-line decision boundaries, is fully interpretable via its coefficients, and fails visibly when relationships are nonlinear (which is informative).
- **Random Forest** — an ensemble of decision trees that vote. It captures curved boundaries and feature interactions, is insensitive to feature scaling, and reports feature importances.

A large gap between the two signals nonlinear structure in the data. A small gap signals the problem is close to linear. This comparison is a diagnostic, not a contest.

**Methodological safeguards applied throughout:**
- **Stratified train/test split** (80/20) so the minority class (quasars) is represented proportionally in both sets.
- **Scaler fit on training data only**, then applied to the test set — preventing test-set information from leaking into preprocessing.
- **Features scaled only for scale-sensitive models** (Logistic Regression, KNN); tree models use raw features since they split on thresholds.

### 4. The Leakage Experiment

Each model was trained **twice** — once *without* redshift, once *with* it — holding everything else constant (same split, same seed, same features otherwise).

Redshift is scientifically valid and enormously predictive: stars have redshift near zero, galaxies moderate values, quasars very large ones. But because it comes from spectroscopy performed *after* image capture, a classifier meant to triage raw images cannot legitimately use it. Including it therefore constitutes **proxy data leakage** — the model is handed a feature that, in real deployment, wouldn't exist.

The with/without comparison quantifies exactly how much of each model's performance depends on this unrealistic shortcut.

### 5. Cross-Validated Bake-Off

To check whether the choice of model mattered, seven classifiers were benchmarked with **5-fold cross-validation** (a more robust estimate than a single train/test split, since it averages performance across five different data partitions). Models tested: Logistic Regression, K-Nearest Neighbours, Naive Bayes, Decision Tree, Random Forest, Extra Trees, and Gradient Boosting.

---

## Results

All models were evaluated on a stratified 20% held-out test set.

### The four-way comparison (leakage experiment)

| Model | Without redshift (honest) | With redshift (leaky) |
|---|:---:|:---:|
| Logistic Regression | 88% | 95% |
| Random Forest | 88% | **97%** |

Two observations:

- **Redshift contributes 7–9 percentage points.** This is the leakage, quantified. For a realistic image-only pipeline, the honest accuracy ceiling with these features is roughly **88%**.
- **Random Forest only outperforms Logistic Regression when redshift is included.** Without it, both tie at 88% — meaning the photometric signal is largely linear. With redshift, the nonlinear interaction between redshift and the colour features gives the forest a small edge.

### Cross-validated leaderboard (with redshift)

| Rank | Model | Mean CV Accuracy |
|---|---|:---:|
| 1 | Random Forest | 97.9% |
| 2 | Gradient Boosting | 97.7% |
| 3 | Extra Trees | 97.4% |
| 4 | Decision Tree | 96.6% |
| 5 | Logistic Regression | 95.5% |
| 6 | K-Nearest Neighbours | 94.3% |
| 7 | Naive Bayes | 76.3% |

The three tree ensembles cluster tightly at the top, confirming the original Random Forest choice was near-optimal. **Naive Bayes collapses** because its core assumption — that features are independent given the class — is badly violated by the highly correlated photometric features. This is a useful negative result: when a model's assumptions don't match the data's structure, no tuning can rescue it.

### Feature importance (Random Forest, with redshift)

| Feature | Relative importance |
|---|:---:|
| `redshift` | ~51% |
| `r_i` (colour) | ~14% |
| `g_r` (colour) | ~9% |
| `i_z` (colour) | ~6% |
| `u_g` (colour) | ~5% |
| raw bands (`z`, `g`, `i`, `u`, `r`) | ~12% combined |
| `alpha`, `delta` (position) | ~2% combined |

The ranking is physically sensible and independently recovers a century of astronomical practice: **distance (redshift) separates types coarsely, colour separates them finely, and sky position is irrelevant.** The model was told none of this. Notably, the engineered colour features outrank the raw magnitudes they were derived from — direct evidence that the feature engineering earned its keep.

---

## Interpretation

The most valuable finding is **not** the 97% figure — that number is inflated by leakage. The valuable finding is that the honest ceiling using photometry alone is roughly **88%**, and that this is achievable even with a simple linear model.

In a real deployment classifying raw telescope images before spectroscopy, a well-tuned Logistic Regression would perform comparably to a Random Forest. The natural place to invest further effort would be richer photometric features or better data — **not** more sophisticated models.

This illustrates a broader principle worth stating plainly:

> On well-structured tabular problems, **feature quality dominates model choice.** The gap between a linear and a nonlinear model is often smaller than practitioners expect, while the gap between a leaky feature set and an honest one is usually larger.

---

## Limitations

This is a learning project and should be read as such:

- **Single train/test split** for the headline four-way table (though the bake-off uses cross-validation). Reported accuracies carry some variance.
- **Default hyperparameters** were used throughout. Deliberate tuning would likely narrow the small gap between Logistic Regression and Random Forest.
- **Multicollinearity** between the raw bands and derived colour features inflates individual Logistic Regression coefficients, so per-feature coefficient interpretation should be treated cautiously.
- **SDSS-specific.** The model is trained on SDSS's particular filters and would not transfer directly to other surveys (Pan-STARRS, LSST) without retraining, because their filters have different wavelength profiles.
- **The 88% ceiling is a baseline, not a state-of-the-art result** — it is almost certainly improvable with richer features or more careful modelling.

---

## Future Work

- **Cross-validated benchmarking** of the full four-way table (not just the bake-off) to tighten confidence intervals.
- **"Colours only" ablation** — drop the raw `u`/`g`/`r`/`i`/`z` bands entirely and model with just the four colour features, testing whether they carry essentially all the non-redshift signal.
- **Probability calibration** analysis — checking whether the model's confidence scores are reliable, which matters for triage pipelines that flag uncertain objects for human review.
- **Distribution-shift test** — train on one region of sky, test on another, to probe how well the classifier generalises across the survey.
- **Domain adaptation** to other photometric surveys.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/stellar-classification.git
cd stellar-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    Available on Kaggle: "Stellar Classification Dataset - SDSS17"
#    Place star_classification.csv in the project root (or update the path in the notebook)

# 4. Open the notebook
jupyter notebook stellar_classification.ipynb
```

The notebook runs top to bottom. If using Google Colab, mount your Drive and update the `file_path` variable to point at the CSV.

---

## Tech Stack

- **Python 3**
- **pandas**, **NumPy** — data handling
- **scikit-learn** — modelling (Logistic Regression, Random Forest, cross-validation, metrics)
- **matplotlib**, **seaborn** — visualisation
- **Google Colab** — development environment

---

## Data Source

Sloan Digital Sky Survey (SDSS) — stellar classification dataset, widely available on Kaggle as *"Stellar Classification Dataset - SDSS17"*. The survey data itself is a public release of the SDSS collaboration.

---

*Built as a self-directed learning project. Comments, corrections, and follow-up questions are welcome.*
# Stellar-Classification-Algorithm
End to end ML project classifying SDSS sky objects as galaxies, stars, or quasars from photometric data. Focus on quantifying data leakage (redshift) and honest reporting over headline accuracy. Compares Logistic Regression vs Random Forest across a cross-validated bake-off. Python, scikit-learn, pandas.
