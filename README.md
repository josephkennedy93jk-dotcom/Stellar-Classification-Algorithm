<h1 align="center">Stellar Classification with Machine Learning</h1>

<p align="center">
  <em>Classifying Sloan Digital Sky Survey objects as galaxies, stars, or quasars — with a controlled data-leakage experiment at the heart of the methodology.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square"/>
  <img src="https://img.shields.io/badge/scikit--learn-Modelling-F7931E?style=flat-square"/>
  <img src="https://img.shields.io/badge/Domain-Astronomy-6C4CB4?style=flat-square"/>
  <img src="https://img.shields.io/badge/Method-Leakage%20Experiment-EB6E4B?style=flat-square"/>
  <img src="https://img.shields.io/badge/Validation-5--Fold%20CV-4479A1?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-SDSS-000000?style=flat-square"/>
</p>

---

## 1. About This Project

Most stellar-classification projects report an accuracy number and stop. This one asks the harder question: **how much of that accuracy depends on a feature that wouldn't be available in a realistic image-only pipeline?**

The centrepiece is a controlled **data-leakage experiment**: each model is trained with and without redshift, a feature that in a real pre-spectroscopy triage workflow would not yet exist. The gap between those two scores is the story.

Built end-to-end on the Sloan Digital Sky Survey (SDSS) stellar classification dataset — roughly 100,000 rows × 12 features — the project delivers a full analytics stack:

- Interrogated the data with four blunt EDA questions before touching a model
- Engineered four physically motivated colour indices from the raw photometric bands
- Benchmarked two model families side-by-side to diagnose the *shape* of the problem, not crown a winner
- Ran the with/without-redshift experiment as a controlled comparison, holding split, seed and features constant
- Cross-validated seven classifiers to confirm the choice of model wasn't fragile

---

## Table of Contents

- [1. About This Project](#1-about-this-project)
- [2. The Analytical Question](#2-the-analytical-question)
- [3. Executive Summary](#3-executive-summary)
- [4. Key Findings](#4-key-findings)
- [5. Recommendations to Survey Programme Leadership](#5-recommendations-to-survey-programme-leadership)
- [6. Data Foundation](#6-data-foundation)
- [7. Data Cleaning & Preparation](#7-data-cleaning--preparation)
- [8. Exploratory & Descriptive Analysis](#8-exploratory--descriptive-analysis)
- [9. Feature Engineering](#9-feature-engineering)
- [10. Modelling Approach & The Leakage Experiment](#10-modelling-approach--the-leakage-experiment)
- [11. Cross-Validated Bake-Off](#11-cross-validated-bake-off)
- [12. Feature Importance & Interpretation](#12-feature-importance--interpretation)
- [13. Limitations](#13-limitations)
- [14. Deliverables](#14-deliverables)
- [15. What I'd Do Next](#15-what-id-do-next)
- [16. Author](#16-author)

---

## 2. The Analytical Question

Modern sky surveys photograph millions of objects every night — far more than any team of astronomers could classify by hand. The task is therefore operational, not just scientific:

> Can we automatically sort each observed object into **galaxy, star, or quasar** using only the measurements a telescope already captures?

A subtle but important distinction runs through the whole project: classifying an object **before** spectroscopic follow-up (from images alone) is a genuinely harder and more useful task than classifying it **after**, because the most predictive feature — redshift — is only obtained through that expensive follow-up. Most of the methodology here exists to respect this distinction.

The analytics stack was designed to answer four questions:

- **Can photometric measurements alone sort the three classes?** — a triage-quality decision from image data.
- **How much of the reported accuracy is redshift doing?** — quantify the leakage explicitly rather than hiding it in a headline number.
- **Does the choice of model matter?** — is the problem linear or does it need nonlinearity to crack?
- **What features carry the honest signal?** — separate the physically meaningful drivers from the shortcuts.

---

## 3. Executive Summary

The controlled leakage experiment shows redshift is worth 7–9 percentage points of accuracy — a large, often-hidden shortcut. Without it, both a linear and a nonlinear model tie at ~88%, meaning the photometric signal is largely linear and the honest accuracy ceiling for an image-only triage pipeline sits around that mark.

The project reframes the deliverable from *"predict class"* to *"predict class honestly, and know what feature quality is doing for you"* — a discipline transferable to any predictive setting where post-hoc features risk sneaking into the training set.

| Metric | Value |
|---|---|
| Rows in analytical base | ~100,000 SDSS objects |
| Baseline (guess "galaxy") | 60% |
| Honest accuracy ceiling (photometry alone) | ~88% |
| Leaky accuracy ceiling (with redshift) | ~97% |
| Redshift's contribution | 7–9 percentage points |
| Best cross-validated model | Random Forest (97.9%) |

---

## 4. Key Findings

The analysis surfaced five findings that shaped the interpretation.

- **Redshift contributes 7–9 points on its own.** When held constant, everything else, this is the size of the shortcut. Any headline accuracy number that includes redshift is not comparable to one that excludes it.
- **The photometric signal is largely linear.** Without redshift, a logistic regression and a random forest tie at 88%. The nonlinear model only pulls ahead once redshift is added — meaning the interaction between redshift and the colour features is where the nonlinear structure lives.
- **Engineered colours outrank the raw bands they were derived from.** Direct evidence that borrowing domain knowledge (astronomers' century-old colour-index convention) then empirically validating it earned its keep in the model.
- **Naive Bayes collapses on this problem.** Its independence assumption is badly violated by the highly correlated photometric bands. A useful negative result: when model assumptions don't match data structure, no tuning rescues it.
- **The Random Forest independently recovers a century of astronomical practice.** Distance (redshift) separates types coarsely, colour separates them finely, sky position is irrelevant. The model was told none of this — the fact that the importance ranking reads exactly this way is confirmation that the pipeline is physically sensible.

---

## 5. Recommendations to Survey Programme Leadership

The following recommendations are drawn from the findings above and are intended as strategic guidance for anyone building an image-based classification or triage pipeline on top of a photometric survey.

### 1. Report accuracy against a clearly-declared feature set
The single most important discipline is that every accuracy number should be labelled with the features it used. A 97% "with redshift" and an 88% "photometry only" are two different problems, not two versions of the same one.

### 2. Design the pipeline around the pre-spectroscopy problem, not the post-
An image-only triage classifier is the operationally useful artefact. Building for the harder problem from the start prevents accidental deployment of a model that requires data the production pipeline won't have.

### 3. Invest in richer photometric features before more sophisticated models
On this dataset, the gap between a linear and a nonlinear model on the honest problem is negligible. The next 5 points of accuracy will come from better data — additional bands, morphological features, time-domain photometry — not from a fancier classifier.

### 4. Use colour indices as first-class features, not derived afterthoughts
The engineered colours outrank the raw magnitudes they are built from in every model tested. Any pipeline that only feeds raw bands is under-using the signal that's already there.

### 5. Add calibrated probabilities, not just class labels
A triage pipeline flags objects for expensive follow-up. Calibrated confidence scores — from Platt scaling or isotonic regression — let the survey team spend follow-up budget on the objects where the classifier is genuinely uncertain, rather than treating every prediction identically.

### 6. Cross-validate model choice; single splits inflate confidence
The 5-fold bake-off in this project (Section 11) showed the tree ensembles cluster tightly at the top — a single split couldn't have distinguished them reliably. Any model-selection claim should carry variance estimates.

### 7. Retrain per survey; SDSS filters don't transfer to LSST or Pan-STARRS
This model is SDSS-specific. A pipeline serving multiple surveys needs per-survey retraining, or an explicit domain-adaptation layer.

### Areas to explore further
Three areas warrant deeper investigation: a colours-only ablation to test whether raw bands add anything the engineered colours don't already carry, a distribution-shift test across sky regions to probe generalisation, and probability calibration analysis to make the classifier deployable as a triage tool rather than just a labeller.

---

## 6. Data Foundation

The analysis draws on the SDSS stellar classification dataset — one row per observed sky object, covering identifiers, sky position, five photometric brightness bands, and (for a subset) spectroscopic follow-up.

**Scope**
- ~100,000 rows, each describing one observed sky object
- Target: `class` (multi-class — GALAXY / STAR / QSO)
- Class balance: galaxies dominate at ~60% (any model must beat 60%, the score achievable by guessing "galaxy" every time)

**Columns**

| Group | Columns | Role |
|---|---|---|
| **Identifiers** | `obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `spec_obj_ID`, `plate`, `MJD`, `fiber_ID` | Bookkeeping — which telescope run captured the object. No predictive value. **Dropped.** |
| **Sky position** | `alpha` (right ascension), `delta` (declination) | Where the object sits on the celestial sphere. Should be irrelevant to type. |
| **Photometric bands** | `u`, `g`, `r`, `i`, `z` | Brightness through five filters, ultraviolet to infrared. **The core signal.** |
| **Spectroscopic** | `redshift`, `class` | `class` is the target label. `redshift` is highly predictive but only available post-spectroscopy — the leakage risk. |

The five photometric bands are **magnitudes** — a logarithmic brightness scale where *lower numbers mean brighter*. Because they all measure the same object through different filters, they are strongly correlated with one another (a fact that directly motivates the feature engineering below).

Raw source data: [star_classification.csv](Raw%20CSV%20Files/star_classification.csv) · sourced from [Kaggle — Stellar Classification Dataset (SDSS17)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17).

---

## 7. Data Cleaning & Preparation

Data-quality treatment was minimal but deliberate — SDSS data is generally clean, so the value came from removing what should not be modelled rather than fixing what was broken.

- **Duplicates & missing values:** none material.
- **Sentinel outliers:** SDSS uses values like `-9999` to flag failed measurements; these were checked for and handled.
- **ID-like columns dropped:** `obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `spec_obj_ID`, `plate`, `MJD`, `fiber_ID` — arbitrary numbers with no predictive meaning.
- **Stratified 80/20 train/test split** so the minority class (quasars) is represented proportionally in both sets.
- **Scaler fit on training data only**, then applied to the test set — preventing test-set information leaking into preprocessing.
- **Features scaled only for scale-sensitive models** (Logistic Regression, KNN); tree models use raw features since they split on thresholds.

---

## 8. Exploratory & Descriptive Analysis

Before modelling, the data was interrogated with four blunt questions.

**Is the data even okay?** — checked shape, missing values, duplicates, sentinel outliers.

**What am I predicting?** — examined class balance. Galaxies dominate at roughly 60%, setting the baseline any model must beat.

<p align="center">
  <img src="Project%20Images/class-distribution.png" alt="Distribution of stellar body classes" width="720">
</p>

**Are the input features usable as-is?** — plotted the distribution of every photometric band. The bands are on similar magnitude scales but each carries a distinct shape.

<p align="center">
  <img src="Project%20Images/band-distributions.png" alt="Distribution of the five photometric bands" width="820">
</p>

**Are inputs redundant?** — a correlation heatmap showed the five bands moving together almost identically, signalling that feeding all five raw magnitudes into a model would be wasteful. This finding pointed directly toward feature engineering.

<p align="center">
  <img src="Project%20Images/correlation-matrix.png" alt="Correlation matrix of numeric features" width="820">
</p>

Also confirmed via a redshift-by-class plot that redshift almost perfectly separates the three types — flagging it as a leakage risk from the start.

---

## 9. Feature Engineering

Absolute brightness is a poor discriminator because it confounds an object's intrinsic nature with its distance: a dim nearby star and a bright faraway galaxy can register similar raw magnitudes.

The fix — a convention astronomers have used for over a century — is the **colour index**: the difference in brightness between two adjacent bands. Colour captures the *shape* of an object's light spectrum, independent of distance or overall brightness. Four colour features were engineered:

```python
df['u_g'] = df['u'] - df['g']
df['g_r'] = df['g'] - df['r']
df['r_i'] = df['r'] - df['i']
df['i_z'] = df['i'] - df['z']
```

This is **borrowed domain knowledge, then empirically validated**. The choice of colour indices was sourced from the field's established practice, then confirmed by the modelling results — colours consistently outrank the raw bands they are built from (see Section 12).

<p align="center">
  <img src="Project%20Images/color-color-diagrams.png" alt="Color–color diagrams of the four engineered colours by class" width="820">
</p>

The colour–colour diagrams show the three classes occupying visibly different regions of the colour space — direct evidence the engineered features carry class-separating signal.

---

## 10. Modelling Approach & The Leakage Experiment

Two model families were used deliberately — not to crown a winner, but to diagnose the shape of the problem.

- **Logistic Regression** — a linear baseline. Draws straight-line decision boundaries, fully interpretable via its coefficients, fails visibly when relationships are nonlinear (which is informative).
- **Random Forest** — an ensemble of decision trees that vote. Captures curved boundaries and feature interactions, insensitive to feature scaling, reports feature importances.

**A large gap between the two signals nonlinear structure. A small gap signals the problem is close to linear. This comparison is a diagnostic, not a contest.**

**The leakage experiment.** Each model was trained **twice** — once *without* redshift, once *with* it — holding everything else constant (same split, same seed, same features otherwise). Redshift is scientifically valid and enormously predictive, but because it comes from spectroscopy performed *after* image capture, a classifier meant to triage raw images cannot legitimately use it. Including it therefore constitutes **proxy data leakage**.

**Logistic Regression — without redshift (honest)**

<p align="center">
  <img src="Project%20Images/confusion-matrix-logreg-no-redshift.png" alt="Logistic regression confusion matrix — no redshift" width="620">
</p>

<p align="center">
  <img src="Project%20Images/logreg-coefficients-no-redshift.png" alt="Logistic regression coefficients — no redshift" width="820">
</p>

**Logistic Regression — with redshift (leaky)**

<p align="center">
  <img src="Project%20Images/confusion-matrix-logreg-with-redshift.png" alt="Logistic regression confusion matrix — with redshift" width="620">
</p>

<p align="center">
  <img src="Project%20Images/logreg-coefficients-with-redshift.png" alt="Logistic regression coefficients — with redshift" width="820">
</p>

Notice how the redshift coefficient dominates when included — the model is doing most of its work on that one feature.

**Result — the four-way comparison**

| Model | Without redshift (honest) | With redshift (leaky) |
|---|:---:|:---:|
| Logistic Regression | 88% | 95% |
| Random Forest | 88% | **97%** |

Two observations:

- **Redshift contributes 7–9 percentage points.** The leakage, quantified. For a realistic image-only pipeline, the honest accuracy ceiling with these features is roughly **88%**.
- **Random Forest only outperforms Logistic Regression when redshift is included.** Without it, both tie at 88% — the photometric signal is largely linear. With redshift, the nonlinear interaction between redshift and the colour features gives the forest a small edge.

---

## 11. Cross-Validated Bake-Off

To check whether the choice of model mattered, seven classifiers were benchmarked with **5-fold cross-validation** — a more robust estimate than a single train/test split, since it averages performance across five different data partitions. Models tested: Logistic Regression, K-Nearest Neighbours, Naive Bayes, Decision Tree, Random Forest, Extra Trees, and Gradient Boosting.

<p align="center">
  <img src="Project%20Images/cv-leaderboard.png" alt="Cross-validated leaderboard across seven classifiers" width="820">
</p>

**Leaderboard (with redshift)**

| Rank | Model | Mean CV Accuracy |
|---|---|:---:|
| 1 | Random Forest | 97.9% |
| 2 | Gradient Boosting | 97.7% |
| 3 | Extra Trees | 97.4% |
| 4 | Decision Tree | 96.6% |
| 5 | Logistic Regression | 95.5% |
| 6 | K-Nearest Neighbours | 94.3% |
| 7 | Naive Bayes | 76.3% |

The three tree ensembles cluster tightly at the top, confirming the original Random Forest choice was near-optimal. **Naive Bayes collapses** because its core assumption — features independent given the class — is badly violated by the highly correlated photometric features. A useful negative result: when a model's assumptions don't match the data's structure, no tuning can rescue it.

---

## 12. Feature Importance & Interpretation

<p align="center">
  <img src="Project%20Images/rf-feature-importance.png" alt="Random Forest feature importance with redshift" width="820">
</p>

<p align="center">
  <img src="Project%20Images/confusion-matrix-rf-with-redshift.png" alt="Random Forest confusion matrix — with redshift" width="620">
</p>

| Feature | Relative importance |
|---|:---:|
| `redshift` | ~51% |
| `r_i` (colour) | ~14% |
| `g_r` (colour) | ~9% |
| `i_z` (colour) | ~6% |
| `u_g` (colour) | ~5% |
| raw bands (`z`, `g`, `i`, `u`, `r`) | ~12% combined |
| `alpha`, `delta` (position) | ~2% combined |

The ranking is physically sensible and independently recovers a century of astronomical practice: **distance (redshift) separates types coarsely, colour separates them finely, and sky position is irrelevant.** The model was told none of this. Notably, the engineered colour features outrank the raw magnitudes they were derived from — direct evidence the feature engineering earned its keep.

**Interpretation**

The most valuable finding is **not** the 97% figure — that number is inflated by leakage. The valuable finding is that the honest ceiling using photometry alone is roughly **88%**, and that this is achievable even with a simple linear model.

In a real deployment classifying raw telescope images before spectroscopy, a well-tuned Logistic Regression would perform comparably to a Random Forest. The natural place to invest further effort would be richer photometric features or better data — **not** more sophisticated models.

> On well-structured tabular problems, **feature quality dominates model choice.** The gap between a linear and a nonlinear model is often smaller than practitioners expect, while the gap between a leaky feature set and an honest one is usually larger.

---

## 13. Limitations

Analytical honesty — the failure modes worth naming explicitly.

- **Single train/test split** for the headline four-way table (though the bake-off uses cross-validation). Reported accuracies carry some variance.
- **Default hyperparameters** were used throughout. Deliberate tuning would likely narrow the small gap between Logistic Regression and Random Forest.
- **Multicollinearity** between the raw bands and derived colour features inflates individual Logistic Regression coefficients, so per-feature coefficient interpretation should be treated cautiously.
- **SDSS-specific.** The model is trained on SDSS's particular filters and would not transfer directly to other surveys (Pan-STARRS, LSST) without retraining, because their filters have different wavelength profiles.
- **The 88% ceiling is a baseline, not a state-of-the-art result** — almost certainly improvable with richer features or more careful modelling.

---

## 14. Deliverables

The tangible outputs of the engagement, each linked below:

| Deliverable | File |
|---|---|
| Full analysis notebook | [Stellar_classification.ipynb](Python%20Files/Stellar_classification.ipynb) |
| Raw source data | [star_classification.csv](Raw%20CSV%20Files/star_classification.csv) |
| Chart gallery | [Project Images](Project%20Images) |

---

## 15. What I'd Do Next

Given more time or a follow-up engagement, five extensions would materially strengthen the project:

- **Cross-validated benchmarking of the full four-way table** (not just the bake-off) to tighten confidence intervals on the leakage-experiment numbers.
- **"Colours only" ablation** — drop the raw `u`/`g`/`r`/`i`/`z` bands entirely and model with just the four colour features, testing whether they carry essentially all the non-redshift signal.
- **Probability calibration analysis** — check whether the model's confidence scores are reliable. Matters for triage pipelines that flag uncertain objects for human review.
- **Distribution-shift test** — train on one region of sky, test on another, to probe how well the classifier generalises across the survey.
- **Domain adaptation** to other photometric surveys (Pan-STARRS, LSST) — the SDSS-specific filter dependence is the biggest barrier to real-world reuse.

---

## 16. Author

**Joseph Kennedy** — Data Analyst

End-to-end delivery: exploratory analysis, feature engineering (photometric colour indices), controlled leakage experiment, benchmarking across seven classifiers with 5-fold cross-validation, feature-importance interpretation grounded in domain knowledge, and honest communication of what the model can and cannot claim.

<sub>Built on the publicly available SDSS Stellar Classification Dataset (SDSS17), sourced via Kaggle. Results are for analytical demonstration only.</sub>
