# Methods and design notes

Long-form writeup of the design behind this project (moved here from the original project README). For setup, see the [root README](../README.md). File paths below are relative to `app/`.

A self-assessment platform that classifies obesity risk from eleven plain-language lifestyle questions, escalates borderline cases to a clinically-grounded NHANES model, scores diabetes risk against ADA 2024 guidelines, and returns citation-grounded nutrition guidance from a RAG pipeline over indexed clinical PDFs.

This document explains not just *how* to run the system, but *why* each component is built the way it is. The design choices are tightly coupled to the clinical-screening problem the project addresses, and copy-pasting the run commands without understanding the rationale tends to produce silent failures (dimension mismatches, escalation never firing, leakage being reintroduced during retraining).

---

## Table of contents

1. [Why this exists: the screening gap](#1-why-this-exists-the-screening-gap)
2. [System architecture at a glance](#2-system-architecture-at-a-glance)
3. [Component map](#3-component-map)
4. [Stage 1 design rationale](#4-stage-1-design-rationale)
5. [The composite confidence score](#5-the-composite-confidence-score)
6. [Stage 2 design rationale](#6-stage-2-design-rationale)
7. [Diabetes risk module](#7-diabetes-risk-module)
8. [RAG pipeline design rationale](#8-rag-pipeline-design-rationale)
9. [Prerequisites](#9-prerequisites)
11. [Start Milvus](#11-start-milvus)
12. [Start the FastAPI server](#12-start-the-fastapi-server)
13. [Start the Streamlit frontend](#13-start-the-streamlit-frontend)
14. [End-to-end startup recipe](#14-end-to-end-startup-recipe)
15. [API reference](#15-api-reference)
16. [Retraining](#16-retraining)
17. [Environment variables](#17-environment-variables)
18. [Troubleshooting](#18-troubleshooting)
19. [Limitations](#19-limitations)

---

## 1. Why this exists: the screening gap

WHO estimates over 650 million adults globally live with obesity, but most are not formally classified until complications appear (type 2 diabetes, cardiovascular disease, musculoskeletal disorders). The barrier is methodological. Accurate BMI requires a scale and a stadiometer, lipid panels and HbA1c require phlebotomy, and the translation of a numeric BMI into a WHO severity category is rarely communicated to patients in actionable terms.

The ML literature on obesity classification has largely sidestepped this barrier by training on `Weight`, `Height`, and exact BMI as features. Since `BMI = Weight / Height²` is an algebraic identity, the model is being asked to solve a deterministic equation rather than learn lifestyle-mediated risk. Near-perfect scores in such papers are the product of that shortcut: those models cannot generalize to a self-assessment context where the user does not know their exact body mass.

This project is built around the deployment scenario where a user answers eleven plain-language lifestyle questions, optionally provides clinical measurements if available, and receives an obesity classification plus a tailored nutrition plan. No laboratory work is required at baseline.

### Why two stages

A single classifier cannot honestly span this problem because:

- The Overweight I and Overweight II boundary (BMI 25–29.9 vs. 30+) is the hardest discrimination in the WHO target space and the one where lifestyle features alone are least able to separate adjacent classes.
- Clinical measurements (waist circumference, fasting glucose, HbA1c) add real discriminative signal in this boundary region, but requiring them upfront defeats the screening goal.

The solution is to make Stage 1 the default path, route only borderline overweight cases to Stage 2 with optional clinical inputs, and degrade gracefully when those inputs are missing (the Model 2 pipeline imputes via medians).

---

## 2. System architecture at a glance

```
                        ┌──────────────────────┐
                        │   Streamlit UI       │
                        │  (3 tabs, port 8501) │
                        └───────┬──────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        Assessment / Nutrition           Index PDFs / chat
                │                               │
                ▼                               ▼
       ┌────────────────┐              ┌──────────────────┐
       │  FastAPI :8001 │              │  Indexer (RAG)   │
       └───────┬────────┘              └─────┬────────────┘
               │                              │
   ┌───────────┴──────────────┐               ├── Milvus :19530
   │                          │               │   (HNSW / IVF_PQ /
   ▼                          ▼               │    DiskANN)
Model 1                  Model 2              │
12-feat RF               40-feat RF           ├── BGE-large-en-v1.5
(lifestyle, 7 classes)   (NHANES, OW_I/II)    │   (1024-dim, local)
                                              │   OR
                                              │   Gemini embedding-001
                                              │   (3072-dim, API)
                                              │
                                              └── LangGraph
                                                  Translate → Retrieve
                                                  → Generate → Reflect
                                                  → Revise (≤2x) → Translate
                                                       │
                                                       ▼
                                           Gemini API  /  Anthropic API 
                                           (chat backend, swappable)
```

The five layers (Streamlit, FastAPI, two-stage model pipeline, Milvus, LangGraph) are independently deployable. The FastAPI server is required for the Assessment tab; Milvus and the indexer are only required for the Nutrition Plan and Index PDFs tabs.

---

## 3. Component map

| File | Role |
|---|---|
| `model_pipeline.py` | Trains Model 1, defines `run_inference`, the composite `compute_confidence_for_class`, the `should_trigger_model2` router, `run_model2_inference`, and `assess_diabetes_risk`. |
| `main.py` | FastAPI server. Loads the Model 1 bundle once at startup, lazy-loads Model 2 bundles by gender on escalation. |
| `obesity_app_v2.py` | Streamlit frontend. Three tabs: Assessment, Nutrition Plan (RAG chat), Index PDFs. Handles backend toggling between Gemini and the Anthropic API. |
| `indexer.py` | PDF ingestion (`pdfminer.six`), chunking (400 tokens, 50 overlap), embedding (BGE local or Gemini API), Milvus collection management with three index types, and the LangGraph pipeline. |
| `resave_bundles.py` | Retrains the three Model 2 bundles (pooled / male / female) from the merged NHANES CSVs. |
| `ObesityDataSet_raw_and_data_sinthetic.csv` | UCI obesity dataset, n=2,111. Real survey responses plus SMOTE augmentation. Used to train Model 1. |
| `merged_data/` | NHANES 2021–2023 fasting subsample, cleaned and merged. Pooled, male, and female variants. Used to train Model 2. |
| `pdfs/` | 14 clinical guideline and research PDFs to index (not redistributed; see [sources.md](sources.md)). |
| `obesity_model_bundle.joblib` | Pre-trained Model 1 bundle: 12-feature RF, scaler, class centroids, demographic profiles, tuned confidence weights, label mapping. ~25 MB. |
| `model2_bundles/` | Three pre-trained Model 2 bundles + `model2_metadata.json` listing the 40 NHANES features. |
| `obesity_rf_metadata.json` | Human-readable Model 1 metadata: hyperparameters, features, confidence weights. |

Pre-trained bundles are published as GitHub Release assets (see the root README), so retraining is only needed if you change the data.

---

## 4. Stage 1 design rationale

### Dataset

UCI Obesity Dataset, n=2,111 across seven WHO classes:

```
Insufficient_Weight, Normal_Weight,
Overweight_Level_I, Overweight_Level_II,
Obesity_Type_I, Obesity_Type_II, Obesity_Type_III
```

The data is partially synthetic: real survey responses augmented with SMOTE to balance class distribution. Pre-augmentation imbalance ratio is 1.29, modest enough that further resampling is not used during training.

### The leakage problem and its cost

The raw dataset contains `Weight` (kg) and `Height` (m). Exact BMI is `Weight / Height²`, an algebraic identity. Including either weight, height, or exact BMI gives the model a deterministic route to the WHO target class, so a model trained on them is solving an identity rather than learning lifestyle-mediated risk. `notebooks/00_model1_development.ipynb` compares an exact-BMI configuration against the leakage-free BMI-bucket configuration to make that shortcut explicit. Any future feature engineering that recomputes BMI from `Weight` and `Height` reintroduces the leak; the training pipeline drops `Height`, `Weight`, `BMI`, and any BMI-derived interactions explicitly to prevent this.

### Why a four-level BMI bucket

The user does not need to enter exact body mass in a self-assessment. They self-select one of four buckets aligned to WHO cutoffs:

```
0 = Underweight (<18.5)
1 = Normal      (18.5–25)
2 = Overweight  (25–30)
3 = Obese       (>30)
```

This preserves the spirit of self-report (no scale required) while keeping coarse body-size information. `BMI_bucket` is the top feature in the Stage 1 importance ranking.

### The 12-feature set

Final features: `Gender`, `Age`, `family_history_with_overweight`, `FCVC` (vegetable consumption), `NCP` (meals per day), `CAEC` (snacking frequency), `CH2O` (water intake), `FAF` (physical activity days), `TUE` (screen time), `CALC` (alcohol consumption), `BMI_bucket`, `Age_x_FCVC`.

What was dropped and why:

- `Height`, `Weight`, `BMI` - leakage, see above.
- `SMOKE`, `SCC` - near-zero variance (98% and 95% modal-value frequency); the model learns nothing from a column that is the same value 98% of the time.
- `MTRANS` (transport mode dummies) - combined SHAP importance below 0.015.
- `FAVC` (frequent high-calorie food) - SHAP importance below the same 0.015 threshold.
- `Weight_x_FAF`, `BMI_x_FAF` interactions - derived from the leaked features, dropped with them.

What was added: `Age_x_FCVC`, the product of standardized age and vegetable consumption. The intuition is that older individuals who eat more vegetables break the expected obesity pattern. This interaction is kept on the basis of its mean SHAP importance.

### Encoding strategy

| Feature type | Variables | Strategy | Why |
|---|---|---|---|
| Binary yes/no | Gender, family_history | 0/1 map | Single bit is sufficient |
| Frequency-ordered | CAEC, CALC | Ordinal 0–3 | Preserves intensity ordering (no, sometimes, frequently, always) |
| Continuous | Age, FCVC, NCP, CH2O, FAF, TUE | Standardized (μ=0, σ=1) | Required for cosine similarity in the confidence scorer |
| Engineered | Age × FCVC | Product interaction | Captures diet effect varying with age |
| Target | NObeyesdad (7 classes) | Label encoded 0–6 | Ordered severity integer |

The Random Forest itself does not need scaled features for tree construction, but the cosine similarity component of the confidence scorer does, so the same scaler is fit once and used in both places.

### Model selection

Three architectures were compared under five-fold stratified CV during development (notebook 00): Logistic Regression, Gradient Boosting, and Random Forest. Random Forest was selected.

Random Forest was chosen for three reasons: calibrated per-class probabilities (needed for the confidence scorer), Gini importances that closely matched SHAP rankings (post-hoc interpretability without a secondary explanation model), and native handling of non-linearity without feature scaling at the tree level.

Hyperparameters are tuned with `RandomizedSearchCV` over 40 iterations under five-fold stratified CV, optimizing F1-macro. The selected configuration is stored in `obesity_rf_metadata.json`:

```json
{
  "n_estimators": 400, "max_depth": 20, "min_samples_leaf": 1,
  "max_features": "sqrt", "class_weight": "balanced"
}
```

### Error adjacency

Stage 1 errors are inspected for clinical adjacency (one severity step away) in notebook 00. This matters because a patient predicted as Overweight II rather than Obesity I receives nutritionally similar guidance, whereas a distant error (Normal Weight predicted as Obesity II, for example) would be a dangerous failure mode.

---

## 5. The composite confidence score

A raw softmax probability from a Random Forest is not a sufficient routing signal for a clinical decision. Random Forest probabilities are aggregated leaf-vote frequencies, which can be confidently wrong on inputs that fall in regions sparsely sampled by the training data. The confidence scorer mitigates this by combining four signals.

### Formula

```
confidence = a·x + b·y - c·z + d·m
```

| Term | Source | Rationale |
|---|---|---|
| **x** = cosine similarity to predicted class centroid | Geometric agreement | Is the input actually near the class it was assigned to in scaled feature space? |
| **y** = softmax probability for predicted class | Model's own belief | Standard signal, kept but not relied on alone |
| **z** = weighted cosine similarity to top-2 runner-up centroids | Confusion risk (subtracted) | Penalizes inputs near class boundaries |
| **m** = demographic alignment (age, gender, family history) | Population fit | Does the input match the typical demographic profile of the predicted class? |

The `m` term is itself a weighted blend: 0.4 × age agreement + 0.4 × gender match + 0.2 × family history match.

### How the weights are tuned

The four weights `(a, b, c, d)` are optimized empirically against OOF predictions from five-fold CV. The objective is the separation gap between the score distributions of correctly versus incorrectly classified samples. The optimization uses grid search over the four-dimensional parameter space followed by Nelder-Mead fine-tuning.

The tuned weights for the shipped bundle:

```
a (cosine sim)        = 0.0659
b (model proba)       = 0.99
c (confusion risk)    = 0.99
d (demographic align) = 0.0603
```

Note that `b` and `c` carry almost all the signal, and `a` and `d` act as small corrections. This is consistent with the model probability being the strongest single predictor, with confusion-risk subtraction doing the heavy lifting on borderline cases.

### Tier mapping (for UI display)

```
≥ 0.85   High confidence
≥ 0.70   Moderate confidence - consider follow-up
≥ 0.55   Low confidence - borderline case
< 0.55   Very low confidence - inconclusive
```

The 75% threshold for Stage 2 escalation sits inside the "moderate" tier, capturing the borderline OW-OB region while avoiding unnecessary escalation for clear-cut cases.

---

## 6. Stage 2 design rationale

### When it triggers

The router (`should_trigger_model2` in `model_pipeline.py`) fires when **all** of the following hold:

1. Stage 1's top class is `Overweight_Level_I` or `Overweight_Level_II`.
2. **Either** confidence < 75%, **or** the runner-up is the other OW class.

The first condition restricts escalation to the Overweight boundary region, the hardest discrimination for lifestyle features. The second condition catches both the obvious case (low confidence anywhere) and the subtle case (high confidence in OW_I but with OW_II as a close second, or vice versa, which is the exact ambiguity Stage 2 is designed to resolve).

Outside this boundary region, escalation would add latency and clinical-input friction with no expected benefit.

### Training data

NHANES 2021–2023, MEC-examined adults aged 20+, excluding pregnant women and participants with incomplete body measures examinations. After filtering to the OW_I and OW_II classes and the fasting subsample (required for `LBXGLU` and `LBXIN`), n=2,289.

The fasting requirement halves the available NHANES population. The trade-off is conscious: glucose and insulin enable HOMA-IR, which is a strong discriminator for the metabolic-syndrome flavor of overweight that Stage 2 is meant to detect. An ablation comparing the full-variable model on the fasting subsample versus a glucose-free model on the full phlebotomy sample is listed under future work.

### The 40-feature set

Selected via an RF importance screen on the merged NHANES feature set. The 40 features cover four domains (full list in `model2_bundles/model2_metadata.json`):

- **Body composition.** `BMXWAIST`, `BMXHIP`, `BMXARMC`, `BMXARML`, `BMXLEG`, `WAIST_HT_RATIO`, `WAIST_HIP_RATIO`. Waist circumference is the most important single feature in this group, consistent with the clinical literature treating waist as a stronger metabolic-risk predictor than BMI alone.
- **Blood pressure.** `SYS_BP`, `DIA_BP`.
- **Glucose homeostasis.** `LBXGLU` (fasting glucose), `LBXIN` and `LBXIN_log` (fasting insulin), `HOMA_IR` and `HOMA_IR_log`, `LBXGH` (HbA1c). HOMA-IR is computed as `(glucose × insulin) / 405` and log-transformed because of the long right tail.
- **Liver and kidney panel.** `ALT_log`, `AST_log`, `AST_ALT_ratio_log`, `GGT_log`, `LBXSGB`, `LBXSCR`, `eGFR`, `BUN_CREATININE_RATIO`. Liver enzymes are included because NAFLD is overwhelmingly comorbid with metabolic-syndrome obesity.
- **Other.** `LBXTC` (total cholesterol), `LBXSUA` (uric acid), `WHQ070` (self-reported weight perception), `INDFMPIR` (poverty-income ratio), activity-level indicators.

### Why height, weight, exact BMI are still excluded

The same leakage argument as Stage 1. The label `OW_CLASS` is derived from BMI, so including BMI would let the pipeline solve the identity rather than learn from clinical biomarkers. Body composition (waist, hip, arm) is a separate measurement, not algebraically derived from height and weight, so it is retained.

### Pipeline structure

```
SimpleImputer(strategy='median') → RandomForestClassifier(
    n_estimators=400, max_depth=10, min_samples_leaf=4,
    max_features=0.5, random_state=42
)
```

Median imputation is what enables graceful degradation when the user provides only a subset of clinical inputs. If the user provides waist circumference and blood pressure but not glucose or insulin, the median values from the training set fill in the rest. The model output remains valid; only the confidence component reflecting input completeness drops.

### Gender-specific routing

Three bundles are trained: `model2_pooled_bundle.joblib`, `model2_male_bundle.joblib`, `model2_female_bundle.joblib`. At inference time, `_load_model2_bundle(gender)` selects the gender-matched bundle (`gender=0` is male, `gender=1` is female in the NHANES convention) and falls back to the pooled bundle if the gender-specific file is missing. This is biologically motivated: waist-hip ratio cutoffs, fat distribution patterns, and HOMA-IR distributions differ between sexes, so sex-specific models are trained alongside the pooled one.

### Shipped bundles vs notebook models

The shipped bundles were refit on the full data by `resave_bundles.py` with fixed hyperparameters, so they are not identical to the tuned notebook models.

### Stage 2 confidence

Different formula from Stage 1, because the inputs differ:

```
confidence = 0.85 · p + 0.15 · c
```

Where `p` is the model probability for the predicted class and `c` is the input completeness ratio (provided clinical inputs / 10 total). The completeness term acknowledges that a Stage 2 prediction made from waist circumference alone is structurally less trustworthy than one made from the full clinical panel, even when the model probability is identical.

---

## 7. Diabetes risk module

`assess_diabetes_risk` is rule-based, not ML. The thresholds come directly from the ADA Standards of Medical Care 2024, so the output is auditable against published guidelines and does not require a separate validation cohort.

### Inputs and minimum requirement

At least one of `glucose` (fasting plasma glucose, mg/dL) or `hba1c` (%) must be provided. The function returns `None` otherwise.

Optional inputs: `insulin` (enables HOMA-IR computation), `age`, `bmi`, `family_history`.

### Scoring

```
+3.0  Fasting glucose ≥ 126 mg/dL    (diabetic range)
+1.5  Fasting glucose 100–125 mg/dL  (prediabetes range)
+3.0  HbA1c ≥ 6.5%                   (diabetic range)
+1.5  HbA1c 5.7–6.4%                 (prediabetes range)
+1.5  HOMA-IR > 5.0                  (severe insulin resistance)
+0.75 HOMA-IR > 2.5                  (insulin resistance)
+0.5  BMI ≥ 30                       (obesity)
+0.3  Age ≥ 45
+0.5  Family history of overweight/diabetes
```

The total is normalized to 0–1 by dividing by 6.0 and clipping. The classification is determined by the highest-priority threshold that fires.

### Classifications

- **Possible Type 1 Diabetes** - flagged when low insulin (<5 μU/mL) coincides with diabetic-range glucose or HbA1c in someone under 40. This is meant to catch LADA (Latent Autoimmune Diabetes in Adults) presentations that would be missed by a Type 2-only rule. The output explicitly says this requires clinical confirmation and warns against self-diagnosis.
- **Type 2 Diabetes** - diabetic-range values without the Type 1 pattern.
- **Prediabetes** - values in the prediabetes range. The ADA-recommended intervention (5–7% weight loss, 150 min/week moderate activity) is included in the response note.
- **Low diabetes risk** - values within normal ranges.

The output also includes `driving_factors`, a human-readable list of the specific values that pushed the risk up. This is what the Streamlit UI displays in the diabetes risk card.

---

## 8. RAG pipeline design rationale

Classification alone produces a label. The deployment goal is actionable, evidence-grounded guidance. The RAG pipeline closes this gap by indexing 14 clinical PDFs (obesity management guidelines, AACE consensus, EASO guidelines, Joslin nutrition guideline, related papers) and retrieving passages that match the patient's specific obesity category and lifestyle profile.

### Indexing

- **Extraction.** `pdfminer.six` walks the PDF text containers. Tables and figures are skipped; only text blocks contribute to chunks.
- **Chunking.** 400-token segments with 50-token overlap. The overlap matters for clinical text where a key recommendation often spans a section header.
- **Embedding.** Two backends, picked by the user:
  - `BAAI/bge-large-en-v1.5` (1024-dim, local via `sentence-transformers`). No API key needed.
  - `gemini-embedding-001` (3072-dim, API). Requires `GEMINI_API_KEY`. Higher per-query latency due to the API round-trip.
- **Storage.** Milvus collection with COSINE similarity. Three index types are exposed:
  - `HNSW` - default, best recall, modest memory.
  - `IVF_PQ` - lowest memory and fastest queries, slightly lower recall.
  - `DiskANN` - disk-resident, for collections too large to fit in RAM.

### Why the dimension routing matters

A common failure mode: index with BGE (1024-dim), then query with Gemini (3072-dim) without rebuilding. Milvus throws a dimension-mismatch error that is opaque if you do not know what to look for.

The indexer mitigates this two ways. First, the Milvus schema is constructed using the active embedding dimension at collection creation time. Second, on Streamlit startup, the auto-reconnect logic reads the actual stored dimension from the collection schema and updates `EMBED_DIM` on the indexer module before any query is issued. Third, separate collection names (`papers_rag_gemini` and `papers_rag_interactive`) are checked in priority order so users can have both populated simultaneously without collisions.

### LangGraph generation with reflection

The retrieved passages are fed through a LangGraph state machine with seven nodes:

1. **Translate query** - if the input is not English, translate to English so retrieval matches the (English) corpus.
2. **Retrieve** - embed the query and search Milvus, returning top-k chunks with source, page, paper type, and similarity score.
3. **Generate** - draft an answer grounded in the retrieved context. The output format is hard-coded to require recommended-paper citations.
4. **Reflect** - a judge LLM checks the draft against retrieval-quality criteria (does the answer use the retrieved context? does it cite real titles? is it on-topic?) and emits `#Verdict: PASS` or `#Verdict: FAIL`.
5. **Revise** - if verdict is FAIL, regenerate with the reflection feedback as input. Loops up to two iterations.
6. **Iterate** - bumps a counter so the loop terminates.
7. **Translate output** - translate the final English answer back to the user's input language if needed.

The reflection loop addresses a known failure mode of naive RAG: generation models that ignore retrieved context and answer from parametric memory, producing plausible but uncited responses.

### Patient context serialization

When the user moves from Assessment to Nutrition Plan, the Stage 1 and Stage 2 results (predicted class, confidence score, top-3 predictions with individual confidences, all lifestyle inputs) are serialized into a structured natural-language string and prepended to the first RAG query. The generation model therefore has full awareness of the patient's obesity category, lifestyle profile, and prediction uncertainty without the user needing to re-describe themselves.

### Dual chat backend

The generation backend is swappable between two providers without code changes:

- **Gemini API.** `gemini-2.5-flash`, `gemini-2.5-pro`, or `gemini-2.0-flash-lite`. Requires `GEMINI_API_KEY`.
- **Anthropic API.** Public Anthropic Messages API via the `anthropic` SDK (Claude Sonnet, Haiku, or Opus). Requires `ANTHROPIC_API_KEY`.

The active indexer's `_call_claude` method is monkey-patched at query time with the appropriate API client and model identifier. Web search grounding (via Gemini's google_search tool) is available when enabled and adds a 3–5 sentence current-information snippet to the retrieval context.

### Export and resume

The indexed vector store can be exported as a compressed `.npz` archive (embeddings, texts, metadata) and reloaded in a later session without re-indexing. Both indexing and chat generation can be interrupted mid-execution via dedicated stop buttons in the Streamlit interface, which matters for a 14-PDF index pass that takes 1–3 minutes on first run after the BGE model download.

---

## 9. Prerequisites

### System

- Python 3.11 (matches the `__pycache__/` bytecode; 3.10 and 3.12 also work)
- Docker, only if you want a local Milvus (skip if you point at Zilliz Cloud or skip RAG entirely)
- ~4 GB free RAM for Milvus + BGE embedder; ~6 GB if you use Gemini 3072-dim embeddings

### API keys

The Assessment tab requires no keys. The Nutrition Plan tab requires one of:

- `GEMINI_API_KEY` (Google AI Studio). Used for both LLM generation and optionally as the embedding backend.
- `ANTHROPIC_API_KEY`, for routing generation through the Anthropic API to Claude.

Both can be entered in the Streamlit sidebar at runtime; environment variables are optional.

---

## 10. Install

See the Quickstart in the [root README](../README.md). Notes:

- `sentence-transformers` pulls `torch` as a dependency. Apple Silicon works out of the box; on Linux/CUDA, install a matching `torch` wheel first.
- `pymilvus` is the client library; it does not embed a Milvus server.
- The first run of the BGE embedder downloads `BAAI/bge-large-en-v1.5` (~1.3 GB) into the HuggingFace cache.

---

## 11. Start Milvus

Skip this section if you only want the Assessment tab. The indexer defaults to `http://localhost:19530`.

The simplest local setup is the official standalone Docker Compose file:

```bash
mkdir ~/milvus && cd ~/milvus
curl -L https://github.com/milvus-io/milvus/releases/latest/download/milvus-standalone-docker-compose.yml \
  -o docker-compose.yml
docker compose up -d
```

Verify it is reachable:

```bash
docker ps | grep milvus
curl http://localhost:9091/healthz
```

For Zilliz Cloud or a remote cluster, set the URI and token instead:

```bash
export MILVUS_URI="https://<your-zilliz-endpoint>"
export MILVUS_TOKEN="<your-token>"
export MILVUS_COLLECTION="papers_rag"   # optional, default papers_rag
```

The indexer reads these at startup.

---

## 12. Start the FastAPI server

This is required for the Streamlit Assessment tab regardless of whether you use RAG.

From the `app/` directory in its own terminal:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

`main.py` loads `obesity_model_bundle.joblib` once at import time. The Model 2 bundles in `model2_bundles/` are loaded lazily on first escalation, keyed by gender.

Sanity check:

```bash
curl http://localhost:8001/health
# {"status":"healthy"}

curl http://localhost:8001/features
# Model 1 feature list + Model 2 clinical input dictionary
```

---

## 13. Start the Streamlit frontend

In a second terminal with the venv active:

```bash
export OBESITY_API_URL="http://localhost:8001"   # default if unset
streamlit run obesity_app_v2.py
```

The app opens at `http://localhost:8501` with three tabs.

### Tab 1: Assessment

Form-driven. Collects the 11 lifestyle fields (plain-language radio buttons, no jargon) and any clinical values you want to provide. POSTs to `/predict/full`. Renders four blocks:

- **Final classification** with confidence tier and source attribution (Model 1 vs. Model 2).
- **Model 1 detail** including top-3 predictions, all probabilities, and the four score components (cosine sim, model proba, confusion risk, demographic alignment) so the confidence number is auditable.
- **Model 2 escalation** including whether it triggered, the trigger reason, and the list of clinical inputs the user did not provide (so they know what to supply for a higher-confidence refinement).
- **Diabetes risk** with classification, score, driving factors, and ADA-grounded note.

When Stage 2 triggers, the form exposes a clinical measurements section with a size reference guide (XS–2XL chest, waist, hip, arm circumferences for both sexes), so users without a tape measure can estimate from clothing size. The guide auto-selects the gender panel matching the lifestyle input.

### Tab 2: Nutrition Plan

RAG chat over the indexed PDFs. Sidebar toggle picks the backend:

- **Gemini direct** (default). Set `GEMINI_API_KEY` in the environment before launching, or paste it into the Index PDFs tab. Pick a model from `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash-lite`.
- **Anthropic API.** Toggle on; the key field is prefilled from `ANTHROPIC_API_KEY` (or paste it). The app monkey-patches `_call_claude` to call the Anthropic Messages API, so no Gemini key is needed for generation.

The Assessment results are automatically prepended to the first chat query as patient context. If you have not indexed the PDFs yet, the chat returns nothing useful; index first.

### Tab 3: Index PDFs

Where you build the Milvus collection that the chat queries.

Settings:

- **Embedding model.** `BGE (local, 1024-dim)` runs locally with no key. `Gemini embedding-001 (3072-dim)` requires `GEMINI_API_KEY` and sets `EMBED_DIM = 3072` on the indexer module before construction.
- **Index type.** `HNSW` (default), `IVF_PQ` (compressed), `DiskANN` (depends on your Milvus build supporting it).
- **Collection name.** Defaults to `papers_rag`. Use distinct names per embedding type to avoid dimension conflicts.
- **Chunk size / overlap.** 400 / 50 tokens by default.
- **Drop old collection.** Tick when re-indexing into the same name with different settings.

Click `Index PDFs`. Indexer walks `pdfs/`, extracts text, chunks, embeds, bulk-inserts. With BGE + HNSW on the 14 included PDFs, expect 1–3 minutes after first-run model download.

---

## 14. End-to-end startup recipe

Three terminals, in order:

```bash
# Terminal 1: Milvus (skip if you only want the Assessment tab)
cd ~/milvus && docker compose up -d

# Terminal 2: FastAPI
cd app && source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 3: Streamlit
cd app && source .venv/bin/activate
export GEMINI_API_KEY="<your-key>"   # or skip and paste in the UI
streamlit run obesity_app_v2.py
```

Then in the UI: index the PDFs once (Index PDFs tab), run an assessment (Assessment tab), ask questions (Nutrition Plan tab).

---

## 15. API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service banner with endpoint list |
| GET | `/health` | Liveness probe |
| GET | `/classes` | The 7 obesity classes with their integer codes |
| GET | `/features` | Model 1 feature list and Model 2 clinical input dictionary |
| GET | `/input-guide` | Full input schema with units, ranges, and ADA thresholds |
| POST | `/predict` | Model 1 only, lifestyle inputs |
| POST | `/predict/full` | Model 1 + auto-escalated Model 2 + diabetes risk |

### Model 1 only

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": 1, "age": 25, "family_history": 1,
    "fcvc": 2, "ncp": 3, "caec": 1, "ch2o": 2,
    "faf": 1, "tue": 1, "calc": 1, "bmi_bucket": 2
  }'
```

### Full pipeline

```bash
curl -X POST http://localhost:8001/predict/full \
  -H "Content-Type: application/json" \
  -d '{
    "lifestyle": {
      "gender": 1, "age": 45, "family_history": 1,
      "fcvc": 2, "ncp": 3, "caec": 1, "ch2o": 2,
      "faf": 1, "tue": 1, "calc": 1, "bmi_bucket": 2
    },
    "clinical": {
      "waist_cm": 102.0, "hip_cm": 105.0,
      "sys_bp": 135.0, "dia_bp": 85.0,
      "glucose": 118.0, "hba1c": 6.1,
      "exact_bmi": 28.4
    }
  }'
```

Response shape:

```
final_classification:    predicted_class, confidence_pct, tier, source
model1:                  predicted_class, confidence_pct, tier, close_call,
                         top3, all_probabilities, score_components
model2_escalation:       triggered, trigger_reason, result, clinical_inputs_missing
diabetes_risk:           risk_classification, risk_score, risk_score_pct,
                         note, driving_factors, values_used
```

### Input ranges (from the Pydantic schema)

```
gender:         0=Female  1=Male
age:            10–100
family_history: 0=No  1=Yes
fcvc:           1=Never  2=Sometimes  3=Always
ncp:            1–4 main meals/day
caec:           0=No  1=Sometimes  2=Frequently  3=Always
ch2o:           1=<1L  2=1–2L  3=>2L per day
faf:            0–3 active days/week
tue:            0=0–2h  1=3–5h  2=5h+ screen time
calc:           0=Never  1=Sometimes  2=Frequently  3=Always
bmi_bucket:     0=Under  1=Normal  2=Over  3=Obese
```

Clinical inputs (all optional):

```
waist_cm: 40–200    hip_cm: 40–200    arm_cm: 10–60
sys_bp: 60–250      dia_bp: 30–150
glucose: 40–700 (mg/dL)    insulin: 0.1–300 (μU/mL)
hba1c: 3.0–20 (%)          cholesterol: 50–600 (mg/dL)
exact_bmi: 10–80 (kg/m²)
```

---

## 16. Retraining

### Model 1

```bash
python model_pipeline.py
```

Steps performed:

1. Load `ObesityDataSet_raw_and_data_sinthetic.csv`.
2. Encode binary, frequency, and categorical features as documented above.
3. Drop leakage columns (`Height`, `Weight`, `BMI`) and low-importance columns.
4. Add the `Age × FCVC` interaction.
5. `RandomizedSearchCV` over 40 candidates under five-fold stratified CV, optimizing F1-macro.
6. Optimize the four confidence-score weights via grid search then Nelder-Mead, maximizing the correct/incorrect separation gap on OOF predictions.
7. Compute class centroids (in scaled feature space) and demographic profiles.
8. Save `obesity_model_bundle.joblib` and `obesity_rf_metadata.json`.

Restart `uvicorn` afterward; the bundle is loaded once at startup.

### Model 2

```bash
python resave_bundles.py
```

Steps performed:

1. Read each NHANES CSV in `merged_data/`.
2. Drop object-typed columns and the BMI label-construction columns.
3. RF importance screen for top-40 features.
4. Fit `SimpleImputer(median) → RandomForestClassifier(400 trees, depth 10)`.
5. Save the three bundles into `model2_bundles/`.

Restart `uvicorn` afterward.

---

## 17. Environment variables

| Variable | Default | Used by |
|---|---|---|
| `OBESITY_API_URL` | `http://localhost:8001` | Streamlit frontend |
| `MODEL2_BUNDLE_DIR` | `model2_bundles` | `model_pipeline.py` |
| `MILVUS_URI` | `http://localhost:19530` | `indexer.py` |
| `MILVUS_TOKEN` | empty | `indexer.py`, for Zilliz Cloud |
| `MILVUS_COLLECTION` | `papers_rag` | `indexer.py` |
| `GEMINI_API_KEY` | empty | `indexer.py`, Streamlit |
| `GOOGLE_API_KEY` | mirrored from `GEMINI_API_KEY` | LangChain Google integration |
| `ANTHROPIC_API_KEY` | empty | Streamlit Anthropic backend (prefills the key field); `indexer.py` direct Anthropic calls when no Gemini patch is active |
| `TRANSLATOR_URL` | `http://127.0.0.1:8080` | `indexer.py`, optional translation service (falls back to Gemini) |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` (set in code) | OpenMP workaround on macOS |
| `OMP_NUM_THREADS` | `4` (set in code) | Caps embedder thread count |

---

## 18. Troubleshooting

**`uvicorn` raises `FileNotFoundError: obesity_model_bundle.joblib`.**
You launched it from the wrong directory. The bundle path is relative; `cd app` before running.

**Streamlit shows "Connection refused" or "obesity API unreachable".**
The FastAPI server is not running, or `OBESITY_API_URL` does not match its address. Hit `http://localhost:8001/health` in a browser to confirm.

**`pymilvus` raises a connection error.**
Milvus is not up, or you are pointing at a remote URI without setting `MILVUS_TOKEN`. `docker ps | grep milvus` to verify locally; for Zilliz, double-check the token.

**Dimension mismatch when querying after switching embeddings.**
The collection was built with one dimension and you are querying with another. Either tick "Drop old collection" and re-index, or use a different `MILVUS_COLLECTION` name. The auto-reconnect logic checks `papers_rag_gemini` and `papers_rag_interactive` separately to make this less likely; if you used a custom name, the safest thing is to drop and rebuild.

**Gemini calls fail with a 401.**
The key is not in the environment for the running process. Setting it in the Streamlit UI patches the current process, but a freshly forked indexer worker may not see it. Easiest fix: `export GEMINI_API_KEY=...` before `streamlit run`.

**Anthropic mode says "Key required".**
The sidebar key field is empty. Export `ANTHROPIC_API_KEY` before launching or paste the key into the sidebar field. The input is `password`-typed and persists for the session only.

**Model 2 never triggers.**
By design. Model 2 only runs when Model 1 predicts OW_I or OW_II with low confidence or an OW runner-up. To force it for testing, feed inputs known to land in the OW boundary region (a 35-year-old with `bmi_bucket=2`, mixed signals on activity and diet).

**Confidence is "very low" but the prediction looks right.**
The composite score penalizes inputs that are far from the predicted class centroid in scaled feature space (the `x` term) or close to runner-up centroids (the `z` term). Both happen for inputs that are unusual relative to the training distribution. The score is doing its job; the prediction may still be correct, but the routing logic correctly flags it for follow-up.

**Stage 1 misclassifies but Stage 2 does not run.**
Stage 2 only fires for OW_I/OW_II boundary cases. Misclassifications outside that range (Insufficient Weight predicted as Normal, for example) are not eligible for escalation in the current router. The router targets the Overweight boundary only; other misclassifications are left to the confidence tier to flag.

**Apple Silicon: `OMP: Error #15` on import.**
The `KMP_DUPLICATE_LIB_OK=TRUE` workaround is set in code in both `indexer.py` and `obesity_app_v2.py`. If it still fires, ensure you did not `unset` it in your shell before launching.

**First chat query returns "no context retrieved".**
The Milvus collection is empty. Either you have not run Index PDFs yet, or the indexer connected to a different collection name than the chat. Check the Index PDFs tab for the collection it wrote to and make sure the chat is reading from the same one (the auto-reconnect code prefers `papers_rag_gemini` over `papers_rag_interactive`).

---

## 19. Limitations

- **Stage 1 training set is partially synthetic.** SMOTE augmentation introduces interpolated samples that may not represent naturally-occurring lifestyle combinations. Behavior on extreme input combinations may be governed by synthetic neighbours rather than real patient data.
- **Stage 2 fasting subsample halves N.** Including `LBXGLU` and `LBXIN` reduces the available NHANES population. An ablation comparing the full-feature model on the fasting subsample against a glucose-free model on the full phlebotomy sample is recommended before finalizing the feature set.
- **Cross-site generalizability is unquantified.** Stage 1 is trained on a single pooled dataset with no site-level stratification. External validation on an independent cohort (NHANES 2017–2020 or UK Biobank) is required before clinical deployment.
- **No prospective validation.** All evaluation is cross-validated on retrospective data. Real-world behavior on prospectively collected user inputs is unknown.
- **Diabetes risk module is rule-based, not validated independently.** It uses ADA 2024 thresholds directly, so it inherits the validity of those thresholds. It is not a diagnostic tool; the output explicitly directs users to clinical confirmation.
