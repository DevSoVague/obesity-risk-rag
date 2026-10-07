"""
obesity_pipeline.py
====================
Run this file once to:
  1. Load + encode the dataset
  2. Train the final Random Forest on the lean 12-feature set
  3. Optimise confidence score weights (OOF)
  4. Save obesity_model_bundle.joblib  ← used by main.py (FastAPI)

Usage:
    python obesity_pipeline.py

Requirements:
    pip install pandas numpy scikit-learn scipy joblib fastapi uvicorn pydantic
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from itertools import product
from pathlib import Path
from scipy.optimize import minimize

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, cross_val_predict, train_test_split,
)
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "ObesityDataSet_raw_and_data_sinthetic.csv"
BUNDLE_PATH = "obesity_model_bundle.joblib"
META_PATH   = "obesity_rf_metadata.json"
RANDOM_SEED = 42

OB_ORDER = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I",  "Overweight_Level_II",
    "Obesity_Type_I",      "Obesity_Type_II",  "Obesity_Type_III",
]

def train():
    """Run full training pipeline. Call via: python model_pipeline.py"""

    # ── Step 1: Load & encode ─────────────────────────────────────────────────
    print("\n── Step 1: Load & encode ─────────────────────────────────────────────")

    df = pd.read_csv(DATA_PATH)
    df["NObeyesdad"] = pd.Categorical(df["NObeyesdad"], categories=OB_ORDER, ordered=True)
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    df_enc = df.copy()
    binary_map = {"yes": 1, "no": 0, "Male": 1, "Female": 0}
    for col in ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]:
        df_enc[col] = df_enc[col].map(binary_map)

    freq_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    df_enc["CAEC"] = df_enc["CAEC"].map(freq_map)
    df_enc["CALC"] = df_enc["CALC"].map(freq_map)
    df_enc = pd.get_dummies(df_enc, columns=["MTRANS"], drop_first=True)
    df_enc["BMI_bucket"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]
    ).astype(int)
    df_enc["NObeyesdad"] = df_enc["NObeyesdad"].cat.codes
    label_mapping = {name: int(code) for code, name in enumerate(OB_ORDER)}

    DROP = ["SMOKE", "SCC", "MTRANS_Bike", "MTRANS_Motorbike",
            "MTRANS_Walking", "MTRANS_Public_Transportation", "FAVC",
            "Height", "Weight", "BMI", "Weight_x_FAF", "BMI_x_FAF", "NObeyesdad"]
    df_enc["Age_x_FCVC"] = df_enc["Age"] * df_enc["FCVC"]
    y = df_enc["NObeyesdad"]
    X = df_enc.drop(columns=[c for c in DROP if c in df_enc.columns])
    FEATURE_NAMES = X.columns.tolist()
    print(f"Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
    print(f"Dataset shape: {X.shape}")

    # ── Step 2: Tune & train ──────────────────────────────────────────────────
    print("\n── Step 2: Hyperparameter tuning ─────────────────────────────────────")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    param_dist = {
        "n_estimators":     [200, 400, 600],
        "max_depth":        [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features":     ["sqrt", "log2", 0.5],
        "class_weight":     ["balanced", None],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        param_dist, n_iter=40, cv=cv,
        scoring="f1_macro", random_state=RANDOM_SEED, n_jobs=-1, verbose=1,
    )
    search.fit(X, y)
    print(f"Best CV F1-macro: {search.best_score_:.4f}")
    print(f"Best params:      {search.best_params_}")

    params = search.best_params_.copy()
    params["random_state"] = RANDOM_SEED
    params["n_jobs"] = -1
    model = RandomForestClassifier(**params)
    model.fit(X, y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    model_holdout = RandomForestClassifier(**params)
    model_holdout.fit(X_tr, y_tr)
    print(f"\nHoldout accuracy: {model_holdout.score(X_te, y_te):.4f}")
    print(classification_report(y_te, model_holdout.predict(X_te), target_names=OB_ORDER))

    # ── Step 3: Scaler + centroids + demographic profiles ─────────────────────
    print("\n── Step 3: Build scaler, centroids, demographic profiles ─────────────")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    centroids = {}
    demographic_profiles = {}
    for class_idx in range(7):
        mask = (y == class_idx).values
        centroids[class_idx] = X_scaled[mask].mean(axis=0)
        raw = X[mask]
        demographic_profiles[class_idx] = {
            "age_mean":      float(raw["Age"].mean()),
            "age_std":       float(raw["Age"].std()),
            "gender_rate":   float(raw["Gender"].mean()),
            "fam_hist_rate": float(raw["family_history_with_overweight"].mean()),
            "fcvc_mean":     float(raw["FCVC"].mean()),
        }
    print(f"Centroids built for {len(centroids)} classes ✓")

    # ── Step 4: Confidence weight optimisation (OOF) ──────────────────────────
    print("\n── Step 4: Optimise confidence weights (OOF) ─────────────────────────")
    cv5      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_prob = cross_val_predict(model, X, y, cv=cv5, method="predict_proba")
    oof_pred = oof_prob.argmax(axis=1)
    oof_ok   = (oof_pred == y.values).astype(int)
    print(f"OOF correct: {oof_ok.sum()} | OOF wrong: {(1 - oof_ok).sum()}")

    components = []
    for i in range(len(X)):
        row      = X.iloc[[i]]
        proba    = oof_prob[i]
        pred_idx = proba.argmax()
        top3     = proba.argsort()[::-1][:3]
        runner_up = [(top3[1], proba[top3[1]]), (top3[2], proba[top3[2]])]
        inp_scaled = scaler.transform(row)
        x = float(max(0, cosine_similarity(inp_scaled, centroids[pred_idx].reshape(1,-1))[0,0]))
        z_parts = []
        for ru_idx, ru_p in runner_up:
            cos_ru = float(max(0, cosine_similarity(inp_scaled, centroids[ru_idx].reshape(1,-1))[0,0]))
            z_parts.append(ru_p * cos_ru)
        z = float(np.mean(z_parts))
        inp = row.iloc[0]
        profile = demographic_profiles[pred_idx]
        age_z = abs(inp["Age"] - profile["age_mean"]) / (profile["age_std"] + 1e-6)
        age_m = max(0.0, 1 - age_z / 3)
        g_m = (inp["Gender"] * profile["gender_rate"] +
               (1 - inp["Gender"]) * (1 - profile["gender_rate"]))
        f_m = float(inp["family_history_with_overweight"] == round(profile["fam_hist_rate"]))
        m = 0.4 * age_m + 0.4 * g_m + 0.2 * f_m
        components.append({"x": x, "y": float(proba[pred_idx]), "z": z, "m": m})

    comp_df = pd.DataFrame(components)
    correct = oof_ok

    def _sw(w, df):
        a, b, c, d = w
        return np.clip(a*df["x"].values + b*df["y"].values - c*df["z"].values + d*df["m"].values, 0, 1)

    def _obj(w, df, ok, alpha=0.7, beta=0.3):
        if any(v < 0 for v in w): return 999.0
        s = _sw(w, df); sc = s[ok==1]; sw_ = s[ok==0]
        if len(sc)==0 or len(sw_)==0: return 999.0
        sep = sc.mean() - sw_.mean()
        if np.isnan(sep): return 999.0
        return float(-alpha*sep - beta*(sc>0.7).mean() + (1-alpha-beta)*(sw_>0.5).mean())

    best_loss, best_w = 999.0, None
    for a, b, c, d in product(np.arange(0.10,0.55,0.05), np.arange(0.20,0.65,0.05),
                               np.arange(0.05,0.35,0.05), np.arange(0.05,0.30,0.05)):
        loss = _obj((a,b,c,d), comp_df, correct)
        if loss < best_loss:
            best_loss, best_w = loss, (a,b,c,d)
    print(f"Grid best: a={best_w[0]:.2f} b={best_w[1]:.2f} c={best_w[2]:.2f} d={best_w[3]:.2f}  loss={best_loss:.4f}")

    opt = minimize(_obj, x0=list(best_w), args=(comp_df, correct),
                   method="Nelder-Mead", options={"xatol":1e-5,"fatol":1e-5,"maxiter":50000})
    opt_w = tuple(np.clip(opt.x, 0.01, 0.99))
    print(f"Optimized: a={opt_w[0]:.4f} b={opt_w[1]:.4f} c={opt_w[2]:.4f} d={opt_w[3]:.4f}  loss={opt.fun:.4f}")

    candidates = {"default":(0.35,0.40,0.15,0.10), "grid":best_w, "optimized":opt_w}
    gaps = {k: (_sw(w,comp_df)[correct==1].mean() - _sw(w,comp_df)[correct==0].mean())
            for k,w in candidates.items()}
    FINAL_WEIGHTS = candidates[max(gaps, key=gaps.get)]
    print(f"\n✅ Final weights ({max(gaps, key=gaps.get)}): {tuple(round(float(v),4) for v in FINAL_WEIGHTS)}")
    print(f"   Separation gap: {gaps[max(gaps, key=gaps.get)]:.4f}")

    # ── Step 5: Save bundle ───────────────────────────────────────────────────
    print("\n── Step 5: Save bundle ───────────────────────────────────────────────")
    bundle = {
        "model": model, "scaler": scaler, "centroids": centroids,
        "demographic_profiles": demographic_profiles,
        "final_weights": FINAL_WEIGHTS, "feature_names": FEATURE_NAMES,
        "ob_order": OB_ORDER, "label_mapping": label_mapping,
    }
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"Bundle saved → {BUNDLE_PATH}")

    clean_params = {k: (int(v) if hasattr(v,"item") else v)
                    for k,v in params.items() if v is not None}
    metadata = {
        "model": "RandomForestClassifier", "params": clean_params,
        "features": FEATURE_NAMES, "n_features": len(FEATURE_NAMES),
        "target_classes": OB_ORDER, "label_mapping": label_mapping,
        "cv_f1_macro": round(float(search.best_score_), 4),
        "confidence_weights": {
            "a_cosine_sim":     round(float(FINAL_WEIGHTS[0]),4),
            "b_model_proba":    round(float(FINAL_WEIGHTS[1]),4),
            "c_confusion_risk": round(float(FINAL_WEIGHTS[2]),4),
            "d_demographic":    round(float(FINAL_WEIGHTS[3]),4),
            "formula": "score = a*x + b*y - c*z + d*m",
        },
        "notes": [
            "BMI_bucket: 0=underweight(<18.5) 1=normal(18.5-25) 2=overweight(25-30) 3=obese(>30)",
            "Age_x_FCVC interaction feature included",
            "MTRANS, FAVC, SMOKE, SCC, Height, Weight, BMI dropped",
        ],
    }
    Path(META_PATH).write_text(json.dumps(metadata, indent=2))
    print(f"Metadata saved → {META_PATH}")
    print("\nDone. Run:  uvicorn main:app --reload")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPER (imported by main.py)
# ══════════════════════════════════════════════════════════════════════════════
def _load_bundle(path: str = BUNDLE_PATH):
    return joblib.load(path)


def compute_confidence_for_class(
    class_idx: int,
    proba: np.ndarray,
    input_scaled: np.ndarray,
    inp_row,
    centroids: dict,
    demographic_profiles: dict,
    final_weights: tuple,
) -> float:
    """Compute composite confidence score treating class_idx as the prediction."""
    a, b, c, d = final_weights

    top3      = proba.argsort()[::-1][:3]
    runner_up = [(i, proba[i]) for i in top3 if i != class_idx][:2]

    # x
    x = float(max(0, cosine_similarity(
        input_scaled, centroids[class_idx].reshape(1, -1))[0, 0]))

    # z
    z_parts = []
    for ru_idx, ru_p in runner_up:
        cos_ru = float(max(0, cosine_similarity(
            input_scaled, centroids[ru_idx].reshape(1, -1))[0, 0]))
        z_parts.append(ru_p * cos_ru)
    z = float(np.mean(z_parts)) if z_parts else 0.0

    # m
    profile = demographic_profiles[class_idx]
    age_z   = abs(inp_row["Age"] - profile["age_mean"]) / (profile["age_std"] + 1e-6)
    age_m   = max(0.0, 1 - age_z / 3)
    g_m     = (inp_row["Gender"] * profile["gender_rate"] +
               (1 - inp_row["Gender"]) * (1 - profile["gender_rate"]))
    f_m     = float(inp_row["family_history_with_overweight"] == round(profile["fam_hist_rate"]))
    m       = 0.4 * age_m + 0.4 * g_m + 0.2 * f_m

    y = float(proba[class_idx])
    return float(np.clip(a * x + b * y - c * z + d * m, 0.0, 1.0))


def confidence_tier(score: float) -> str:
    if score >= 0.85:   return "High confidence"
    if score >= 0.70:   return "Moderate confidence — consider follow-up"
    if score >= 0.55:   return "Low confidence — borderline case"
    return "Very low confidence — inconclusive"


def run_inference(
    gender, age, family_history, fcvc, ncp,
    caec, ch2o, faf, tue, calc, bmi_bucket,
    bundle: dict,
) -> dict:
    model_              = bundle["model"]
    scaler_             = bundle["scaler"]
    centroids_          = bundle["centroids"]
    demographic_profiles_ = bundle["demographic_profiles"]
    final_weights_      = bundle["final_weights"]
    feature_names_      = bundle["feature_names"]
    ob_order_           = bundle["ob_order"]

    features = pd.DataFrame([{
        "Gender":                           gender,
        "Age":                              age,
        "family_history_with_overweight":   family_history,
        "FCVC":                             fcvc,
        "NCP":                              ncp,
        "CAEC":                             caec,
        "CH2O":                             ch2o,
        "FAF":                              faf,
        "TUE":                              tue,
        "CALC":                             calc,
        "Age_x_FCVC":                       age * fcvc,
        "BMI_bucket":                       bmi_bucket,
    }])[feature_names_]

    proba        = model_.predict_proba(features)[0]
    pred_idx     = int(proba.argmax())
    input_scaled = scaler_.transform(features)
    inp_row      = features.iloc[0]

    # Top-3 with individual confidence scores
    top3_idx = proba.argsort()[::-1][:3]
    top3 = []
    for rank, idx in enumerate(top3_idx):
        conf = compute_confidence_for_class(
            idx, proba, input_scaled, inp_row,
            centroids_, demographic_profiles_, final_weights_,
        )
        top3.append({
            "rank":        rank + 1,
            "class":       ob_order_[idx],
            "probability": round(float(proba[idx]), 4),
            "confidence":  round(conf, 4),
            "confidence_pct": round(conf * 100, 1),
            "tier":        confidence_tier(conf),
        })

    # Primary prediction (rank 1)
    primary = top3[0]

    return {
        # Primary result
        "predicted_class":   primary["class"],
        "confidence":        primary["confidence"],
        "confidence_pct":    primary["confidence_pct"],
        "tier":              primary["tier"],
        "close_call":        top3[1]["probability"] > 0.20 if len(top3) > 1 else False,

        # Top-3 (key addition)
        "top3":              top3,

        # Runner-ups (for backwards compat)
        "runner_up": [
            {"class": t["class"], "probability": t["probability"]}
            for t in top3[1:]
        ],

        # Full distribution
        "all_probabilities": {
            ob_order_[i]: round(float(p), 4) for i, p in enumerate(proba)
        },

        # Score components for primary
        "score_components": {
            "x_cosine_sim":     round(float(max(0, cosine_similarity(
                input_scaled, centroids_[pred_idx].reshape(1, -1))[0, 0])), 4),
            "y_model_proba":    round(float(proba[pred_idx]), 4),
            "z_confusion_risk": round(float(np.mean([
                float(max(0, cosine_similarity(
                    input_scaled, centroids_[i].reshape(1, -1))[0, 0])) * proba[i]
                for i in proba.argsort()[::-1][1:3]
            ])), 4),
            "m_demographic":    round(
                compute_confidence_for_class(
                    pred_idx, proba, input_scaled, inp_row,
                    centroids_, demographic_profiles_, (0, 0, 0, 1),  # m only
                ), 4),
        },
    }


if __name__ == "__main__":
    # Run training pipeline, then self-test
    train()

    print("\n── Self-test ─────────────────────────────────────────────────────────")
    b = _load_bundle()
    result = run_inference(
        gender=1, age=25, family_history=1,
        fcvc=2, ncp=3, caec=1, ch2o=2,
        faf=1, tue=1, calc=1, bmi_bucket=2,
        bundle=b,
    )
    print(f"Predicted: {result['predicted_class']}  ({result['confidence_pct']}%  {result['tier']})")
    print("Top 3:")
    for t in result["top3"]:
        print(f"  {t['rank']}. {t['class']:25s}  prob={t['probability']:.3f}  conf={t['confidence_pct']:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — NHANES ESCALATION (OW_I vs OW_II refinement)
# ══════════════════════════════════════════════════════════════════════════════
import os

MODEL2_BUNDLE_DIR = os.environ.get("MODEL2_BUNDLE_DIR", "model2_bundles")

# ── Clinical input feature names (must match NHANES training columns) ─────────
# These are the features Model 2 was trained on that a user can self-report.
# All others default to NaN — the RF pipeline handles them via median imputation.
M2_USER_FEATURES = {
    # Body measurements (collected by form)
    "BMXWAIST":        "Waist circumference (cm)",
    "BMXHIP":          "Hip circumference (cm)",
    "BMXARMC":         "Mid-upper arm circumference (cm)",
    # Blood pressure (from a physical)
    "SYS_BP":          "Systolic blood pressure (mmHg) — average of readings",
    "DIA_BP":          "Diastolic blood pressure (mmHg) — average of readings",
    # Lab values (from a blood panel)
    "LBXGLU":          "Fasting plasma glucose (mg/dL)",
    "LBXIN":           "Fasting insulin (µU/mL)",
    "LBXGH":           "HbA1c (%)",
    "LBXTC":           "Total cholesterol (mg/dL)",
    # Passed through from Model 1
    "RIDAGEYR":        "Age (years)",
    "RIAGENDR":        "Gender (0=male, 1=female)",
}


def _load_model2_bundle(gender: int):
    """Load the appropriate Model 2 bundle based on gender. Falls back to pooled."""
    bundle_dir = Path(MODEL2_BUNDLE_DIR)
    gender_file = "model2_male_bundle.joblib" if gender == 0 else "model2_female_bundle.joblib"
    pooled_file = "model2_pooled_bundle.joblib"

    path = bundle_dir / gender_file
    if not path.exists():
        path = bundle_dir / pooled_file
    if not path.exists():
        return None
    return joblib.load(path)


def should_trigger_model2(model1_result: dict) -> bool:
    """
    Router: decide whether to escalate to Model 2.

    Triggers if Model 1 predicted OW_I or OW_II AND either:
      (a) confidence < 75%, or
      (b) the runner-up is the other OW class (close_call between the two)
    """
    OW_CLASSES = {"Overweight_Level_I", "Overweight_Level_II"}
    top3 = model1_result.get("top3", [])
    if not top3:
        return False
    primary = top3[0]["class"]
    if primary not in OW_CLASSES:
        return False
    confidence_low   = top3[0]["confidence_pct"] < 75.0
    adjacent_runner  = len(top3) > 1 and top3[1]["class"] in OW_CLASSES
    return confidence_low or adjacent_runner


def _build_model2_input(
    gender, age,
    waist_cm=None, hip_cm=None, arm_cm=None,
    sys_bp=None, dia_bp=None,
    glucose=None, insulin=None, hba1c=None, cholesterol=None,
    feature_names=None,
):
    """
    Build the full feature row for Model 2. Known values are set; everything
    else is NaN and handled by the pipeline's SimpleImputer.
    """
    row = {f: np.nan for f in (feature_names or [])}

    # Derived features that can be computed from inputs
    waist_ht_ratio  = None
    waist_hip_ratio = None
    homa_ir         = None
    homa_ir_log     = None
    bmi_self_delta  = None  # not available without measured BMI
    lbxin_log       = None
    alt_log         = None

    if insulin is not None:
        lbxin_log = float(np.log1p(insulin))
    if glucose is not None and insulin is not None:
        homa_ir     = (glucose * insulin) / 405.0
        homa_ir_log = float(np.log1p(homa_ir))

    # Map known values into row
    direct_map = {
        "RIDAGEYR":        age,
        "RIAGENDR":        gender,
        "BMXWAIST":        waist_cm,
        "BMXHIP":          hip_cm,
        "BMXARMC":         arm_cm,
        "SYS_BP":          sys_bp,
        "DIA_BP":          dia_bp,
        "LBXGLU":          glucose,
        "LBXIN":           insulin,
        "LBXIN_log":       lbxin_log,
        "LBXGH":           hba1c,
        "LBXTC":           cholesterol,
        "HOMA_IR":         homa_ir,
        "HOMA_IR_log":     homa_ir_log,
    }
    if waist_cm and hip_cm:
        direct_map["WAIST_HIP_RATIO"] = waist_cm / hip_cm
    if waist_cm and arm_cm:
        # Approximate height from arm circumference is not reliable;
        # WAIST_HT_RATIO requires height which comes from Model 1 inputs
        pass

    for col, val in direct_map.items():
        if col in row and val is not None:
            row[col] = float(val)

    return pd.DataFrame([row])[feature_names] if feature_names else pd.DataFrame([row])


def _model2_confidence_score(proba_ow2: float, n_inputs_provided: int,
                              total_inputs: int = 10) -> dict:
    """
    Composite confidence for Model 2.

    Components:
      p  = model probability for predicted class
      c  = input completeness ratio (provided / total clinical inputs)
      Combined: confidence = 0.85*p + 0.15*c
    """
    p = float(proba_ow2)
    c = min(1.0, n_inputs_provided / total_inputs)
    score = float(np.clip(0.85 * p + 0.15 * c, 0.0, 1.0))

    if score >= 0.85:
        tier = "High confidence"
    elif score >= 0.70:
        tier = "Moderate confidence — consider follow-up"
    elif score >= 0.55:
        tier = "Low confidence — borderline case"
    else:
        tier = "Very low confidence — inconclusive"

    return {
        "score":          round(score, 4),
        "score_pct":      round(score * 100, 1),
        "tier":           tier,
        "p_model_proba":  round(p, 4),
        "c_completeness": round(c, 4),
        "inputs_provided": n_inputs_provided,
        "inputs_total":    total_inputs,
    }


def run_model2_inference(
    gender, age,
    waist_cm=None, hip_cm=None, arm_cm=None,
    sys_bp=None, dia_bp=None,
    glucose=None, insulin=None, hba1c=None, cholesterol=None,
) -> dict:
    """
    Run Model 2 inference. Returns refined OW_I/OW_II classification.
    Returns None if bundles are not available.
    """
    bundle = _load_model2_bundle(gender)
    if bundle is None:
        return None

    pipe          = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    ob_order      = bundle["ob_order"]      # ['OW_I', 'OW_II']

    X = _build_model2_input(
        gender=gender, age=age,
        waist_cm=waist_cm, hip_cm=hip_cm, arm_cm=arm_cm,
        sys_bp=sys_bp, dia_bp=dia_bp,
        glucose=glucose, insulin=insulin,
        hba1c=hba1c, cholesterol=cholesterol,
        feature_names=feature_names,
    )

    proba     = pipe.predict_proba(X)[0]   # [P(OW_I), P(OW_II)]
    pred_idx  = int(proba.argmax())
    pred_cls  = ob_order[pred_idx]

    # Count provided clinical inputs
    clinical_inputs = [waist_cm, hip_cm, arm_cm, sys_bp, dia_bp,
                       glucose, insulin, hba1c, cholesterol]
    n_provided = sum(1 for v in clinical_inputs if v is not None)

    conf = _model2_confidence_score(proba[pred_idx], n_provided)

    return {
        "model":          "model2_nhanes",
        "gender_model":   "male" if gender == 0 else "female",
        "predicted_class": pred_cls,
        "confidence_pct": conf["score_pct"],
        "tier":           conf["tier"],
        "probabilities":  {
            "OW_I":  round(float(proba[0]), 4),
            "OW_II": round(float(proba[1]), 4),
        },
        "confidence_components": conf,
        "inputs_used": {
            "waist_cm":    waist_cm,
            "hip_cm":      hip_cm,
            "arm_cm":      arm_cm,
            "sys_bp":      sys_bp,
            "dia_bp":      dia_bp,
            "glucose":     glucose,
            "insulin":     insulin,
            "hba1c":       hba1c,
            "cholesterol": cholesterol,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# DIABETES RISK ASSESSMENT (Model 3 — rule-based + clinical thresholds)
# ══════════════════════════════════════════════════════════════════════════════
#
# No separate model needed here — clinical thresholds are well-established
# (ADA 2024 guidelines). We compute a risk score and classification from
# the same clinical inputs collected for Model 2.
#
# Outputs:
#   - diabetes_risk: "Low" / "Prediabetes" / "Type 2 Diabetes" / "Possible Type 1"
#   - risk_score: 0.0–1.0 composite
#   - driving_factors: list of what pushed the risk up
# ─────────────────────────────────────────────────────────────────────────────

def assess_diabetes_risk(
    glucose=None, hba1c=None, insulin=None,
    age=None, bmi=None, family_history=None,
) -> dict:
    """
    ADA 2024 guideline-based diabetes risk assessment.

    Returns risk classification, numeric score, and driving factors.
    Returns None if insufficient inputs (need at least glucose OR hba1c).
    """
    if glucose is None and hba1c is None:
        return None

    factors = []
    risk_points = 0.0

    # ── Fasting glucose (ADA thresholds) ─────────────────────────────────────
    glucose_category = None
    if glucose is not None:
        if glucose >= 126:
            glucose_category = "diabetic"
            risk_points += 3.0
            factors.append(f"Fasting glucose {glucose} mg/dL — diabetic range (≥126)")
        elif glucose >= 100:
            glucose_category = "prediabetes"
            risk_points += 1.5
            factors.append(f"Fasting glucose {glucose} mg/dL — prediabetes range (100–125)")
        else:
            glucose_category = "normal"
            factors.append(f"Fasting glucose {glucose} mg/dL — normal (<100)")

    # ── HbA1c (ADA thresholds) ────────────────────────────────────────────────
    hba1c_category = None
    if hba1c is not None:
        if hba1c >= 6.5:
            hba1c_category = "diabetic"
            risk_points += 3.0
            factors.append(f"HbA1c {hba1c}% — diabetic range (≥6.5%)")
        elif hba1c >= 5.7:
            hba1c_category = "prediabetes"
            risk_points += 1.5
            factors.append(f"HbA1c {hba1c}% — prediabetes range (5.7–6.4%)")
        else:
            hba1c_category = "normal"
            factors.append(f"HbA1c {hba1c}% — normal (<5.7%)")

    # ── HOMA-IR (insulin resistance) ──────────────────────────────────────────
    homa_ir = None
    if glucose is not None and insulin is not None:
        homa_ir = (glucose * insulin) / 405.0
        if homa_ir > 5.0:
            risk_points += 1.5
            factors.append(f"HOMA-IR {homa_ir:.2f} — severe insulin resistance (>5.0)")
        elif homa_ir > 2.5:
            risk_points += 0.75
            factors.append(f"HOMA-IR {homa_ir:.2f} — insulin resistance (>2.5)")

    # ── Risk factors ──────────────────────────────────────────────────────────
    if bmi is not None and bmi >= 30:
        risk_points += 0.5
        factors.append(f"BMI {bmi:.1f} — obesity increases T2D risk")
    if age is not None and age >= 45:
        risk_points += 0.3
        factors.append(f"Age {age} — risk increases after 45")
    if family_history == 1:
        risk_points += 0.5
        factors.append("Family history of overweight/diabetes")

    # ── Classify ──────────────────────────────────────────────────────────────
    # Determine primary classification
    high_glucose  = glucose_category  == "diabetic"
    high_hba1c    = hba1c_category    == "diabetic"
    pre_glucose   = glucose_category  == "prediabetes"
    pre_hba1c     = hba1c_category    == "prediabetes"

    # Possible Type 1 flag: low insulin + young age + diabetic glucose/HbA1c
    possible_type1 = False
    if (high_glucose or high_hba1c) and insulin is not None and age is not None:
        if insulin < 5.0 and age < 40:
            possible_type1 = True

    if possible_type1:
        risk_class = "Possible Type 1 Diabetes — consult a physician"
        risk_note  = ("Low insulin with diabetic-range glucose in a younger adult may indicate "
                      "Type 1 or LADA (Latent Autoimmune Diabetes in Adults). "
                      "This requires clinical confirmation — do not self-diagnose.")
    elif high_glucose or high_hba1c:
        risk_class = "Type 2 Diabetes — clinical confirmation recommended"
        risk_note  = ("Values are in the diabetic range per ADA 2024 guidelines. "
                      "A formal diagnosis requires confirmation on a separate day "
                      "or a second test. Consult a healthcare provider.")
    elif pre_glucose or pre_hba1c:
        risk_class = "Prediabetes — lifestyle intervention recommended"
        risk_note  = ("Values indicate prediabetes. The ADA recommends 5–7% weight loss, "
                      "150 min/week of moderate activity, and dietary changes. "
                      "Reassess in 3–6 months.")
    else:
        risk_class = "Low diabetes risk based on provided values"
        risk_note  = "Current glucose and HbA1c values are within normal ranges."

    # Normalise risk score 0–1
    risk_score = float(np.clip(risk_points / 6.0, 0.0, 1.0))

    return {
        "risk_classification": risk_class,
        "risk_score":          round(risk_score, 3),
        "risk_score_pct":      round(risk_score * 100, 1),
        "note":                risk_note,
        "driving_factors":     factors,
        "values_used": {
            "glucose":     glucose,
            "hba1c":       hba1c,
            "insulin":     insulin,
            "homa_ir":     round(homa_ir, 3) if homa_ir is not None else None,
            "age":         age,
            "bmi":         bmi,
            "family_history": family_history,
        },
        "thresholds_reference": {
            "glucose_normal":      "< 100 mg/dL",
            "glucose_prediabetes": "100–125 mg/dL",
            "glucose_diabetes":    "≥ 126 mg/dL",
            "hba1c_normal":        "< 5.7%",
            "hba1c_prediabetes":   "5.7–6.4%",
            "hba1c_diabetes":      "≥ 6.5%",
            "homa_ir_normal":      "< 2.5",
            "homa_ir_resistant":   "> 2.5",
        },
    }
