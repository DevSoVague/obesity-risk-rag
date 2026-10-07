# NHANES 2021–2023 Model 2 Feature Engineering Analysis

**Project:** Two-stage BMI Classification System  
**Model 2 Task:** Binary classifier - OW_I (BMI 25–29.9) vs OW_II (BMI 30+)  
**Cycle:** August 2021 – August 2023  
**Source Files Reviewed:** 14 NHANES files across 8 planning documents

---

## 1. Overview and Model Architecture

Model 2 operates as a second-stage classifier triggered when Model 1 (the primary BMI-range predictor) returns a low-confidence output in the overweight/obese range. Its sole task is to distinguish **OW_I** (overweight, BMI 25–29.9) from **OW_II** (obese, BMI 30+).

The target label is constructed from the exam-measured variable `BMXBMI`:

```
y = 0 if 25 ≤ BMXBMI < 30  (OW_I)
y = 1 if BMXBMI ≥ 30       (OW_II)
```

`BMXBMI` is **dropped as a feature** after label construction to prevent data leakage. The analytic population is restricted to MEC-examined (`RIDSTATR=2`), non-pregnant adults aged 20+.

---

## 2. Sample Size and Subgroup Design Constraints

The most critical analytic decision is whether to include fasting biomarkers.

| Subsample | Key Variables | Approximate N |
|---|---|---|
| Full NHANES 2021–2023 | All | ~11,933 |
| MEC-examined adults (20+) | BMX, BP, balance | ~7,000+ |
| Phlebotomy subsample | HbA1c, total cholesterol | ~6,700+ |
| **Fasting subsample** | **Glucose, insulin** | **~3,361** |

Including fasting glucose (`LBXGLU`) or insulin (`LBXIN`) shrinks the usable sample to roughly half. The trade-off is significant: these are among the most powerful OW_I/OW_II discriminators, but at substantial N cost. An ablation comparing a full-variable model on the fasting subsample versus a glucose-free model on the phlebotomy sample is strongly recommended before finalizing the design.

---

## 3. Feature Groups and Decisions

### 3.1 Demographics (DEMO_L)

**Kept features:** Sex (`RIAGENDR`), age (`RIDAGEYR`), race/ethnicity (`RIDRETH3`, one-hot), nativity (`DMDBORN4`), education (`DMDEDUC2`, ordinal 1–5), marital status (`DMDMARTZ`, one-hot), and income-to-poverty ratio (`INDFMPIR`, continuous).

**Dropped:** Survey design variables (`SDMVSTRA`, `SDMVPSU`), survey weights as features (retain separately for weighted evaluation), pediatric-only variables (`RIDAGEMN`, `RIDEXAGM`, household reference person fields), and the redundant 5-category race variable `RIDRETH1` (superseded by the 6-category `RIDRETH3`).

**Maybe:** Military service (`DMQMILIZ`), exam season (`RIDEXMON`), household size (`DMDHHSIZ`), and years of US residency for foreign-born (`DMDYRUSR`).

Key note: `RIDEXPRG` (pregnancy status) should be used as an **exclusion criterion** rather than a feature. Pregnant participants should be removed from training because BMI during pregnancy does not reflect stable adiposity.

---

### 3.2 Body Measures and Blood Pressure (BMX_L, BPXO_L)

These are the closest physical proxies for the target and require the most careful leakage management.

**Kept - high priority:**
- `BMXWAIST` - waist circumference (cm): the single most clinically important feature for distinguishing OW_I vs OW_II beyond BMI. Captures central adiposity directly.
- `BMXHIP` - hip circumference (cm): used to compute the **waist-to-hip ratio (WHR = BMXWAIST / BMXHIP)**, a strong fat distribution discriminator.
- `BMXARMC` - mid-upper arm circumference (cm): not part of the BMI formula, no leakage risk.
- `BPAOCSZ` - BP cuff size (ordinal 0–3): a coarse proxy for arm size and obesity class; correlated with BMXARMC but worth including.

**Dropped - leakage risk:** `BMXWT` (weight), `BMXHT` (height), and all their QC flag columns are dropped because they directly enter the BMI calculation.

**Dropped - pediatric only:** Recumbent length, head circumference, and child BMI category.

**Engineered features:**
- `WHR = BMXWAIST / BMXHIP`
- `BPXOSY_avg = mean(BPXOSY1, BPXOSY2, BPXOSY3)` - average systolic BP
- `BPXODI_avg = mean(BPXODI1, BPXODI2, BPXODI3)` - average diastolic BP
- `BPXOPLS_avg = mean(BPXOPLS1, BPXOPLS2, BPXOPLS3)` - average pulse rate

The three BP readings are designated "maybe" - include as averaged composites rather than raw individual readings to reduce noise.

---

### 3.3 Laboratory Biomarkers

#### 3.3.1 Comprehensive Metabolic Panel (BIOPRO_L)

This file has **zero refusal or don't-know codes** - all variables are continuous lab measurements and all missingness becomes NaN automatically when reading the XPT file via pandas.

**Kept (11 features), all continuous:**

| Variable | Analyte | Clinical Relevance |
|---|---|---|
| `LBXSATSI` | ALT (IU/L) | Fatty liver / NAFLD - strongly associated with obesity severity. Log-transform. |
| `LBXSAL` | Albumin (g/dL) | Nutritional status; lower in severe obesity or malnutrition. |
| `LBXSASSI` | AST (IU/L) | Liver enzyme; AST/ALT ratio is a useful derived feature. Log-transform. |
| `LBXSBU` | BUN (mg/dL) | Kidney function; elevated in obesity-related dehydration and high protein intake. |
| `LBXSCR` | Creatinine (mg/dL) | Kidney function and muscle mass; use to compute eGFR. |
| `LBXSGB` | Globulin (g/dL) | Inflammatory proteins; elevated in chronic inflammation common in obesity. |
| `LBXSGTSI` | GGT (IU/L) | Highly sensitive liver enzyme; metabolic syndrome marker. Log-transform. |
| `LBXMAGN` | Magnesium (mg/dL) | Hypomagnesemia common in metabolic syndrome. |
| `LBXSTB` | Total bilirubin (mg/dL) | Inverse association with obesity. Log-transform; 132 values below LLOD already imputed. |
| `LBXSTP` | Total protein (g/dL) | Derived from albumin + globulin - consider dropping if both components are kept. |
| `LBXSUA` | Uric acid (mg/dL) | Strong, independent obesity/metabolic syndrome marker. |

**Maybe (13):** ALP, Bicarbonate, CPK, Chloride, Iron, LDH, Phosphorus, Potassium, Sodium, Calcium - plus fallback versions of Glucose, Cholesterol, and Triglycerides if their reference-method files are unavailable.

**Dropped (18):** All `LBDS*` SI-unit duplicates (exact linear transforms, add multicollinearity), Osmolality (derived from other features - perfect multicollinearity if included alongside them), and both comment code flags.

**Special warnings:**
- `LBXSLDSI` (LDH): ~34% missing - requires careful imputation or exclusion.
- `LBXSNASI` (Sodium): poor instrument bridging correlation (r=0.645) - caution for cross-cycle analyses.
- **Prefer reference-method files** for Glucose (`LBXGLU` from GLU_L), Cholesterol (`LBXTC` from TCHOL_L), and Triglycerides (`LBXTR` from TRIGLY_L) over BIOPRO_L counterparts.

**Engineered features suggested:**
- `AST_ALT_ratio = LBXSASSI / LBXSATSI` - ratio >2 suggests alcoholic liver disease
- `eGFR` via CKD-EPI equation using creatinine, age, and sex
- `BUN_Creatinine_ratio = LBXSBU / LBXSCR` - ratio >20 suggests prerenal causes

#### 3.3.2 Core Lab Panel (GHB_L, GLU_L, INS_L, TCHOL_L)

**Sample weight rules (critical):**
- Analyses using `LBXGLU` or `LBXIN`: use `WTSAF2YR`. Rows where `WTSAF2YR=0` must be excluded (no specimen or fasting criteria not met).
- Analyses using only `LBXGH` or `LBXTC`: use `WTPH2YR`. Rows where `WTPH2YR=0` must be excluded.
- Never mix weight variables in the same weighted analysis.

| Variable | Source | Clinical Relevance | Transform |
|---|---|---|---|
| `LBXGH` | GHB_L | HbA1c (%) - average blood glucose over 2–3 months; strong insulin resistance marker. | Consider log-transform if right-skewed. |
| `LBXGLU` | GLU_L | Fasting plasma glucose (mg/dL) - primary diabetes diagnosis marker. | Check skew; optional log. |
| `LBXIN` | INS_L | Fasting serum insulin (µU/mL) - hyperinsulinemia is a hallmark of OW_II. | **log1p required** (extreme right skew). |
| `LBXTC` | TCHOL_L | Total cholesterol (mg/dL) - cardiovascular risk marker; 14.6% missing. | Leave continuous. |

**Derived feature - HOMA-IR:**
```
HOMA_IR = (LBXGLU × LBXIN) / 405
```
This is the standard clinical composite for insulin resistance and is a stronger OW_I/OW_II discriminator than either glucose or insulin alone. Compute on the fasting subsample. Apply log1p after computing.

**Dropped:** All SI-unit duplicates (`LBDGLUSI`, `LBDINSI`, `LBDTCSI`), below-detection-limit flag (`LBDINLC`, only 1 record flagged - negligible).

**Missingness to watch:**
- `LBXTC`: 14.6% missing - the worst in this batch. Run an ablation to verify it earns its N cost before accepting the smaller sample.

---

### 3.4 Comorbidity Questionnaires

#### 3.4.1 Blood Pressure & Cholesterol (BPQ_L)

All four kept variables are strong obesity comorbidity signals:

| Variable | Meaning | Notes |
|---|---|---|
| `BPQ020` | Ever diagnosed with hypertension | Asked of all eligible (16+). Clean codes 7, 9 → NaN. |
| `BPQ150` | Currently on BP medication | Conditional on BPQ020=Yes. Impute 0 for confirmed non-hypertensives. |
| `BPQ080` | Ever told high cholesterol | Asked of all eligible. Clean codes 7, 9 → NaN (53 DKs - do not impute as No). |
| `BPQ101D` | Currently on cholesterol medication | Asked of **all** (not conditional). Clean code 9 → NaN. |

`BPQ030` (hypertension confirmed on 2+ visits) is a maybe - add only if feature importance warrants.

**Engineered composites suggested:**
- `any_cardiometabolic_dx` - any hypertension, high cholesterol, or diabetes/prediabetes diagnosis
- `on_any_cardiometabolic_med` - taking any BP, cholesterol, insulin, or oral diabetes medication
- `cardiometabolic_med_count` - count of medication categories active (0–4)

#### 3.4.2 Diabetes (DIQ_L)

`DIQ010` is the **most important variable in this entire questionnaire batch**. It captures diabetes status in three levels - No, Borderline/Prediabetes, and Yes - and should be encoded as an ordinal integer (No=0, Borderline=1, Yes=2) because the severity gradient is real and clinically meaningful. Collapsing to binary would lose important signal.

| Variable | Decision | Notes |
|---|---|---|
| `DIQ010` | Keep (ordinal: No=0, Borderline=1, Yes=2) | 4 DKs → NaN. Code 3 (Borderline) must not be collapsed. |
| `DIQ160` | Keep (binary) | Prediabetes flag for non-diabetics. Impute 0 for confirmed non-diabetics; impute 1 for DIQ010=3. |
| `DIQ050` | Keep (binary) | Currently taking insulin - strong metabolic severity signal. No refusal codes. |
| `DIQ070` | Keep (binary) | Currently on oral diabetes medication (e.g., metformin). |

**Dropped:** `DID040` (age at diabetes onset - 96% missing), `DID060`/`DIQ060U` (insulin duration - 97% missing), `DIQ180` (blood sugar test in borderline subgroup only), `DIQ159`/`DIQ065` (routing CHECK ITEMs containing no data).

The structural skip-pattern imputation is critical for this file: participants who were confirmed non-diabetic (DIQ010=2) and non-prediabetic (DIQ160=2) should have `DIQ050=0` and `DIQ070=0` imputed, not NaN.

#### 3.4.3 Health Status (HSQ_L)

**Entire file dropped.** The sole variable, `HSQ590` (HIV testing history), has no biological or clinical pathway to OW_I vs OW_II classification.

---

### 3.5 Medical Conditions (MCQ_L)

Twelve variables kept across cardiovascular, respiratory, and metabolic conditions:

**Cardiovascular (all binary, codes 7/9 → NaN):**
- `MCQ160b` - congestive heart failure
- `MCQ160c` - coronary heart disease
- `MCQ160d` - angina
- `MCQ160e` - heart attack (MI)
- `MCQ160f` - stroke

**Metabolic/other:**
- `MCQ160a` - arthritis (especially OA - prevalence substantially higher in OW_II due to weight-bearing joint stress)
- `MCQ160l` - any liver condition (NAFLD prevalence rises sharply with BMI)
- `MCQ510a` - fatty liver specifically (most directly tied to obesity and insulin resistance)
- `MCQ160m` - thyroid problem (hypothyroidism associated with weight gain)
- `MCQ160p` - COPD/emphysema/chronic bronchitis
- `MCQ010` - asthma (more prevalent and severe at higher obesity classes)
- `MCQ550` - gallstones (cholesterol supersaturation in bile is higher in OW_II)

**Dropped:** Deeply conditional variables with >94% structural missingness (cancer subtypes, insulin duration), pediatric-only items (MCQ149, MCQ157), and variables with near-zero variance (MCQ510b - liver fibrosis, 4 respondents only).

---

### 3.6 Kidney and Urological Function (KIQ_U_L)

Urinary incontinence indicators together form the **Incontinence Severity Index (ISI)** and are strong obesity-class markers:

| Variable | Meaning | Encoding |
|---|---|---|
| `KIQ022` | Ever told had weak/failing kidneys | Binary (1/0) |
| `KIQ005` | Urinary leakage frequency | Ordinal 0–4 |
| `KIQ010` | Volume per leakage episode | Ordinal 0–2 |
| `KIQ042` | Stress incontinence (leaked during physical activity) | Binary (1/0) |
| `KIQ044` | Urge incontinence (leaked before reaching toilet) | Binary (1/0) |
| `KIQ052` | Impact of leakage on daily activities | Ordinal 0–4 |
| `KIQ481` | Nocturia frequency (times waking per night) | Integer 0–5 |

Nocturia is associated with OSA and metabolic syndrome, both of which are more prevalent in OW_II. `KIQ025` (dialysis, 23 Yes responses) is dropped for near-zero variance.

---

### 3.7 Physical Activity (PAQ_L)

All six raw variables are kept, but the time-unit columns (`PAD790U`, `PAD810U`) are consumed during unit standardization and then dropped.

**Derived features:**
```python
# Convert frequency to weekly using unit column
moderate_ltpa_min_per_week = PAD790Q_weekly × PAD800
vigorous_ltpa_min_per_week = PAD810Q_weekly × PAD820
```

Sedentary time (`PAD680`, minutes/day sitting) is one of the strongest lifestyle predictors of obesity class. Winsorize values ≥1,080 min (18 hours) which trigger a CAPI soft-edit warning.

**NaN codes are 4-digit (7777/9999)** for this file - different from most other files.

---

### 3.8 Sleep (SLQ_L)

| Variable | Decision | Notes |
|---|---|---|
| `SLD012` | Keep continuous | Weekday sleep hours (derived by NHANES); range 2–14. |
| `SLD013` | Keep continuous | Weekend sleep hours (derived by NHANES). |
| `sleep_debt` (derived) | Compute | `SLD012 - SLD013` - negative = weekend catch-up; a metabolic marker. |
| Raw time strings (SLQ300/310/320/330) | Maybe | Use only to derive `social_jetlag = |weekend_midpoint - weekday_midpoint|`; 5-digit NaN codes (77777/99999). |

---

### 3.9 Smoking (SMQ_L)

Adult smoking status cannot be read from a single column - it requires combining two variables:

```
never_smoker    = (SMQ020 == 2)
former_smoker   = (SMQ020 == 1) AND (SMQ040 == 3)
current_somedays = (SMQ020 == 1) AND (SMQ040 == 2)
current_daily   = (SMQ020 == 1) AND (SMQ040 == 1)
```

Encode as three binary flags with `never_smoker` as the reference category (all three = 0).

`SMD650` (cigarettes per day, current smokers only) is kept as a continuous dose variable. Impute as 0 for never/former smokers or handle via a two-part model.

`SMD641` (days smoked in past 30) is a maybe - only 268 non-missing values across 9,015 records. Useful only if computing `pack_days_30 = SMD641 × SMD650`.

All youth-scoped variables (SMQ621, SMD630) are dropped since Model 2 targets adults (20+). SMAQUEX2 becomes constant after age filtering and is dropped.

---

### 3.10 Weight History (WHQ_L)

Lean file but directly on-task. Three derived features are planned:

| Derived Feature | Formula | Signal |
|---|---|---|
| `weight_change_1yr` | `WHD020 - WHD050` | OW_II participants more likely to have gained or plateaued at higher weight. Winsorize at ±50 lbs. |
| `self_reported_bmi` | `(WHD020 / WHD010²) × 703` | Cross-check against measured BMI. |
| `bmi_self_vs_measured_delta` | `self_reported_bmi - BMXBMI` | Under-reporters cluster in OW_II - systematic self-report bias differs by class. |

`WHQ070` (tried to lose weight in past 12 months) is a direct behavioral discriminator - weight-loss attempt rates are markedly higher in OW_II due to both health awareness and clinical recommendation pressure.

---

### 3.11 Alcohol Use (ALQ_L)

**Critical trap - the frequency scale is non-monotone.** Codes 1–10 do not run low-to-high: code 1 = "every day" (highest frequency) and code 10 = "1–2 times per year" (lowest). Skipping the reordering step would cause any tree or linear model to learn the wrong direction on these features.

Required reordering: `{0:0, 10:1, 9:2, 8:3, 7:4, 6:5, 5:6, 4:7, 3:8, 2:9, 1:10}`

**Structural missingness is the central challenge.** `ALQ111=No` (never drank) sends participants to end of section - their downstream missing values mean zero, not unknown. The preprocessing script handles this explicitly:

| Variable | Decision | Notes |
|---|---|---|
| `ALQ111` | Keep (binary) | Ever-drank gateway. Impute 0 for all downstream if ALQ111=0. |
| `ALQ121_freq` (reordered) | Keep (ordinal 0–10) | Past-year drinking frequency. |
| `ALQ130` | Keep (continuous 1–15) | Drinks per drinking day; 15 = ceiling-coded. |
| `ALQ142_freq` (reordered) | Keep (ordinal 0–10) | Binge drinking frequency. |
| `ALQ151` | Keep (binary) | Lifetime heavy drinking history. |
| `ALQ170` | Keep (continuous 0–30) | Past-30-day binge occasions; 30 = ceiling. |

`ALQ270` and `ALQ280` are maybe - both have 63% missingness and heavily overlap with `ALQ142`. Drop unless ablation shows added value.

**Composite feature:** `ALQ_weekly_drinks = (ALQ121_days_per_year / 52) × ALQ130` - a dose × frequency composite.

---

### 3.12 Balance and Falls (BAX_L / BAQ_L)

**Important caveat:** Balance impairment and falls are partly **downstream consequences of obesity**, not independent risk factors. OW_II individuals have greater mechanical joint load, reduced proprioception, and higher OSA/vestibular impairment rates - making these useful discriminators, but introducing potential reverse causality. Note this in the model card.

**BAX_L (balance exam - physical tests):** Conditions 1–5 of the Modified Romberg Test progress from easiest (firm floor, eyes open) to hardest (foam surface, eyes closed, head moving). Condition 4 (foam, eyes closed) has the highest failure rate (~33%) and is the most clinically meaningful for obesity-related balance impairment.

Kept features include trial-1 pass/fail and duration for all five conditions, plus trial-2 data for Conditions 4 and 5 where sample sizes are substantial. Composite features recommended:
- `COND4_PASS = 1 if (BAXPF41==1 OR BAXPF42==1)`
- `COND5_PASS = 1 if (BAXPF51==1 OR BAXPF52==1)` (NaN for ineligible participants)
- `COND4_MAX_DURATION = max(BAXTC41, BAXTC42)`

**BAQ_L (self-reported balance/falls questionnaire):**

| Variable | Decision | Notes |
|---|---|---|
| `BAQ321C` | Keep (binary) | Unsteadiness - stronger obesity connection than vertigo. |
| `BAQ321D` | Keep (binary) | Light-headedness/fainting - cardiovascular dysregulation increases with obesity. |
| `BAQ530` | Keep (ordinal 1–7) | Falls in past 5 years. |
| `BAQ550` | Keep (ordinal 1–5) | Falls in past 12 months. Impute 1 for BAQ530=1. |
| `BAQ401` | Maybe (binary) | Balance limited daily activity. Impute 0 for no-symptom participants. |
| `BAQ560` | Maybe (binary) | Injury from fall. Impute 0 for no-fall participants. |

`BAQ321A` (vertigo) is a maybe - weaker obesity specificity than unsteadiness. `BAQ321B` (vision blurring) is dropped.

---

## 4. Cross-File Derived Features

After merging all files on SEQN, these cross-file features are recommended:

| Feature | Formula | Rationale |
|---|---|---|
| `HOMA_IR` | `(LBXGLU × LBXIN) / 405` | Standard insulin resistance index - stronger OW_I/OW_II discriminator than either component alone. |
| `WHR` | `BMXWAIST / BMXHIP` | Abdominal fat distribution - powerful adiposity signal not captured by BMI. |
| `AST_ALT_ratio` | `LBXSASSI / LBXSATSI` | Liver disease pattern indicator. |
| `eGFR` | CKD-EPI (creatinine + age + sex) | More clinically meaningful than raw creatinine. |
| `bmi_self_vs_measured_delta` | `self_reported_bmi - BMXBMI` | Under-reporters cluster in OW_II - systematic bias differs by class. |
| `pack_days_30` | `SMD641 × SMD650` | Cigarette exposure composite (only if SMD641 missingness acceptable). |
| `weight_change_1yr` | `WHD020 - WHD050` | Weight trajectory signal. |
| `cardiometabolic_med_count` | Sum of 4 medication flags | Ordinal proxy for metabolic disease severity (0–4). |
| `BAQ_any_balance` | `OR(BAQ321A, BAQ321C, BAQ321D)` | Any balance symptom composite. |
| `ALQ_weekly_drinks` | `(ALQ121_days_yr / 52) × ALQ130` | Drinking dose-frequency composite. |
| `moderate_ltpa_min_per_week` | `PAD790Q_weekly × PAD800` | Total moderate activity volume. |
| `vigorous_ltpa_min_per_week` | `PAD810Q_weekly × PAD820` | Total vigorous activity volume. |
| `sleep_debt` | `SLD012 - SLD013` | Weekday-weekend sleep discrepancy. |

---

## 5. Global Preprocessing Rules

### 5.1 NHANES Refusal/Don't-Know Code Cleaning

All codes below must be replaced with `NaN` **before** any encoding or imputation:

| Code | Meaning | Context |
|---|---|---|
| 7, 9 | Refused / Don't know | Single-digit responses |
| 77, 99 | Refused / Don't know | Two-digit responses |
| 777, 999 | Refused / Don't know | Three-digit responses |
| 7777, 9999 | Refused / Don't know | Four-digit (PAQ, WHQ files) |
| 77777, 99999 | Refused / Don't know | Five-digit (SLQ time strings) |

BIOPRO_L is the one exception - it has no categorical survey codes.

### 5.2 Structural vs. Item Missingness

Structural missingness (participants skipped because of a valid prior answer) must be **imputed as a meaningful zero**, not treated as random missingness, before any downstream imputer runs. Getting this wrong would cause an imputer to fill median drinking quantities for non-drinkers, corrupting those features. Key cases:
- `ALQ111=No` → all downstream ALQ variables = 0
- `ALQ142=0` (no binge drinking) → `ALQ270`, `ALQ280`, `ALQ170` = 0
- `BPQ020=No` → `BPQ030`, `BPQ150` = 0
- `DIQ010=No` → `DIQ050`, `DIQ160` = 0, `DIQ070` = 0
- `BAQ530=1` (never fell in 5 years) → `BAQ550` = 1, `BAQ560` = 0
- `SMQ020=No` → `SMQ040` derivatives = never_smoker

### 5.3 Sample Filters (Apply in Order)

```python
df = df[df['RIDSTATR'] == 2]                               # MEC-examined only
df = df[df['RIDAGEYR'] >= 20]                              # Adults only
df = df[~((df['RIAGENDR'] == 2) & (df['RIDEXPRG'] == 1))] # Exclude pregnant women
df = df[df['BMDSTATS'] == 1]                               # Complete body measures exam
# For balance features: df = df[df['BAXMSTAT'].isin([1, 2])]
# For fasting features: df = df[df['WTSAF2YR'] > 0]
```

### 5.4 Transformation Recommendations

| Analyte | Transform |
|---|---|
| ALT, AST, GGT, Total Bilirubin | `log1p` - right-skewed liver enzymes |
| Fasting insulin (`LBXIN`) | `log1p` - extreme right skew |
| CPK | `log1p` after capping at 99th percentile |
| Triglycerides | `log1p` |
| HOMA-IR | `log1p` after computation |
| ALQ drinking variables | Ordinal reordering required (non-monotone scale) |

---

## 6. Key Analytical Warnings and Watch Points

1. **Fasting subsample N cost.** Including LBXGLU or LBXIN shrinks usable N to ~3,361. Decide this early - it affects every downstream analysis.

2. **ALQ scale trap.** The drinking frequency scale (ALQ121, ALQ142) is NOT monotone as stored. Code 1 = every day; code 10 = 1–2 times/year. Reorder before any modeling.

3. **BAQ is downstream of obesity.** Falls and balance impairment are partly caused by severe obesity - useful discriminators but with reverse causality risk. Flag in model card.

4. **Unit duplicates throughout BIOPRO_L.** All `LBDS*` columns are SI-unit versions of their `LBX*` counterparts. Drop all 10+ of them.

5. **LBXSOSSI (Osmolality) is derived.** It is computed from sodium, glucose, and BUN - including it alongside those three introduces perfect multicollinearity.

6. **Instrument changes in GHB_L and TCHOL_L.** Both files had mid-cycle equipment changes. Bridging studies confirmed no correction needed (r=0.999 and r=0.998 respectively). Use as-is.

7. **Total cholesterol missingness.** LBXTC has 14.6% missing - worst in the lab batch. Run an ablation before accepting the N cost.

8. **Smoking status requires combining two variables.** SMQ040 alone is meaningless without SMQ020. Construct the 4-level `smoking_status` variable explicitly.

9. **Self-report vs. measured BMI delta.** OW_II individuals systematically under-report their weight more than OW_I individuals. The delta `bmi_self_vs_measured_delta = self_reported_bmi - BMXBMI` may itself be a meaningful feature capturing this bias.

10. **Sodium instrument bridging.** `LBXSNASI` (sodium) has the poorest inter-instrument correlation of any BIOPRO_L analyte (r=0.645). Use with caution in cross-cycle analyses.

---

## 7. Final Feature Inventory Summary

| Domain | File(s) | Kept | Maybe | Dropped |
|---|---|---|---|---|
| Demographics | DEMO_L | 9 | 5 | 14 |
| Body measures | BMX_L | 4 | 2 | 12 |
| Blood pressure | BPXO_L | 1 | 9 | 1 |
| Balance exam | BAX_L | 16 | 5 | 18 |
| Bioprofile labs | BIOPRO_L | 11 | 13 | 18 |
| Core labs | GHB/GLU/INS/TCHOL | 4 | 0 | 5 |
| BP/Cholesterol Qx | BPQ_L | 4 | 1 | 1 |
| Diabetes Qx | DIQ_L | 4 | 1 | 5 |
| Health status Qx | HSQ_L | 0 | 0 | 2 |
| Kidney/Urology Qx | KIQ_U_L | 7 | 0 | 2 |
| Medical conditions | MCQ_L | 12 | 8 | 14 |
| Physical activity | PAQ_L | 4 | 0 | 2 |
| Sleep | SLQ_L | 2 | 4 | 2 |
| Smoking | SMQ_L | 3 | 1 | 4 |
| Weight history | WHQ_L | 4 | 1 | 0 |
| Alcohol | ALQ_L | 6 | 2 | 0 |
| Balance/falls Qx | BAQ_L | 4 | 4 | 8 |
| **Cross-file derived** | All | **13** | - | - |

---

*Analysis compiled from 8 variable planning documents covering 14 NHANES 2021–2023 source files.*
