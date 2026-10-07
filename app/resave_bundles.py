# resave_bundles.py
import joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
OB_ORDER    = ['OW_I', 'OW_II']

def prepare_XY(df):
    df = df.copy()
    y  = (df['OW_CLASS'] == 'OW_II').astype(int)
    X  = df.drop(columns=[c for c in ['OW_CLASS','BMXBMI'] if c in df.columns])
    X  = X.drop(columns=X.select_dtypes(include='object').columns.tolist())
    return X, y

def retrain(csv_path, bundle_path, label):
    df = pd.read_csv(csv_path)
    X, y = prepare_XY(df)

    # Use the same top-40 feature selection approach — quick RF screen
    from sklearn.impute import SimpleImputer as SI
    imp = SI(strategy='median')
    Xi  = imp.fit_transform(X)
    rf_screen = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf_screen.fit(Xi, y)
    top40 = pd.Series(rf_screen.feature_importances_, index=X.columns)\
              .sort_values(ascending=False).head(40).index.tolist()
    X = X[top40]

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf',     RandomForestClassifier(
                        n_estimators=400, max_depth=10,
                        min_samples_leaf=4, max_features=0.5,
                        random_state=RANDOM_SEED, n_jobs=-1)),
    ])
    pipe.fit(X, y)

    bundle = {
        'pipeline':      pipe,
        'feature_names': top40,
        'label_mapping': {'OW_I': 0, 'OW_II': 1},
        'ob_order':      OB_ORDER,
        'gender':        label,
    }
    joblib.dump(bundle, bundle_path)
    print(f'Saved {bundle_path}  ({len(top40)} features)')

Path('model2_bundles').mkdir(exist_ok=True)
retrain('merged_data/ow_fasting_clean.csv',        'model2_bundles/model2_pooled_bundle.joblib',  'pooled')
retrain('merged_data/ow_male_fasting_clean.csv',   'model2_bundles/model2_male_bundle.joblib',    'male')
retrain('merged_data/ow_female_fasting_clean.csv', 'model2_bundles/model2_female_bundle.joblib',  'female')
print('Done — restart uvicorn')