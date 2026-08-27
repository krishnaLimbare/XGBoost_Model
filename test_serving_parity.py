"""
Training / Serving Parity Tests
================================
These tests exist because of a real defect: app.py kept its own copy of the
feature-engineering code, it drifted from the training version (Age_Band ->
Age_Group, Cardio_Risk_Score -> Cardiovascular_Risk), and the column-alignment
step silently zero-filled the mismatched columns. The result was that every
served prediction dropped the model's second-largest coefficient, and about 4%
of users were shown the wrong risk tier.

Nothing here needs a running server. Run with:

    python -m pytest test_serving_parity.py -v

or, without pytest installed:

    python test_serving_parity.py
"""

import os
import pickle

import numpy as np
import pandas as pd

from src.diabetes_screening_model import (
    RAW_FEATURE_COLS,
    NUMERICAL_COLS,
    ONEHOT_COLS,
    build_model_matrix,
    predict_risk,
    engineer_features,
    encode_features,
    prepare_features,
    assign_risk_tier,
    calibration_report,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_screening_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'diabetes_screening_dataset.csv')

# Tolerance for float round-trips through pandas/numpy
ATOL = 1e-9


def load_package():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def sample_rows(n=500, seed=0):
    df = pd.read_csv(DATA_PATH)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# =============================================================================
# THE CORE PARITY TEST
# =============================================================================

def test_serving_matrix_matches_training_matrix():
    """
    build_model_matrix() must reproduce, exactly, the matrix the training
    script builds via engineer_features() + encode_features().

    This is the test that would have caught the Age_Band / Cardio_Risk_Score
    drift.
    """
    pkg = load_package()
    df = sample_rows(500)

    # --- training-side construction ---
    X, _ = prepare_features(df, target_col='diabetes')
    train_enc, feature_names = encode_features(X)
    for col in pkg['feature_names']:
        if col not in train_enc.columns:
            train_enc[col] = 0
    train_enc = train_enc[pkg['feature_names']].astype(float)
    train_enc[pkg['numerical_cols']] = pkg['scaler'].transform(
        train_enc[pkg['numerical_cols']]
    )

    # --- serving-side construction ---
    serve_enc = build_model_matrix(
        df, pkg['feature_names'], pkg['scaler'], pkg['numerical_cols']
    )

    assert list(serve_enc.columns) == list(train_enc.columns), \
        "Serving path produced different columns than the training path"

    diff = np.abs(serve_enc.values - train_enc.values).max()
    assert diff < ATOL, (
        f"Serving path diverges from training path (max abs diff {diff:.3e}). "
        f"Feature engineering has drifted."
    )


def test_no_feature_is_silently_zero_filled():
    """
    Every non-one-hot feature the model expects must actually be produced by
    the feature pipeline. A zero-filled engineered feature is the exact failure
    mode that caused Cardio_Risk_Score to be dropped at serving time.
    """
    pkg = load_package()
    df = sample_rows(200)

    engineered = engineer_features(df[RAW_FEATURE_COLS].copy())
    encoded = pd.get_dummies(engineered, columns=ONEHOT_COLS, drop_first=True, dtype=int)

    onehot_prefixes = tuple(f"{c}_" for c in ONEHOT_COLS)
    missing_non_onehot = [
        col for col in pkg['feature_names']
        if col not in encoded.columns and not col.startswith(onehot_prefixes)
    ]

    assert not missing_non_onehot, (
        f"These engineered features are expected by the model but not produced "
        f"by the pipeline, so they would be silently zero-filled: "
        f"{missing_non_onehot}"
    )


def test_build_model_matrix_raises_on_drift():
    """
    A renamed engineered feature must raise, not silently zero-fill.
    """
    pkg = load_package()
    df = sample_rows(10)

    bogus = list(pkg['feature_names']) + ['Cardiovascular_Risk']
    try:
        build_model_matrix(df, bogus, pkg['scaler'], pkg['numerical_cols'])
    except KeyError as e:
        assert 'diverged' in str(e) or 'Cardiovascular_Risk' in str(e)
    else:
        raise AssertionError(
            "build_model_matrix silently accepted an unknown feature name "
            "instead of raising - the zero-fill bug is back."
        )


def test_missing_raw_input_raises():
    """A missing required input must fail loudly, not predict on garbage."""
    pkg = load_package()
    df = sample_rows(10).drop(columns=['bmi'])
    try:
        build_model_matrix(df, pkg['feature_names'], pkg['scaler'], pkg['numerical_cols'])
    except KeyError as e:
        assert 'bmi' in str(e)
    else:
        raise AssertionError("Missing 'bmi' did not raise")


def test_extra_columns_are_ignored_safely():
    """
    Unused form fields (waist circumference etc.) must not affect the matrix.
    """
    pkg = load_package()
    df = sample_rows(100)

    base = build_model_matrix(df, pkg['feature_names'], pkg['scaler'], pkg['numerical_cols'])

    noisy = df.copy()
    noisy['waist_circumference_cm'] = 999.0
    noisy['sugary_drink_frequency'] = 7
    noisy['FamilyHistory'] = 1
    withextra = build_model_matrix(
        noisy, pkg['feature_names'], pkg['scaler'], pkg['numerical_cols']
    )

    assert np.abs(base.values - withextra.values).max() < ATOL, \
        "Untrained extra columns leaked into the model matrix"


# =============================================================================
# APP-LEVEL PARITY (exercises the real Flask route)
# =============================================================================

def test_flask_route_matches_direct_prediction():
    """
    The /predict endpoint must return the same probability as the shared
    pipeline for the same person.
    """
    import app as flask_app

    pkg = load_package()
    df = sample_rows(25, seed=7)

    with flask_app.app.test_client() as client:
        for _, row in df.iterrows():
            payload = {
                'gender': row['gender'],
                'age': float(row['age']),
                'bmi': float(row['bmi']),
                'hypertension': int(row['hypertension']),
                'heart_disease': int(row['heart_disease']),
                'diet': row['Diet'],
                'physical_activity': row['PhysicalActivity'],
            }
            resp = client.post('/predict', json=payload)
            assert resp.status_code == 200, resp.get_data(as_text=True)
            served = resp.get_json()['probability']

            expected = predict_risk(pd.DataFrame([{
                'gender': row['gender'],
                'age': float(row['age']),
                'bmi': float(row['bmi']),
                'hypertension': int(row['hypertension']),
                'heart_disease': int(row['heart_disease']),
                'Diet': row['Diet'],
                'PhysicalActivity': row['PhysicalActivity'],
            }]), pkg)[0]

            assert abs(served - expected) < 1e-9, (
                f"Flask route returned {served:.6f} but the shared pipeline "
                f"returned {expected:.6f}"
            )


def test_untrained_form_fields_are_reported_not_hidden():
    """
    The form collects fields the model cannot use. The API must disclose them
    rather than silently discarding them.
    """
    import app as flask_app

    payload = {
        'gender': 'Male', 'age': 50, 'bmi': 30,
        'hypertension': 0, 'heart_disease': 0,
        'diet': 'Unhealthy', 'physical_activity': 'Sedentary',
        'family_history': 1, 'waist_circumference': 105,
    }
    with flask_app.app.test_client() as client:
        body = client.post('/predict', json=payload).get_json()

    assert 'family_history' in body['ignored_inputs']
    assert 'waist_circumference' in body['ignored_inputs']
    assert body['disclaimer']


def test_untrained_fields_do_not_change_the_result():
    """
    Documents the current limitation explicitly: toggling family history cannot
    move the number, because the model was never trained on it.
    """
    import app as flask_app

    base = {
        'gender': 'Male', 'age': 50, 'bmi': 30,
        'hypertension': 0, 'heart_disease': 0,
        'diet': 'Unhealthy', 'physical_activity': 'Sedentary',
    }
    with flask_app.app.test_client() as client:
        a = client.post('/predict', json={**base, 'family_history': 0}).get_json()
        b = client.post('/predict', json={**base, 'family_history': 1}).get_json()

    assert a['probability'] == b['probability']


# =============================================================================
# CALIBRATION GUARDS
# =============================================================================

def test_probabilities_are_calibrated():
    """
    Mean predicted risk must track observed prevalence. This fails if anyone
    reintroduces class_weight='balanced' or scale_pos_weight, which inflate
    probabilities by roughly the inverse prevalence ratio.
    """
    pkg = load_package()
    df = sample_rows(20000, seed=1)

    proba = predict_risk(df, pkg)
    calib = calibration_report(df['diabetes'].values, proba)

    assert abs(calib['mean_predicted'] - calib['observed_prevalence']) < 0.03, (
        f"Predictions are miscalibrated: mean predicted "
        f"{calib['mean_predicted']:.3f} vs observed prevalence "
        f"{calib['observed_prevalence']:.3f}. Has class re-weighting been "
        f"reintroduced?"
    )
    assert calib['calibration_error'] < 0.05, \
        f"Calibration error too high: {calib['calibration_error']:.4f}"


def test_model_package_declares_calibration_and_threshold():
    """The deployed package must carry its operating point, not rely on 0.5."""
    pkg = load_package()
    assert pkg.get('calibrated') is True
    assert 0.0 < pkg['decision_threshold'] < 1.0
    assert pkg['risk_thresholds']['low'] < pkg['risk_thresholds']['high']
    assert pkg.get('data_disclaimer'), "Model package is missing its data disclaimer"


def test_risk_tiers_are_monotonic_in_outcome():
    """Higher tier must mean higher observed diabetes rate."""
    pkg = load_package()
    df = sample_rows(20000, seed=2)

    proba = predict_risk(df, pkg)
    tiers = [assign_risk_tier(p, pkg['risk_thresholds'])['tier'] for p in proba]
    frame = pd.DataFrame({'tier': tiers, 'actual': df['diabetes'].values})

    rates = {
        t: frame.loc[frame['tier'] == t, 'actual'].mean()
        for t in ('Low Risk', 'Medium Risk', 'High Risk')
        if (frame['tier'] == t).any()
    }
    ordered = [rates[t] for t in ('Low Risk', 'Medium Risk', 'High Risk') if t in rates]
    assert ordered == sorted(ordered), f"Risk tiers are not monotonic: {rates}"


# =============================================================================
# Plain-python runner (no pytest required)
# =============================================================================

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}\n        {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
