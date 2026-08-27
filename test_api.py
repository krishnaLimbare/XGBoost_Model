"""
Manual smoke test against a RUNNING server.

    python app.py            # in one terminal
    python test_api.py       # in another

For the automated suite that needs no server, use test_serving_parity.py.

NOTE: the earlier version of this script compared two payloads that differed
only in `family_history` - a field the model is not trained on - and therefore
always reported a 0.0% difference while claiming to measure "impact of
lifestyle changes". The cases below vary inputs the model actually uses.
"""

import json

import requests

URL = 'http://127.0.0.1:5000/predict'

# Identical age/BMI/comorbidities; only diet and physical activity differ.
# These ARE model inputs, so the difference is real.
HIGH_RISK_LIFESTYLE = {
    'gender': 'Male', 'age': 50, 'bmi': 30,
    'hypertension': 0, 'heart_disease': 0,
    'diet': 'Unhealthy', 'physical_activity': 'Sedentary',
}

LOW_RISK_LIFESTYLE = {
    'gender': 'Male', 'age': 50, 'bmi': 30,
    'hypertension': 0, 'heart_disease': 0,
    'diet': 'Healthy', 'physical_activity': 'Active',
}


def call(label, payload):
    response = requests.post(URL, json=payload, timeout=10)
    response.raise_for_status()
    body = response.json()

    print(f"\n--- {label} ---")
    print(f"  Tier:        {body['tier']}")
    print(f"  Probability: {body['probability']:.4f}")
    print(f"  Threshold:   {body['decision_threshold']:.4f}")
    print(f"  Calibrated:  {body['calibrated']}")
    if body.get('ignored_inputs'):
        print(f"  Ignored (not model inputs): {body['ignored_inputs']}")
    return body


def main():
    try:
        high = call('HIGH RISK LIFESTYLE', HIGH_RISK_LIFESTYLE)
        low = call('LOW RISK LIFESTYLE', LOW_RISK_LIFESTYLE)
    except requests.exceptions.ConnectionError:
        print("Could not reach the server. Start it with: python app.py")
        return
    except requests.exceptions.RequestException as exc:
        print(f"Request failed: {exc}")
        return

    delta = high['probability'] - low['probability']
    print(f"\nDiet + activity difference at the same age/BMI: "
          f"{delta * 100:+.1f} percentage points of absolute risk")
    print(f"  ({low['probability'] * 100:.1f}% -> "
          f"{high['probability'] * 100:.1f}%)")

    print(f"\n{high['disclaimer']}")


if __name__ == '__main__':
    main()
