import requests
import json

url = 'http://127.0.0.1:5000/predict'
headers = {'Content-Type': 'application/json'}

# Case 1: High Risk Lifestyle
data_high = {
    'gender': 'Male', 'age': 50, 'bmi': 30, 'hypertension': 0, 'heart_disease': 0,
    'family_history': 1, 'diet': 'Unhealthy', 'physical_activity': 'Sedentary'
}

# Case 2: Low Risk Lifestyle (Same Age/BMI)
data_low = {
    'gender': 'Male', 'age': 50, 'bmi': 30, 'hypertension': 0, 'heart_disease': 0,
    'family_history': 0, 'diet': 'Healthy', 'physical_activity': 'Active'
}

try:
    print("\n--- TEST 1: HIGH RISK LIFESTYLE ---")
    r1 = requests.post(url, json=data_high)
    res1 = r1.json()
    print(f"Full Response: {json.dumps(res1, indent=2)}")
    print(f"Result: {res1.get('result', 'N/A')}")
    print(f"Probability: {res1.get('probability', 'N/A')}")

    print("\n--- TEST 2: LOW RISK LIFESTYLE ---")
    r2 = requests.post(url, json=data_low)
    res2 = r2.json()
    print(f"Full Response: {json.dumps(res2, indent=2)}")
    print(f"Result: {res2['result']}")
    print(f"Probability: {res2['probability']:.4f}")
    
    diff = res1['probability'] - res2['probability']
    print(f"\n✅ Impact of Lifestyle Changes: {diff*100:.1f}% reduction in risk")

except Exception as e:
    print(f"Error: {e}")
