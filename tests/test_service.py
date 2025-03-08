import pytest
import requests

BASE_URL = "http://localhost:5003"  # Replace with your deployed server URL if different

def test_has_diabetes_prediction():
    """
    Test for the /has_diabetes endpoint with valid input data.
    It should return a prediction in the response.
    """
    response = requests.post(f"{BASE_URL}/has_diabetes", json={
        'Pregnancies': 0,
        'Glucose': 30,
        'BloodPressure': 88,
        'SkinThickness': 60,
        'Insulin': 110,
        'BMI': 20.0,
        'DiabetesPedigreeFunction': 0.962,
        'Age': 20
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], (int, float))
    assert response.json()["prediction"] == 0