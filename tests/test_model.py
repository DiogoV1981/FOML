import pytest
import pandas as pd
import mlflow


@pytest.fixture(scope="module")
def model() -> mlflow.pyfunc.PyFuncModel:
    mlflow.set_tracking_uri("./mlruns")
    model_name = "logistic_reg"
    model_version = 1
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )


def test_model_out(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'Pregnancies': 0,
        'Glucose': 30,
        'BloodPressure': 88,
        'SkinThickness': 60,
        'Insulin': 110,
        'BMI': 20.0,
        'DiabetesPedigreeFunction': 0.962,
        'Age': 20
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 0


def test_model_inv(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'Pregnancies': 0,
        'Glucose': 9999,
        'BloodPressure': 88,
        'SkinThickness': 60,
        'Insulin': 110,
        'BMI': 20.0,
        'DiabetesPedigreeFunction': 0.962,
        'Age': 20
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 1


def test_model_out_shape(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'Pregnancies': 0,
        'Glucose': 30,
        'BloodPressure': 88,
        'SkinThickness': 60,
        'Insulin': 110,
        'BMI': 20.0,
        'DiabetesPedigreeFunction': 0.962,
        'Age': 20
    }])
    prediction = model.predict(data=input)
    assert prediction.shape == (1, )