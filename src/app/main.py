import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles# acrescentado para correr.
import mlflow
from pydantic import BaseModel, conint
import pandas as pd
import json
import uvicorn


# Define the inputs expected in the request body as JSON
class Request(BaseModel):
    """
    Request model for the API, defining the input structure.

    Attributes:
        Pregnancies (int): Number of pregnancies.
        Glucose (int): Plasma glucose concentration.
        BloodPressure (int): Diastolic blood pressure.
        SkinThickness (int): Skin thickness in mm.
        Insulin (int): 2-Hour serum insulin.
        BMI (float): Body Mass Index.
        DiabetesPedigreeFunction (float): Diabetes pedigree function.
        Age (int): Age of the individual.
    """
    Pregnancies: conint(ge=0) = 0
    Glucose: int = 118
    BloodPressure: int = 84
    SkinThickness: int = 47
    Insulin: int = 230
    BMI: float = 45.8
    DiabetesPedigreeFunction: float = 0.551
    Age: int = 31

# Create a FastAPI application
app = fastapi.FastAPI()

# Add CORS middleware to allow all origins, methods, and headers for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar a pasta frontend para servir arquivos estáticos
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend") ## acrescentado para correr.

@app.on_event("startup")
async def startup_event():
    """
    Set up actions to perform when the app starts.

    Configures the tracking URI for MLflow to locate the model metadata
    in the local mlruns directory.
    """
    mlflow.set_tracking_uri("./mlruns")
    
    # Load the application configuration
    with open('./config/app.json') as f:
        config = json.load(f)

    # Load the registered model specified in the configuration
    app.model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{config['model_name']}/{config['model_version']}"
    )
    
    print(f"Loaded model {config['model_name']}/{config['model_version']}")


@app.post("/has_diabetes")
async def predict(input: Request):  
    """
    Prediction endpoint that processes input data and returns a model prediction.

    Parameters:
        input (Request): Request body containing input values for the model.

    Returns:
        dict: A dictionary with the model prediction under the key "prediction".
    """

    # Build a DataFrame from the request data
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.dict().items()})

    # Predict using the model and retrieve the first item in the prediction list
    prediction = app.model.predict(input_df)

    # Return the prediction result as a JSON response
    return {"prediction": prediction.tolist()[0]}

if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=5003)
#Run the app on port 5003
#uvicorn.run(app=app, port=5003)
