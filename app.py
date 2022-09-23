import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

# Load the app configuration file
with open("app_config.yaml") as f:
    APP_CONFIG = yaml.safe_load(f)


class Prediction(BaseModel):
    """Define API response object class.

    Args:
        BaseModel (pydantic.BaseModel): Base model for API response classes.
    """

    prediction: str
    score: float


# Create fastAPI instance
app = FastAPI()


@app.get("/predict", response_model=Prediction)
async def predict(input: str) -> dict:
    """Receive an input, make a prediction and return the prediction and score.

    Args:
        input (str): Input text.

    Returns:
        dict: A dictionary containing a model's prediction and score.
    """
    # Dummy hardcoded function, to show how to return the output of the model
    output_dict = {"prediction": "", "score": 0.0}
    if input.lower() == "hello world":
        output_dict["prediction"] = "Hello World!"
        output_dict["score"] = 1.0

    return output_dict


if __name__ == "__main__":
    uvicorn.run(app, port=APP_CONFIG["port"])
