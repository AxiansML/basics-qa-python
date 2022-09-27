import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


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
    uvicorn.run(app, port=8000)
