import os
from typing import List

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from src.logger import get_logger

# Get Logger
my_logger = get_logger(__name__)

# Create fastAPI instance
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Prediction(BaseModel):
    """Define API response object class.

    Args:
        BaseModel (pydantic.BaseModel): Base model for API response classes.
    """

    prediction: str
    score: float


def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Forbidden"
        )


@app.get("/predict", response_model=Prediction, dependencies=[Depends(api_key_auth)])
async def predict(input_texts: List[str]):
    """Receive an input, make a prediction and return the prediction and score.

    Args:
        input (str): Input text.

    Returns:
        output_dict: A dictionary containing the model's prediction and score.
    """
    my_logger.info(f"Logs file path: {os.getenv('LOGS_DIR')}")

    # Dummy hardcoded function, to show how to return the output of the model
    output_dict = {"prediction": "", "score": 0.0}
    if input_texts[0].lower() == "hello world":
        output_dict["prediction"] = "Hello World!"
        output_dict["score"] = 1.0

    return output_dict


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
