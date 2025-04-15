"""
Deploy the trained model using FastAPI
"""
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

class Input(BaseModel):
    pixels: list

@app.post("/predict")
def predict(input: Input):
    model = torch.load("mnist_model.pth")
    model.eval()
    with torch.no_grad():
        x = torch.tensor(input.pixels).float().view(-1, 28*28)
        output = model(x)
        _, predicted = torch.max(output, 1)
    return {"prediction": int(predicted)}
