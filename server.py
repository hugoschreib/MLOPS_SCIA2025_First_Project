from fastapi import FastAPI
from pydantic import BaseModel
import streamlit as st
import joblib as jl

model = jl.load("regression.joblib")


app = FastAPI()

class Body(BaseModel):
    size: int
    number_bedroom: int
    has_garden: int


@app.post("/predict")
def predict(body: Body):
    return {"price": model.predict([[body.size, body.number_bedroom,body.has_garden]])[0]}
