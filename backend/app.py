import uvicorn
from fastapi import FastAPI
from keras.models import load_model

app = FastAPI()
model = load_model("model.h5")


@app.get("/")
def index():
    return {"message": "Cars Recommender ML API"}
