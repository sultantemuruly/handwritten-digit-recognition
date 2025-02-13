import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

import os
import io

from keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from digit_image import DigitImage

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model_path = "../deeplearning-model/model.h5"
model = load_model(model_path)


@app.get("/")
def index():
    return {"message": "Cars Recommender ML API"}


@app.post("/predict")
async def predict_digit_type(file: UploadFile = File(...)):
    if file.content_type != "image/png":
        return {"error": "Only PNG files are allowed"}

    # Read and process the uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    processed_img = preprocess_image(image)

    # Make prediction
    prob = model.predict(processed_img)
    pred = prob.argmax(axis=1)

    return {"prediction": int(pred[0])}


@app.get("/check")
def check_image_preprocessing():
    path = "../public/digit.png"
    img = Image.open(path).convert("L")
    if not img:
        print("!Image Not Found!")
        return None

    processed_img = preprocess_image(img)
    prob = model.predict(processed_img)
    pred = prob.argmax(axis=1)

    return {"prediction": int(pred[0])}


def preprocess_image(image: Image.Image):
    img = image.resize(
        (28, 28), Image.Resampling.LANCZOS
    )  # Resize image to 28x28 pixels (if not already)
    img_array = np.array(img)

    if np.mean(img_array) > 127:
        img_array = 255 - img_array  # Invert image: white becomes black and vice versa.

    img_array = img_array / 255.0  # Normalize pixel values to the range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
