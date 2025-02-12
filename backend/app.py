import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io

from keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from digit_image import DigitImage

app = FastAPI()

model_path = "../deeplearning-model/model.h5"
model = load_model(model_path)


@app.get("/")
def index():
    return {"message": "Cars Recommender ML API"}


# @app.post('/predict')
# def predict_car_type(data:DigitImage):


@app.get("/check")
def check_image_preprocessing():
    path = "../public/digit.png"
    processed_img = preprocess_image(path)

    # Create a Matplotlib figure and plot the processed image
    fig, ax = plt.subplots()
    ax.imshow(np.squeeze(processed_img))

    # Save the figure to a BytesIO buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    if not img:
        print("!Image Not Found!")
        return None

    img = img.resize(
        (28, 28), Image.Resampling.LANCZOS
    )  # Resize image to 28x28 pixels (if not already)
    img_array = np.array(img)

    if np.mean(img_array) > 127:
        img_array = 255 - img_array  # Invert image: white becomes black and vice versa.

    img_array = img_array / 255.0  # Normalize pixel values to the range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
