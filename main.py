import numpy as np
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import uvicorn
#import tensorflow as tf


app = FastAPI()

#MODEL = tf.saved_model.load("C:/Users/krpee/PotatoDiseaseClassification/saved_models/1")
#CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image_image = Image.open(BytesIO(data))
    width, height = image_image.size
    image = np.array([image_image.getpixel((x, y)) for y in range(height) for x in range(width)])
    image = image.reshape((height, width, -1))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    #img_batch = np.expand_dims(image, 0)

    #predictions = MODEL.predict(img_batch)
    return


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)