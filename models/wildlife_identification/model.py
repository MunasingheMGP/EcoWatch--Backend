import cv2
import numpy as np
from pathlib import Path
from keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
model = load_model(f"{BASE_DIR}/model.h5", compile=False)

# The list of animal classes
animal_classes = ["antelope", "bear", "boar", "deer", "eagle", "elephant", "fox", "goat", "lion", "owl", "porcupine",
                  "reindeer", "squirrel", "swan", "tiger", "wolf"]


def predict(input):
    img = np.array(input)

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    # Resize the image to fit the model's input shape
    img = cv2.resize(img, (224, 224))

    # Normalize the image
    img = img / 255.0

    # Expand the dimensions to match the model's expected input shape
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    # Decode the prediction
    predicted_class = animal_classes[np.argmax(prediction)]

    return predicted_class, 1
