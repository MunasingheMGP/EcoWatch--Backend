from keras.models import load_model
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
model = load_model(f"{BASE_DIR}/keras_model.h5", compile=False)

# The list of animal classes
forest_classes = ["  boreal", "  dry_and_desert", "  rainforest" ]

def predict(input):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(input, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = forest_classes[index]
    confidence_score = float(prediction[0][index])  # Convert numpy.float32 to regular float

    return class_name[2:], confidence_score  # returning class_name[2:] to skip initial characters


