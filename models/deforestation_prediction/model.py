import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from keras.models import load_model
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
import cv2
from models.deforestation_prediction.smooth_tiled_predictions import predict_img_with_smooth_windowing

BASE_DIR = Path(__file__).resolve().parent
scaler = MinMaxScaler()
BACKBONE = 'resnet34'
PATCH_SIZE = 256
n_classes = 6

preprocess_input = sm.get_preprocessing(BACKBONE)
model = load_model(f"{BASE_DIR}/35_epochs_model.hdf5", compile=False)


def predict(input):
    img = np.array(input)

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    input_img = scaler.fit_transform(
        img.reshape(-1, img.shape[-1])).reshape(img.shape)
    input_img = preprocess_input(input_img)

    predictions_smooth = predict_img_with_smooth_windowing(
        input_img,
        window_size=PATCH_SIZE,
        subdivisions=2,
        nb_classes=n_classes,
        pred_func=(
            lambda img_batch_subdiv: model.predict((img_batch_subdiv))
        )
    )

    final_prediction = np.argmax(predictions_smooth, axis=2)
    unique_values, count = np.unique(final_prediction, return_counts=True)
    value = {int(k): int(v) for k, v in zip(unique_values, count)}

    return final_prediction, value
