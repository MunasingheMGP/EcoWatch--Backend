from fastapi import BackgroundTasks, UploadFile, HTTPException
from PIL import Image
import asyncio
import uuid
import base64
import os
import io
from models.classification_prediction.model import predict

prediction_results = {}


class ClassificationService():

    async def initiate_prediction(self, file: UploadFile):
        name, extension = os.path.splitext(file.filename)
        contents = await file.read()

        if extension not in [".png", ".jpg", ".jpeg", ".tif"]:
            raise HTTPException(
                status_code=400, detail="File type not supported")

        task_id = uuid.uuid4()
        print(f"Task ID: {task_id}")

        # image = Image.open(io.BytesIO(contents))
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        class_name, score, input_img = await self.__process_model(name, image)

        print(f"Class: {class_name}, Score: {score}")

        prediction_results[task_id] = {"filename": name,
                                       "class": class_name,
                                       "score": score,
                                       "input_img": input_img}

        return task_id

    async def __process_model(self, name: str, input_img):
        if input_img:
            class_name, score = predict(input_img)

            buffered_input = io.BytesIO()
            input_img.save(buffered_input, format="JPEG")
            input_img_base64 = base64.b64encode(
                buffered_input.getvalue()).decode()

        if class_name is None or score is None:
            raise Exception(f'Error processing {name}')

        return class_name, score, input_img_base64

    def get_results(self, task_id: uuid.UUID):
        results = prediction_results.get(task_id, None)
        return results
