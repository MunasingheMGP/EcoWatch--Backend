from fastapi import BackgroundTasks, UploadFile, HTTPException
from PIL import Image
import asyncio
import uuid
import base64
import os
import io
from models.forest_health.model import predict


prediction_results = {}


class ForestHealthService():

    async def initiate_prediction(self, file: UploadFile):
        name, extension = os.path.splitext(file.filename)
        contents = await file.read()

        if extension not in [".tif"]:
            raise HTTPException(
                status_code=400, detail="File type not supported")

        task_id = uuid.uuid4()

        sum_biomass, estimated_carbon_storage, biomass_density, carbon_density, input_img, total_area_km2 = await self.__process_model(name, contents)

        prediction_results[task_id] = {"filename": file.filename,
                                        "sum_biomass": sum_biomass,
                                        "estimated_carbon_storage": estimated_carbon_storage,
                                        "biomass_density": biomass_density,
                                        "carbon_density": carbon_density,
                                        "total_area_km2": total_area_km2,
                                        "input_img": input_img}
        
        return task_id
        
        
    async def __process_model(self, name: str, input_img):
        if input_img:
            sum_biomass, estimated_carbon_storage, biomass_density, carbon_density, total_area_km2 = predict(name, input_img)

            # Open the image and convert it to a suitable mode
            image = Image.open(io.BytesIO(input_img))
            image = image.convert("RGB")  # Convert the image to RGB mode

            buffered_input = io.BytesIO()
            image.save(buffered_input, format="JPEG")
            input_img_base64 = base64.b64encode(buffered_input.getvalue()).decode()
            
        return sum_biomass, estimated_carbon_storage, biomass_density, carbon_density, input_img_base64, total_area_km2
    
    
    def get_results(self, task_id:uuid.UUID):
        results = prediction_results.get(task_id, None)
        return results
