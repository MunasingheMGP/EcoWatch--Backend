from fastapi import APIRouter, UploadFile, File, HTTPException
import uuid
from services.foresthealth_comp.foresthealth_service import ForestHealthService

router = APIRouter()
service = ForestHealthService()


@router.get("/")
async def health_check():
    return {"component": "Forest Health Monitor", "status": "ok"}


@router.post("/predict")
async def detect_foresthealth(file: UploadFile = File(...)):
    try:
        task_id = await service.initiate_prediction(file)
        return {"task_id": task_id}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{task_id}")
async def get_results(task_id: uuid.UUID):
    result = service.get_results(task_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"Task_ID": task_id, "Results": result}