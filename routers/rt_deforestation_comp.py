from fastapi import APIRouter, WebSocket, UploadFile, BackgroundTasks, Depends, File, HTTPException
from websockets.exceptions import ConnectionClosed
import uuid
import asyncio
import copy
from services.deforestation_comp.deforestation_service import DeforestationService

router = APIRouter()
service = DeforestationService()


@router.get("/")
async def health_check():
    return {"component": "deforestation", "status": "ok"}


@router.post("/predict")
async def predict_deforestation(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        task_id = await service.initiate_prediction(file, background_tasks)
        return {"task_id": task_id}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: uuid.UUID):
    await websocket.accept()
    last_sent_progress = None
    while True:
        try:
            progress = service.get_progress(task_id)
            if progress != last_sent_progress:
                await websocket.send_json({"progress": progress})
                last_sent_progress = copy.deepcopy(progress)
            await asyncio.sleep(1)
        except ConnectionClosed:
            break


@router.get("/results/{task_id}")
async def get_results(task_id: uuid.UUID):
    result = service.get_results(task_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"Task_ID": task_id, "Results": result}
