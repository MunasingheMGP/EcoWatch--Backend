from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import uuid
from fastapi.responses import FileResponse
import os
from services.wildlife_comp.wildlife_service import WildlifeService

router = APIRouter()
service = WildlifeService()

@router.get("/")
async def health_check():
    return {"component": "Wild Species Identification", "status": "ok"}

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...), 
    location: str = Form(...), 
    date: str = Form(...), 
    time: str = Form(...),
):
    try:
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        
        output_video_path = file.filename.rsplit('.', 1)[0] + '_processed.mp4'

        task_id = service.process_video(file.filename, output_video_path, location, date, time)
        return {"task_id": task_id}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/video/input/{task_id}")
async def get_results(task_id: uuid.UUID):
    results = service.get_results(task_id)
    
    path = results["video"]
    base_dir = os.path.dirname(os.path.realpath(__file__)) 
    video_path = os.path.join(base_dir, f"../{path}")

    return FileResponse(video_path, media_type="video/mp4")


@router.get("/results/video/output/{task_id}")
async def get_results(task_id: uuid.UUID):
    results = service.get_results(task_id)
    
    path = results["output"]
    base_dir = os.path.dirname(os.path.realpath(__file__)) 
    video_path = os.path.join(base_dir, f"../{path}")

    return FileResponse(video_path, media_type="video/mp4")

@router.get("/results/csv/{task_id}")
async def get_excel(task_id: uuid.UUID):
    results = service.get_results(task_id)

    path = results["csv"]
    base_dir = os.path.dirname(os.path.realpath(__file__)) 
    csv_path = os.path.join(base_dir, f"../{path}")

    return FileResponse(csv_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")