from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.rt_deforestation_comp import router as deforestation_router
from routers.rt_wildlife_comp import router as wildlife_router
from routers.rt_foresthealth_comp import router as health_router
from routers.rt_classification_comp import router as classification_router

app = FastAPI()

origins = [
    "http://localhost:3000",  # React
    "http://localhost:8000",  # FastAPI
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    return {"status": "ok"}

app.include_router(deforestation_router, prefix="/deforestation")
app.include_router(wildlife_router, prefix="/wildlife")
app.include_router(health_router, prefix="/ecohealth")
app.include_router(classification_router, prefix="/classification")