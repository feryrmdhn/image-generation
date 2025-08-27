from fastapi import FastAPI
from app.services.generate import router as generate_router

app = FastAPI(
    title="Imagen API",
    description="AI Image Generation API using AWS Bedrock",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "message": "Imagen API - AI Image Generation Service",
        "version": "1.0.0",
    }

app.include_router(generate_router)

