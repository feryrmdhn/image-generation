from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.titan_generate import router as generate_router
from app.services.stability_generate import router as stability_router
app = FastAPI(
    title="Imagen API",
    description="AI Image Generation API using AWS Bedrock",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Imagen API - AI Image Generation Service",
        "version": "1.0.0",
    }

app.include_router(generate_router)
app.include_router(stability_router)
