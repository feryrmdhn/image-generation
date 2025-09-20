import boto3
import json
import base64
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid
import random
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.utils.utils import get_bedrock_client

load_dotenv()

# Load environment variables
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
AWS_BEDROCK_MODEL_ID = os.getenv('AWS_BEDROCK_MODEL_ID')
AWS_BEDROCK_GUARDRAIL_ID = os.getenv('AWS_BEDROCK_GUARDRAIL_ID')

router = APIRouter(prefix="/titan-generate", tags=["Titan Image Generation"])

# Pydantic models
class ImageGenerateRequest(BaseModel):
    prompt: str
    number_of_images: Optional[int] = 1
    input_size: Optional[str] = "landscape"  # 'landscape' atau 'potrait'

class ImageMetadata(BaseModel):
    prompt: str
    number_of_images: int

class ImageGenerateResponse(BaseModel):
    status: int
    message: str
    data: List[str]
    metadata: ImageMetadata

def generate_image_service(
    prompt: str,
    number_of_images: int = 1,
    input_size: str = "landscape"
):
    try:
        # Validasi input
        if not prompt:
            return {"error": "Prompt tidak boleh kosong"}
        if not AWS_BUCKET_NAME:
            return {"error": "AWS_BUCKET_NAME tidak ditemukan"}
        if number_of_images > 5:
            return {"error": "Maksimal jumlah gambar yang dapat di-generate adalah 5."}
        if input_size not in ["landscape", "portrait"]:
            return {"error": "input_size harus 'landscape' atau 'portrait'"}

        # Set ukuran gambar sesuai input_size
        if input_size == "landscape":
            width, height = 1152, 768
        else:
            width, height = 448, 576
        
        # Inisialisasi Bedrock client
        bedrock_client = get_bedrock_client()
        
        # Inisialisasi S3 client
        s3_client = boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        negative_prompt = "low quality, bad quality, worst quality, blurry, out of focus, deformed, bad anatomy, cross-eye, deformed eyes"
        
        # Prepare request body untuk Amazon Titan Image Generator (correct format)
        request_body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": negative_prompt
            },
            "imageGenerationConfig": {
                "numberOfImages": min(number_of_images, 5),
                "quality": "premium",
                "width": width,
                "height": height,
                "cfgScale": 8,
                "seed": random.randint(0, 9999999)
            }
        }
        
        # Generate image
        response = bedrock_client.invoke_model(
            modelId=AWS_BEDROCK_MODEL_ID,
            body=json.dumps(request_body),
            contentType='application/json',
            accept='application/json',
            guardrailIdentifier=AWS_BEDROCK_GUARDRAIL_ID,
            guardrailVersion="DRAFT"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())

        guardrail_action = response_body.get("amazon-bedrock-guardrailAction")
        
        if guardrail_action == "INTERVENED":
            return {
                "status": 502,
                "message": "Maaf, Prompt yang anda gunakan melanggar kebijakan kami",
                "data": [],
                "metadata": {
                    "prompt": prompt,
                    "number_of_images": number_of_images
                }
            }
        
        # Store images ke S3 (Titan response format)
        image_urls = []
        for i, image in enumerate(response_body['images']):
            # Decode base64 image
            image_data = base64.b64decode(image)
            
            # Generate nama file unik
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_images/{timestamp}_{uuid.uuid4().hex[:8]}_{i+1}.png"
            
            # Upload ke S3
            s3_client.put_object(
                Bucket=AWS_BUCKET_NAME,
                Key=filename,
                Body=image_data,
                ContentType='image/png'
            )
            
            # Generate S3 URL
            image_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"
            image_urls.append(image_url)
        
        return {
            "status": 201,
            "message": f"Berhasil generate {len(image_urls)} image",
            "data": image_urls,
            "metadata": {
                "prompt": prompt,
                "number_of_images": number_of_images
            }
        }
        
    except Exception as e:
        return {
            "status": 500,
            "message": str(e)
        }

@router.post("/image", response_model=ImageGenerateResponse)
async def generate_image_endpoint(request: ImageGenerateRequest):
    result = generate_image_service(
        prompt=request.prompt,
        number_of_images=request.number_of_images,
        input_size=request.input_size or "landscape"
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    if result["status"] >= 400:
        raise HTTPException(status_code=result["status"], detail=result["message"])
    return ImageGenerateResponse(**result)
