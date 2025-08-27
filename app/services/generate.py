import boto3
import json
import base64
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

load_dotenv()

# Load environment variables
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
AWS_BEDROCK_MODEL_ID = os.getenv('AWS_BEDROCK_MODEL_ID')

router = APIRouter(prefix="/generate", tags=["Image Generation"])

# Pydantic models
class ImageGenerateRequest(BaseModel):
    prompt: str
    number_of_images: Optional[int] = 1

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
    number_of_images: int = 1
):
    try:
        # Validate inputs
        if not prompt:
            return {"error": "Prompt tidak boleh kosong"}
        
        if not AWS_BUCKET_NAME:
            return {"error": "AWS_BUCKET_NAME tidak ditemukan"}
        
        # Setup Bedrock client
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Setup S3 client
        s3_client = boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        negative_prompt = "low quality, bad quality, worst quality, blurry, out of focus, deformed, unrealistic, unreal, bad anatomy"
        
        # Prepare request body untuk Amazon Titan Image Generator (correct format)
        request_body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": negative_prompt
            },
            "imageGenerationConfig": {
                "numberOfImages": min(number_of_images, 5),
                "quality": "standard",
                "height": 1024,
                "width": 1024,
                "cfgScale": 8,
                "seed": 0
            }
        }
        
        # Generate image
        response = bedrock_client.invoke_model(
            modelId=AWS_BEDROCK_MODEL_ID,
            body=json.dumps(request_body),
            contentType='application/json'
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
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
    try:
        result = generate_image_service(
            prompt=request.prompt,
            number_of_images=request.number_of_images
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return ImageGenerateResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def get_bedrock_client():
    return boto3.client(
        'bedrock-runtime',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )