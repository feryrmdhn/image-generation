import boto3
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
from botocore.exceptions import ClientError
from app.utils.utils import get_bedrock_client

load_dotenv()

AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
AWS_STABILITY_MODEL_ID = os.getenv('AWS_STABILITY_MODEL_ID') 
AWS_BEDROCK_GUARDRAIL_ID = os.getenv('AWS_BEDROCK_GUARDRAIL_ID') 

router = APIRouter(prefix="/stability-generate", tags=["Stability Image Generation"])

# Pydantic models
class ImageGenerateRequest(BaseModel):
    prompt: str
    output_format: Optional[str] = "png"

class ImageMetadata(BaseModel):
    prompt: str
    output_format: str

class ImageGenerateResponse(BaseModel):
    status: int
    message: str
    data: str
    metadata: ImageMetadata

def generate_image_service(
    prompt: str,
    output_format: str
):
    try:
        # Validasi input
        if not prompt:
            return {"error": "Prompt tidak boleh kosong"}
        if not AWS_BUCKET_NAME:
            return {"error": "AWS_BUCKET_NAME tidak ditemukan"}
        if output_format not in ["png", "jpeg"]:
            return {"error": "output_format harus 'png' atau 'jpeg'"}

        bedrock_client = get_bedrock_client()

        # Inisialisasi S3 client
        s3_client = boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        negative_prompt = "low quality, bad quality, worst quality, blurry, out of focus, deformed, bad anatomy, cross-eye, deformed eyes"

        # Request body untuk Stable Image Core v1.0 (support hanya 1 image)
        request_body = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "aspect_ratio": "3:2",
            "output_format": output_format,
            "seed": random.randint(0, 9999999)
        }

        response = bedrock_client.invoke_model(
            modelId=AWS_STABILITY_MODEL_ID,
            body=json.dumps(request_body),
            contentType='application/json',
            accept='application/json',
            guardrailIdentifier=AWS_BEDROCK_GUARDRAIL_ID,
            guardrailVersion="DRAFT"
        )

        response_body = json.loads(response['body'].read())

        images = response_body.get("images", [])
        guardrail_action = response_body.get("amazon-bedrock-guardrailAction")

        if not images:
            if guardrail_action == "INTERVENED":
                msg = "Maaf, Prompt yang anda gunakan melanggar kebijakan kami"
            else:
                msg = json.dumps(response_body, ensure_ascii=False)
            return {
                "status": 502,
                "message": msg,
                "data": [],
                "metadata": {
                    "prompt": prompt,
                    "output_format": output_format
                }
            }

        image_data = base64.b64decode(images[0])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "jpg" if output_format in ["jpeg", "jpg"] else "png"
        filename = f"generated_images/stability_{timestamp}_{uuid.uuid4().hex[:8]}_1.{ext}"
        
         # Upload ke S3
        s3_client.put_object(
            Bucket=AWS_BUCKET_NAME,
            Key=filename,
            Body=image_data,
            ContentType=f'image/{ext}'
        )

        image_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"

        return {
            "status": 201,
            "message": "Berhasil generate image",
            "data": image_url,
            "metadata": {
                "prompt": prompt,
                "output_format": output_format
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
            output_format=request.output_format or "png"
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return ImageGenerateResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
