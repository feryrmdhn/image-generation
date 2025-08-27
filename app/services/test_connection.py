import boto3
from dotenv import load_dotenv
import os

# Run python -m app.services.test_connection

load_dotenv()

AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

def check_bedrock():
    try:
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            return "GAGAL menemukan credentials"
        
        client = boto3.client(
            'bedrock',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        models = client.list_foundation_models()
        return f"SUKSES: {len(models['modelSummaries'])} model tersedia"

    except Exception as e:
        return f"Error: {str(e)}"

def get_bedrock_client():
    return boto3.client(
        'bedrock-runtime',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

if __name__ == "__main__":
    print(check_bedrock())