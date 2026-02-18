import requests
from fastapi import HTTPException

async def send_message(message):
    try:
        response = requests.post("http://localhost:8000/dance_videos")
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))