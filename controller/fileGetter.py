import os
import sys
import shutil
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List

# Ensure project root is on path (so imports work regardless of launch method)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.video_processor import process_videos
from controller.feedbackSender import save_results

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure upload folder (root/uploaded_videos)
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploaded_videos')

# Ensure directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

import asyncio
from fastapi.responses import StreamingResponse

# Global event queue for broadcasting
# In a real app this might be a list of queues (one per client)
start_event = asyncio.Event()
status_queue = asyncio.Queue()

async def event_generator():
    """
    Yields events to the client.
    Initial state: waiting for start signal.
    """
    while True:
        # Check if we have messages in the queue
        try:
            # Wait for a message with a timeout so we can send keep-alive comments
            message = await asyncio.wait_for(status_queue.get(), timeout=1.0)
            yield f"data: {message}\n\n"
        except asyncio.TimeoutError:
            # Send a comment to keep the connection alive
            yield ": keep-alive\n\n"

@app.get("/events")
async def events():
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/dance_videos")
async def upload_files(teacher: UploadFile = File(...), student: UploadFile = File(...)):
    try:
        # Define paths
        teacher_path = os.path.join(UPLOAD_FOLDER, teacher.filename)
        student_path = os.path.join(UPLOAD_FOLDER, student.filename)
        
        # Save teacher file
        with open(teacher_path, "wb") as buffer:
            shutil.copyfileobj(teacher.file, buffer)
            
        # Save student file
        with open(student_path, "wb") as buffer:
            shutil.copyfileobj(student.file, buffer)

        # Event handler bridging function
        async def sse_status_handler(message: str):
            await status_queue.put(message)

        # Create a session-specific output directory
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(ROOT_DIR, 'data', session_id)
        
        results = await process_videos(teacher_path, student_path, output_dir, event_handler=sse_status_handler)

        if results is not None:
            # Persist results to disk for later retrieval via feedbackSender
            save_results(session_id, results)
            await sse_status_handler("Processing complete.")

            return {
                "message": "Files uploaded and processed successfully",
                "session_id": session_id,
                "overall_score": results.get("overall_score"),
                "skeleton_score": results.get("skeleton_score"),
                "trajectory_score": results.get("trajectory_score"),
                "mask_score": results.get("mask_score"),
                "feedback": results.get("feedback", []),
            }
        else:
            raise HTTPException(status_code=500, detail="Video processing failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(f"Starting server on http://localhost:8000/dance_videos")
    uvicorn.run(app, host="127.0.0.1", port=8000)

