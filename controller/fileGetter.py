"""
File upload endpoint.

Provides a POST endpoint to upload teacher and student dance videos,
run the comparison pipeline, and return feedback results.
Uses FastAPI with CORS and Server-Sent Events for real-time progress.
"""
import asyncio
import json
import os
import shutil
import sys
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.video_processor import VideoProcessor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploaded_videos')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

status_queue = asyncio.Queue()


class FileGetter:
    """Groups file handling and result persistence logic."""

    @staticmethod
    def save_results(session_id: str, results: dict) -> None:
        """Persist comparison results to disk as JSON.

        Converts numpy arrays to lists for JSON serialization.

        Args:
            session_id: Session directory name (e.g. '20260306_134500').
            results: Dict returned by process_videos (with feedback attached).
        """
        import numpy as np

        session_dir = os.path.join(DATA_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, dict):
                serializable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable[key] = value

        results_path = os.path.join(session_dir, 'results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)


# ── Endpoints ────────────────────────────────────────────────────────────


@app.post("/dance_videos")
async def upload_files(
    teacher: UploadFile = File(...),
    student: UploadFile = File(...),
) -> dict:
    """Upload teacher and student dance videos, process and compare them.

    Args:
        teacher: The reference dance video file.
        student: The student's dance video file.

    Returns:
        Dict with session_id, scores, and feedback.
    """
    try:
        teacher_path = os.path.join(UPLOAD_FOLDER, teacher.filename)
        student_path = os.path.join(UPLOAD_FOLDER, student.filename)

        with open(teacher_path, "wb") as buffer:
            shutil.copyfileobj(teacher.file, buffer)

        with open(student_path, "wb") as buffer:
            shutil.copyfileobj(student.file, buffer)

        async def sse_status_handler(message: str) -> None:
            await status_queue.put(message)

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(ROOT_DIR, 'data', session_id)

        results = await VideoProcessor.process_videos(teacher_path, student_path, output_dir, event_handler=sse_status_handler)

        if results is not None:
            FileGetter.save_results(session_id, results)
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
    print("Starting server on http://localhost:8000/dance_videos")
    uvicorn.run(app, host="127.0.0.1", port=8000)
