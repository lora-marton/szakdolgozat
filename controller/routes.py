"""
FastAPI application with all API endpoints.

Provides endpoints for video upload and processing, 
feedback retrieval, video streaming, and session listing.
"""
import asyncio
import os
import shutil
import sys
from datetime import datetime

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from controller.services import SessionService, DATA_DIR
from model.video_processor import VideoProcessor

UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploaded_videos')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

status_queue = asyncio.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown lifecycle.

    On Windows, resets SIGINT to default so Ctrl+C stops the server
    and suppresses spurious ConnectionResetError from the Proactor loop.
    """
    if sys.platform == "win32":
        import signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        loop = asyncio.get_running_loop()

        def _suppress_proactor_errors(loop, context):
            exception = context.get("exception")
            if isinstance(exception, ConnectionResetError):
                return
            loop.default_exception_handler(context)

        loop.set_exception_handler(_suppress_proactor_errors)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        output_dir = os.path.join(DATA_DIR, session_id)

        results = await VideoProcessor.process_videos(
            teacher_path, student_path, output_dir,
            event_handler=sse_status_handler,
        )

        if results is not None:
            SessionService.save_results(session_id, results)
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


@app.get("/events")
async def sse_events():
    """Stream real-time processing status updates via Server-Sent Events."""

    async def event_generator():
        while True:
            message = await status_queue.get()
            yield f"data: {message}\n\n"
            if message == "Processing complete.":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/feedback")
async def get_feedback(
    session_id: str | None = Query(default=None),
) -> JSONResponse:
    """Retrieve comparison feedback for a session.

    Query params:
        session_id: Optional — if omitted, returns the latest session.

    Returns:
        JSON with overall_score, skeleton_score, trajectory_score,
        mask_score, per_joint_scores, feedback messages, etc.
    """
    if session_id is None:
        session_id = SessionService.find_latest_session()
        if session_id is None:
            raise HTTPException(status_code=404, detail="No sessions found.")

    results = SessionService.load_results(session_id)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for session '{session_id}'.",
        )

    video_file = os.path.join(DATA_DIR, session_id, 'feedback_video.mp4')
    video_url = (
        f"/feedback/video?session_id={session_id}"
        if os.path.isfile(video_file)
        else None
    )

    return JSONResponse(content={
        "session_id": session_id,
        "overall_score": results.get("overall_score"),
        "skeleton_score": results.get("skeleton_score"),
        "trajectory_score": results.get("trajectory_score"),
        "mask_score": results.get("mask_score"),
        "timing_cost": results.get("timing_cost"),
        "per_joint_scores": results.get("per_joint_scores"),
        "feedback": results.get("feedback", []),
        "timeline_markers": results.get("timeline_markers", []),
        "feedback_video_url": video_url,
    })


@app.get("/feedback/detailed")
async def get_detailed_feedback(
    session_id: str | None = Query(default=None),
) -> JSONResponse:
    """Retrieve full comparison results including per-frame data.

    Useful for visualisation / charting on the frontend.
    """
    if session_id is None:
        session_id = SessionService.find_latest_session()
        if session_id is None:
            raise HTTPException(status_code=404, detail="No sessions found.")

    results = SessionService.load_results(session_id)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for session '{session_id}'.",
        )

    return JSONResponse(content={
        "session_id": session_id,
        **results,
    })


@app.get("/feedback/video")
async def get_feedback_video(
    session_id: str | None = Query(default=None),
) -> FileResponse:
    """Stream the feedback comparison video for a session.

    Query params:
        session_id: Optional — if omitted, uses the latest session.
    """
    if session_id is None:
        session_id = SessionService.find_latest_session()
        if session_id is None:
            raise HTTPException(status_code=404, detail="No sessions found.")

    video_path = os.path.join(DATA_DIR, session_id, 'feedback_video.mp4')
    if not os.path.isfile(video_path):
        raise HTTPException(
            status_code=404,
            detail=f"No feedback video found for session '{session_id}'.",
        )

    return FileResponse(
        video_path,
        media_type='video/mp4',
        filename='feedback_video.mp4',
    )


@app.get("/sessions")
async def list_sessions() -> JSONResponse:
    """List all available session IDs."""
    if not os.path.isdir(DATA_DIR):
        return JSONResponse(content={"sessions": []})

    sessions = sorted(
        [d for d in os.listdir(DATA_DIR)
         if os.path.isdir(os.path.join(DATA_DIR, d))],
        reverse=True,
    )
    return JSONResponse(content={"sessions": sessions})


if __name__ == "__main__":
    print("Starting server on http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
