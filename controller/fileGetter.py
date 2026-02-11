import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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
# Get the directory of this script (controller/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to root, then into uploaded_videos
UPLOAD_FOLDER = os.path.join(os.path.dirname(CURRENT_DIR), 'uploaded_videos')

# Ensure directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
            
        return {
            "message": "Files uploaded successfully",
            "teacher_path": teacher_path,
            "student_path": student_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(f"Starting server on http://localhost:8000/dance_videos")
    uvicorn.run(app, host="127.0.0.1", port=8000)
