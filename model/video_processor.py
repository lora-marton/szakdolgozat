import os
from model.extractor import data_extraction

async def process_videos(teacher_file, student_file, event_handler=None):
    """
    Process videos and send status updates via event_handler(message: str)
    """
    async def send_status(msg):
        print(msg)
        if event_handler:
            await event_handler(msg)

    if os.path.exists(teacher_file) and os.path.exists(student_file):
        await send_status("Processing videos...")
    else:
        await send_status("Error: Videos not found.")
        return

    try:
        # Wrapper for sync callback if needed, though we can't await here easily without a loop reference
        # For now, we just let it run.
        data_extraction(teacher_file)
        await send_status("Teacher video extracted.")
            
        data_extraction(student_file)
        await send_status("Student video extracted.")
        
        await send_status("Processing complete.")
            
    except Exception as e:
        await send_status(f"Error processing videos: {e}")