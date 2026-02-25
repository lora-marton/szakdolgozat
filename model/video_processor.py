import os
import asyncio
from model.extractor import data_extraction
from model.comparator import compare_dances


async def process_videos(teacher_file, student_file, output_dir='data', event_handler=None):
    """
    Process videos: extract poses, then compare teacher vs student.
    Runs heavy computation in threads to avoid blocking the async event loop.
    """
    async def send_status(msg):
        print(msg)
        if event_handler:
            await event_handler(msg)

    if os.path.exists(teacher_file) and os.path.exists(student_file):
        await send_status("Processing videos...")
    else:
        await send_status("Error: Videos not found.")
        return None

    try:
        # Phase 1: Extraction
        await asyncio.to_thread(data_extraction, teacher_file, output_dir, 'teacher')
        await send_status("Teacher video extracted.")

        await asyncio.to_thread(data_extraction, student_file, output_dir, 'student')
        await send_status("Student video extracted.")

        # Phase 2: Comparison
        await send_status("Comparing performances...")
        results = await asyncio.to_thread(compare_dances, output_dir)
        await send_status(f"Comparison complete. Overall score: {results['overall_score']}%")

        return results

    except Exception as e:
        await send_status(f"Error processing videos: {e}")
        return None