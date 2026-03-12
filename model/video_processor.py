import os
import asyncio
import traceback
from model.extraction.extractor import data_extraction
from model.comparison.comparator import compare_dances
from model.comparison.feedback import generate_feedback
from model.comparison.video_feedback import generate_feedback_video
from model.config import DEFAULT_COMPARISON_CONFIG


async def process_videos(teacher_file, student_file, output_dir='data',
                         config=None, event_handler=None):
    """
    Process videos: extract poses, compare teacher vs student, generate feedback.
    Runs heavy computation in threads to avoid blocking the async event loop.
    """
    if config is None:
        config = DEFAULT_COMPARISON_CONFIG

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
        results = await asyncio.to_thread(
            compare_dances, output_dir,
            teacher_video=teacher_file, student_video=student_file,
            config=config,
        )
        await send_status(f"Comparison complete. Overall score: {results['overall_score']}%")

        # Phase 3: Feedback generation
        await send_status("Generating feedback...")
        results['feedback'] = generate_feedback(results, config)
        await send_status("Feedback ready.")

        # Phase 4: Feedback video generation
        await send_status("Generating feedback video...")
        video_path = await asyncio.to_thread(
            generate_feedback_video,
            teacher_file, student_file, output_dir,
            results.get('preprocess_info'),
        )
        results['feedback_video'] = os.path.basename(video_path)
        await send_status("Feedback video ready.")

        return results

    except Exception as e:
        traceback.print_exc()
        await send_status(f"Error processing videos: {type(e).__name__}: {e}")
        return None