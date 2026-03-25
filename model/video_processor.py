"""
Video processing pipeline orchestrator.

Ties together extraction, comparison, feedback generation, and video
rendering into a single async pipeline with real-time SSE status updates.
"""
import asyncio
import os
import traceback
from typing import Callable, Awaitable

from model.comparison.comparator import compare_dances
from model.feedback.feedback import extract_timeline_markers, generate_feedback
from model.feedback.video_feedback import generate_feedback_video
from model.config import DEFAULT_COMPARISON_CONFIG
from model.config.comparison_config import ComparisonConfig
from model.extraction.extractor import data_extraction


class VideoProcessor:
    """Orchestrates the four-phase dance comparison pipeline."""

    @staticmethod
    async def process_videos(
        teacher_file: str,
        student_file: str,
        output_dir: str = 'data',
        config: ComparisonConfig | None = None,
        event_handler: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict | None:
        """Process videos: extract poses, compare, generate feedback.

        Runs heavy computation in threads to avoid blocking the async event loop.

        Args:
            teacher_file: Path to the reference dance video.
            student_file: Path to the student's dance video.
            output_dir: Directory for extracted data and results.
            config: ComparisonConfig to use (defaults to DEFAULT_COMPARISON_CONFIG).
            event_handler: Optional async callback for SSE status updates.

        Returns:
            Dict with scores, feedback, timeline markers, and video filename,
            or None if processing failed.
        """
        if config is None:
            config = DEFAULT_COMPARISON_CONFIG

        async def send_status(msg: str) -> None:
            print(msg)
            if event_handler:
                await event_handler(msg)

        if os.path.exists(teacher_file) and os.path.exists(student_file):
            await send_status("Processing videos...")
        else:
            await send_status("Error: Videos not found.")
            return None

        try:
            await asyncio.to_thread(data_extraction, teacher_file, output_dir, 'teacher')
            await send_status("Teacher video extracted.")

            await asyncio.to_thread(data_extraction, student_file, output_dir, 'student')
            await send_status("Student video extracted.")

            await send_status("Comparing performances...")
            results = await asyncio.to_thread(
                compare_dances, output_dir,
                teacher_video=teacher_file, student_video=student_file,
                config=config,
            )
            await send_status(f"Comparison complete. Overall score: {results['overall_score']}%")

            await send_status("Generating feedback...")
            results['feedback'] = generate_feedback(results, config)
            results['timeline_markers'] = extract_timeline_markers(results, config)
            await send_status("Feedback ready.")

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