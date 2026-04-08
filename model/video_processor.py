"""
Video processing pipeline orchestrator.

Ties together extraction, comparison, feedback generation, and video
rendering into a single async pipeline with real-time SSE status updates.
"""

import asyncio
import os
import traceback
from typing import Awaitable, Callable

from model.comparison.comparator import Comparator
from model.extraction.extractor import Extractor
from model.feedback.feedback_generator import FeedbackGenerator
from model.preprocessing.preprocessor import Preprocessor


class VideoProcessor:
    """Orchestrates the four-phase dance comparison pipeline."""

    @staticmethod
    async def process_videos(
        teacher_file: str,
        student_file: str,
        output_dir: str = "data",
        event_handler: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict | None:
        """Process videos: extract poses, compare, generate feedback.

        Runs heavy computation in threads to avoid blocking the async event loop.

        Args:
            teacher_file: Path to the reference dance video.
            student_file: Path to the student's dance video.
            output_dir: Directory for extracted data and results.
            event_handler: Optional async callback for SSE status updates.

        Returns:
            Dict with scores, feedback, timeline markers, and video filename,
            or None if processing failed.
        """

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
            await asyncio.to_thread(Extractor.data_extraction, teacher_file, output_dir, "teacher")
            await send_status("Teacher video extracted.")

            await asyncio.to_thread(Extractor.data_extraction, student_file, output_dir, "student")
            await send_status("Student video extracted.")

            await send_status("Preprocessing...")
            teacher_data, student_data, preprocess_info = await asyncio.to_thread(
                Preprocessor.preprocess,
                output_dir,
                teacher_file,
                student_file,
            )
            await send_status("Preprocessing complete.")

            await send_status("Comparing performances...")
            results = await asyncio.to_thread(
                Comparator.compare_dances,
                teacher_data,
                student_data,
            )
            results["preprocess_info"] = preprocess_info
            await send_status(f"Comparison complete. Overall score: {results['overall_score']}%")

            await send_status("Generating feedback...")
            feedback_result = await asyncio.to_thread(
                FeedbackGenerator.generate_feedback,
                results,
                teacher_file,
                student_file,
                output_dir,
            )
            results.update(feedback_result)
            await send_status("Feedback ready.")

            return results

        except Exception as e:
            traceback.print_exc()
            await send_status(f"Error processing videos: {type(e).__name__}: {e}")
            return None
