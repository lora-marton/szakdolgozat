"""
Feedback orchestrator.

Coordinates text feedback generation, timeline marker extraction,
and feedback video rendering into a single entry point.
"""

import logging
import os

from model.config import DEFAULT_FEEDBACK_CONFIG
from model.feedback.text_feedback import TextFeedback
from model.feedback.video_feedback import VideoFeedback

logger = logging.getLogger(__name__)

class FeedbackGenerator:
    """Top-level feedback pipeline orchestrator."""

    @staticmethod
    def generate_feedback(
        results: dict,
        teacher_video: str,
        student_video: str,
        output_dir: str,
        config=None,
    ) -> dict:
        """Generate all feedback artifacts from comparison results.

        Args:
            results: Dict returned by Comparator.compare_dances().
            teacher_video: Path to the teacher video file.
            student_video: Path to the student video file.
            output_dir: Session directory for output files.
            config: FeedbackConfig instance (uses default if None).

        Returns:
            Dict with 'feedback' (list of text messages),
            'timeline_markers' (list of marker dicts),
            and 'feedback_video' (output filename).
        """
        if config is None:
            config = DEFAULT_FEEDBACK_CONFIG

        logger.info("Timing cost: %s", results["timing_cost"])

        feedback = TextFeedback.generate_messages(results, config)
        timeline_markers = TextFeedback.extract_timeline_markers(results, config)
        video_path = VideoFeedback.render_video(
            teacher_video,
            student_video,
            output_dir,
            results.get("preprocess_info"),
        )

        return {
            "feedback": feedback,
            "timeline_markers": timeline_markers,
            "feedback_video": os.path.basename(video_path),
        }
