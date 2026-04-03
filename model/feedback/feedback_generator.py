"""
Feedback orchestrator.

Coordinates text feedback generation, timeline marker extraction,
and feedback video rendering into a single entry point.
"""
import os

from model.feedback.text_feedback import TextFeedback
from model.feedback.video_feedback import VideoFeedback


class FeedbackGenerator:
    """Top-level feedback pipeline orchestrator."""

    @staticmethod
    def generate_feedback(
        results: dict,
        teacher_video: str,
        student_video: str,
        output_dir: str,
    ) -> dict:
        """Generate all feedback artifacts from comparison results.

        Args:
            results: Dict returned by Comparator.compare_dances().
            teacher_video: Path to the teacher video file.
            student_video: Path to the student video file.
            output_dir: Session directory for output files.

        Returns:
            Dict with 'feedback' (list of text messages),
            'timeline_markers' (list of marker dicts),
            and 'feedback_video' (output filename).
        """
        feedback = TextFeedback.generate_messages(results)
        timeline_markers = TextFeedback.extract_timeline_markers(results)
        video_path = VideoFeedback.render_video(
            teacher_video, student_video, output_dir,
            results.get('preprocess_info'),
        )

        return {
            'feedback': feedback,
            'timeline_markers': timeline_markers,
            'feedback_video': os.path.basename(video_path),
        }
