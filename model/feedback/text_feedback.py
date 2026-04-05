"""
Rule-based text feedback generation for dance comparison results.
"""
import numpy as np

from model.config import DEFAULT_EXTRACTION_CONFIG


_JOINT_TIPS = {
    'elbow':    'Focus on matching the bend of your elbows — keep them sharper or softer as needed.',
    'knee':     'Pay attention to your knee bend — try to match the depth of the teacher\'s plié or stance.',
    'shoulder': 'Watch your shoulder positioning — they may be too raised or too low.',
    'hip':      'Your hip angles are off — focus on the tilt and rotation of your pelvis.',
    'wrist':    'Your wrist angles differ — check if your hands are angled differently from the teacher.',
    'ankle':    'Your ankle positioning differs — pay attention to foot placement and flexion.',
}


class TextFeedback:
    """Rule-based text feedback and timeline marker generation."""

    @staticmethod
    def generate_messages(results: dict, config) -> list[str]:
        """Generate human-readable feedback from comparison results.

        Args:
            results: Dict returned by Comparator.compare_dances().
            config: FeedbackConfig instance.

        Returns:
            Prioritised list of feedback message strings.
        """
        feedback = []

        overall = results['overall_score']
        skeleton = results['skeleton_score']
        trajectory = results['trajectory_score']
        mask = results['mask_score']

        feedback.append(TextFeedback._overall_summary(overall))

        worst_frames = results.get('worst_frames', [])
        worst_msg = TextFeedback._worst_moment(worst_frames, results)
        if worst_msg:
            feedback.append(worst_msg)

        per_joint = results.get('per_joint_scores', {})
        joint_warnings = TextFeedback._joint_warnings(per_joint, config.joint_warn_threshold)
        feedback.extend(joint_warnings)

        traj_msg = TextFeedback._trajectory_warning(trajectory, config.direction_warn_threshold)
        if traj_msg:
            feedback.append(traj_msg)

        shape_msg = TextFeedback._shape_warning(mask, config.mask_warn_threshold)
        if shape_msg:
            feedback.append(shape_msg)

        energy_details = results.get('energy_details', {})
        energy_msg = TextFeedback._energy_mismatch(energy_details, config)
        if energy_msg:
            feedback.append(energy_msg)

        timing_cost = results.get('timing_cost')
        timing_msg = TextFeedback._timing_warning(timing_cost, config)
        if timing_msg:
            feedback.append(timing_msg)

        praise = TextFeedback._praise(skeleton, trajectory, mask, config.praise_threshold)
        feedback.extend(praise)

        return feedback

    @staticmethod
    def extract_timeline_markers(results: dict, config) -> list[dict]:
        """Extract timestamps for the worst frames for the frontend video player.

        Worst frames are the frames with the lowest mean joint score.
        Markers are spaced at least config.min_marker_gap seconds apart so
        they don't cluster at the same moment.

        Args:
            results: Dict returned by Comparator.compare_dances() with
                preprocess_info added by video_processor.
            config: FeedbackConfig instance.

        Returns:
            List of dicts with 'time' (seconds) and 'label' (score string).
        """
        worst_frames = results.get('worst_frames', [])
        if not worst_frames:
            return []

        alignment = results.get('alignment_path', [])
        preprocess_info = results.get('preprocess_info', {})
        audio_offset = preprocess_info.get('audio_offset', 0)
        student_offset = preprocess_info.get('student_offset', 0)

        teacher_fps = results.get('teacher_fps', 30.0)
        student_fps = results.get('student_fps', 30.0)
        source_fps = min(teacher_fps, student_fps)

        markers = []

        for dtw_idx, frame_score in worst_frames:
            time_sec = TextFeedback._frame_to_seconds(
                dtw_idx, alignment, audio_offset, student_offset, source_fps,
            )
            if time_sec is None:
                continue

            time_rounded = round(time_sec, 2)

            too_close = any(
                abs(time_rounded - m['time']) < config.min_marker_gap
                for m in markers
            )
            if too_close:
                continue

            markers.append({
                'time': time_rounded,
                'label': f"{frame_score}%",
            })

            if len(markers) >= 5:
                break

        return markers

    @staticmethod
    def _frame_to_seconds(
        dtw_idx: int,
        alignment: list,
        audio_offset: float,
        student_offset: int,
        source_fps: float,
    ) -> float | None:
        """Convert a DTW alignment index to a timestamp in the feedback video.

        Args:
            dtw_idx: Index into the alignment path.
            alignment: List of (teacher_idx, student_idx) pairs.
            audio_offset: Audio sync offset in target FPS units.
            student_offset: Frames trimmed by motion energy detection.
            source_fps: Source video FPS.

        Returns:
            Timestamp in seconds, or None if the frame is before the video start.
        """
        target_fps = DEFAULT_EXTRACTION_CONFIG.target_fps
        output_fps = DEFAULT_EXTRACTION_CONFIG.output_fps

        if dtw_idx < len(alignment):
            student_idx_preprocessed = alignment[dtw_idx][1]
        else:
            student_idx_preprocessed = alignment[-1][1]

        raw_student_idx = student_idx_preprocessed + student_offset

        scale = source_fps / target_fps
        trim_frames = int(round(abs(audio_offset) * scale))

        if audio_offset < 0:
            video_frame = raw_student_idx - trim_frames
        else:
            video_frame = raw_student_idx

        if video_frame < 0:
            return None

        return video_frame / output_fps

    @staticmethod
    def _overall_summary(score: float) -> str:
        """One-liner summary based on the overall score."""
        if score >= 80:
            return f"Excellent performance! Overall score: {score}%."
        elif score >= 70:
            return f"Good performance with room for improvement. Overall score: {score}%."
        elif score >= 50:
            return f"Decent attempt — several areas need work. Overall score: {score}%."
        else:
            return f"This needs more practice. Overall score: {score}%."

    @staticmethod
    def _worst_moment(worst_frames: list, results: dict) -> str | None:
        """Highlight the single worst frame with a timestamp."""
        if not worst_frames:
            return None

        dtw_idx, frame_score = worst_frames[0]

        alignment = results.get('alignment_path', [])
        preprocess_info = results.get('preprocess_info', {})
        audio_offset = preprocess_info.get('audio_offset', 0)
        student_offset = preprocess_info.get('student_offset', 0)

        teacher_fps = results.get('teacher_fps', 30.0)
        student_fps = results.get('student_fps', 30.0)
        source_fps = min(teacher_fps, student_fps)

        time_sec = TextFeedback._frame_to_seconds(
            dtw_idx, alignment, audio_offset, student_offset, source_fps,
        )

        if time_sec is not None:
            return (
                f"⚠ Your biggest deviation was at {time_sec:.1f}s "
                f"(score: {frame_score}%)."
            )

        return f"⚠ Your biggest deviation scored {frame_score}%."

    @staticmethod
    def _joint_warnings(per_joint_scores: dict, threshold: float) -> list[str]:
        """Flag joints that scored below the warning threshold, grouped by body part."""
        groups: dict[str, dict[str | None, float]] = {}
        for joint, score in per_joint_scores.items():
            if score >= threshold:
                continue
            if joint.startswith('left_'):
                side, base = 'left', joint[5:]
            elif joint.startswith('right_'):
                side, base = 'right', joint[6:]
            else:
                side, base = None, joint
            groups.setdefault(base, {})[side] = score

        warnings = []
        for base, sides in sorted(groups.items(), key=lambda x: min(x[1].values())):
            tip = None
            for key, msg in _JOINT_TIPS.items():
                if key in base:
                    tip = msg
                    break
            formatted_base = base.replace('_', ' ')
            if tip is None:
                tip = f'Your {formatted_base} positioning needs improvement.'

            if 'left' in sides and 'right' in sides:
                name = formatted_base + 's'
                label = f"{sides['left']}% (left) / {sides['right']}% (right)"
            else:
                side, score = next(iter(sides.items()))
                name = f"{side} {formatted_base}" if side else formatted_base
                label = f"{score}%"

            name = name[0].upper() + name[1:]
            warnings.append(f"⚠ {name} scored {label}. {tip}")

        return warnings

    @staticmethod
    def _trajectory_warning(trajectory_score: float, threshold: float) -> str | None:
        """Warn if floor movement direction/path doesn't match."""
        if trajectory_score < threshold:
            return (
                f"⚠ Trajectory score: {trajectory_score}%. "
                "Your floor movement path differs from the teacher's — "
                "focus on moving in the same direction and covering similar ground."
            )
        return None

    @staticmethod
    def _shape_warning(mask_score: float, threshold: float) -> str | None:
        """Warn about body silhouette differences."""
        if mask_score < threshold:
            return (
                f"⚠ Silhouette score: {mask_score}%. "
                "Your overall body shape differs from the teacher's — "
                "check if your limbs are extended/contracted to the same degree."
            )
        return None

    @staticmethod
    def _energy_mismatch(energy_details: dict, config) -> str | None:
        """Detect if the student is consistently too slow or too fast."""
        teacher_energy = energy_details.get('teacher_energy')
        student_energy = energy_details.get('student_energy')

        if teacher_energy is None or student_energy is None:
            return None
        if len(teacher_energy) == 0:
            return None

        active = teacher_energy > 1e-3
        if not active.any():
            return None

        ratio = np.mean(student_energy[active]) / np.mean(teacher_energy[active])

        if ratio < config.energy_low_threshold:
            return (
                "⚠ Your movements appear less energetic than the teacher's — "
                "try to use more power and bigger motions."
            )
        elif ratio > config.energy_high_threshold:
            return (
                "⚠ Your movements appear more exaggerated than the teacher's — "
                "try to control your motion and match the teacher's intensity."
            )
        return None

    @staticmethod
    def _timing_warning(timing_cost: float | None, config) -> str | None:
        """Feedback on rhythm/sync based on normalized DTW cost."""
        if timing_cost is None:
            return None
        if timing_cost >= config.timing_warn_threshold:
            return (
                "⚠ Your timing seems off — "
                "try to stay in sync with the teacher's rhythm."
            )
        if timing_cost <= config.timing_praise_threshold:
            return "✓ Great timing — you stayed well in sync with the teacher!"
        return None

    @staticmethod
    def _praise(
        skeleton_score: float,
        trajectory_score: float,
        mask_score: float,
        threshold: float,
    ) -> list[str]:
        """Compliment components that scored above the praise threshold."""
        messages = []
        if skeleton_score >= threshold:
            messages.append(f"✓ Great joint accuracy! Skeleton score: {skeleton_score}%.")
        if trajectory_score >= threshold:
            messages.append(f"✓ Excellent floor movement! Trajectory score: {trajectory_score}%.")
        if mask_score >= threshold:
            messages.append(f"✓ Body shape closely matches the teacher! Mask score: {mask_score}%.")
        return messages

    @staticmethod
    def _format_joint(name: str) -> str:
        """Format a joint name for display: 'right_elbow' -> 'Right Elbow'."""
        return name.replace('_', ' ')
