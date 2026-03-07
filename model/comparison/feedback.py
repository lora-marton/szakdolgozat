"""
Rule-based feedback generation for dance comparison results.

Analyses scores from the comparison pipeline and produces a prioritised
list of plain-text messages — from high-level summary down to
component-specific tips and positive reinforcement.
"""


def generate_feedback(results, config):
    """
    Generate human-readable feedback from comparison results.

    Args:
        results: Dict returned by compare_dances(), containing:
            - overall_score, skeleton_score, trajectory_score, mask_score
            - per_joint_scores: dict {joint_name: score 0-100}
            - worst_frames: list of (frame_idx, joint_name, error_degrees)
            - alignment_path, timing_cost
            - per_frame_shape: array (N,) — DTM shape scores per frame
            - energy_details: dict with per_frame_ratios, teacher_energy, student_energy
        config: ComparisonConfig instance with feedback thresholds.

    Returns:
        feedback: List of str — prioritised feedback messages.
    """
    feedback = []

    overall = results['overall_score']
    skeleton = results['skeleton_score']
    trajectory = results['trajectory_score']
    mask = results['mask_score']

    # ── 1. Overall summary ───────────────────────────────────────────
    feedback.append(_overall_summary(overall))

    # ── 2. Joint-specific warnings ───────────────────────────────────
    per_joint = results.get('per_joint_scores', {})
    joint_warnings = _joint_warnings(per_joint, config.feedback_joint_warn_threshold)
    feedback.extend(joint_warnings)

    # ── 3. Worst moment highlight ────────────────────────────────────
    worst_frames = results.get('worst_frames', [])
    worst_msg = _worst_moment(worst_frames)
    if worst_msg:
        feedback.append(worst_msg)

    # ── 4. Trajectory warning ────────────────────────────────────────
    traj_msg = _trajectory_warning(trajectory, config.feedback_direction_warn_threshold)
    if traj_msg:
        feedback.append(traj_msg)

    # ── 5. Silhouette / shape warning ────────────────────────────────
    shape_msg = _shape_warning(mask, config.feedback_mask_warn_threshold)
    if shape_msg:
        feedback.append(shape_msg)

    # ── 6. Energy mismatch ───────────────────────────────────────────
    energy_details = results.get('energy_details', {})
    energy_msg = _energy_mismatch(energy_details)
    if energy_msg:
        feedback.append(energy_msg)

    # ── 7. Positive reinforcement ────────────────────────────────────
    praise = _praise(skeleton, trajectory, mask, config.feedback_praise_threshold)
    feedback.extend(praise)

    return feedback


# ── Helper rule functions ────────────────────────────────────────────────


def _overall_summary(score):
    """One-liner summary based on the overall score."""
    if score >= 90:
        return f"Excellent performance! Overall score: {score}%."
    elif score >= 75:
        return f"Good performance with room for improvement. Overall score: {score}%."
    elif score >= 55:
        return f"Decent attempt — several areas need work. Overall score: {score}%."
    else:
        return f"This needs more practice. Overall score: {score}%."


def _joint_warnings(per_joint_scores, threshold):
    """Flag joints that scored below the warning threshold."""
    _JOINT_TIPS = {
        'elbows':    'Focus on matching the bend of your elbows — keep them sharper or softer as needed.',
        'knees':     'Pay attention to your knee bend — try to match the depth of the teacher\'s plié or stance.',
        'shoulders': 'Watch your shoulder positioning — they may be too raised or too low.',
        'wrists':    'Your wrist angles differ — check if your hands are angled differently from the teacher.',
        'hips':      'Your hip angles are off — focus on the tilt and rotation of your pelvis.',
        'ankles':    'Your ankle positioning differs — pay attention to foot placement and flexion.',
    }
    warnings = []
    for joint, score in sorted(per_joint_scores.items(), key=lambda x: x[0]):
        if score < threshold:
            tip = _JOINT_TIPS.get(joint, f'Your {joint} positioning needs improvement.')
            warnings.append(f"⚠ {joint.capitalize()} scored {score}%. {tip}")
    return warnings


def _worst_moment(worst_frames):
    """Highlight the single worst frame."""
    if not worst_frames:
        return None

    # worst_frames is sorted by score (lowest first), so index 0 is the worst
    frame_idx, joint_name, error_deg = worst_frames[0]
    return (
        f"Your biggest deviation was at frame {frame_idx}: "
        f"{joint_name} was off by {error_deg}°."
    )


def _trajectory_warning(trajectory_score, threshold_pct):
    """Warn if floor movement direction/path doesn't match."""
    # threshold_pct is 0-1 scale, trajectory_score is 0-100
    if trajectory_score < threshold_pct * 100:
        return (
            f"⚠ Trajectory score: {trajectory_score}%. "
            "Your floor movement path differs from the teacher's — "
            "focus on moving in the same direction and covering similar ground."
        )
    return None


def _shape_warning(mask_score, threshold):
    """Warn about body silhouette differences."""
    if mask_score < threshold:
        return (
            f"⚠ Silhouette score: {mask_score}%. "
            "Your overall body shape differs from the teacher's — "
            "check if your limbs are extended/contracted to the same degree."
        )
    return None


def _energy_mismatch(energy_details):
    """Detect if the student is consistently too slow or too fast."""
    import numpy as np

    teacher_energy = energy_details.get('teacher_energy')
    student_energy = energy_details.get('student_energy')

    if teacher_energy is None or student_energy is None:
        return None
    if len(teacher_energy) == 0:
        return None

    # Only look at frames where the teacher is actually moving
    active = teacher_energy > 1e-3
    if not active.any():
        return None

    ratio = np.mean(student_energy[active]) / np.mean(teacher_energy[active])

    if ratio < 0.6:
        return (
            "⚠ Your movements appear less energetic than the teacher's — "
            "try to use more power and bigger motions."
        )
    elif ratio > 1.6:
        return (
            "⚠ Your movements appear more exaggerated than the teacher's — "
            "try to control your motion and match the teacher's intensity."
        )
    return None


def _praise(skeleton_score, trajectory_score, mask_score, threshold):
    """Compliment components that scored above the praise threshold."""
    messages = []
    if skeleton_score >= threshold:
        messages.append(f"✓ Great joint accuracy! Skeleton score: {skeleton_score}%.")
    if trajectory_score >= threshold:
        messages.append(f"✓ Excellent floor movement! Trajectory score: {trajectory_score}%.")
    if mask_score >= threshold:
        messages.append(f"✓ Body shape closely matches the teacher! Mask score: {mask_score}%.")
    return messages
