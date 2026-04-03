"""
Verification test for feedback generation.

Tests several scenarios with synthetic comparison results to ensure
the rule-based feedback generator produces expected messages.
"""
import numpy as np
from model.feedback.text_feedback import TextFeedback


def _make_results(**overrides):
    """Create a baseline results dict with perfect scores, then apply overrides."""
    base = {
        'overall_score': 95.0,
        'skeleton_score': 95.0,
        'trajectory_score': 95.0,
        'mask_score': 95.0,
        'timing_cost': 0.1,
        'alignment_path': [(i, i) for i in range(100)],
        'per_joint_scores': {
            'left_elbow': 95.0, 'right_elbow': 95.0,
            'left_knee': 95.0, 'right_knee': 95.0,
            'left_shoulder': 95.0, 'right_shoulder': 95.0,
            'left_hip': 95.0, 'right_hip': 95.0,
            'left_wrist': 95.0, 'right_wrist': 95.0,
            'left_ankle': 95.0, 'right_ankle': 95.0,
            'left_inner_shoulder': 95.0, 'right_inner_shoulder': 95.0,
            'left_inner_hip': 95.0, 'right_inner_hip': 95.0,
        },
        'worst_frames': [(10, 'left_elbow', 5.0)],
        'per_frame_shape': np.ones(100, dtype=np.float32),
        'energy_details': {
            'energy_score': 0.95,
            'per_frame_ratios': np.ones(99, dtype=np.float32) * 0.95,
            'teacher_energy': np.ones(99, dtype=np.float32) * 5.0,
            'student_energy': np.ones(99, dtype=np.float32) * 4.8,
        },
        'preprocess_info': {'audio_offset': 0, 'student_offset': 0},
        'teacher_fps': 30.0,
        'student_fps': 30.0,
    }
    base.update(overrides)
    return base


def test_excellent_performance():
    """All scores high -> overall praise, no warnings."""
    results = _make_results()
    feedback = TextFeedback.generate_messages(results)

    print("=== Excellent Performance ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('Excellent' in m for m in feedback), "Should have excellent summary"
    assert any('Great joint accuracy' in m for m in feedback), "Should have praise messages"
    print("  PASSED\n")


def test_poor_joint():
    """One joint scores badly -> that joint should be flagged."""
    results = _make_results(
        overall_score=72.0,
        skeleton_score=65.0,
        per_joint_scores={
            'left_elbow': 40.0, 'right_elbow': 90.0,
            'left_knee': 90.0, 'right_knee': 90.0,
            'left_shoulder': 85.0, 'right_shoulder': 85.0,
            'left_hip': 80.0, 'right_hip': 80.0,
            'left_wrist': 80.0, 'right_wrist': 80.0,
            'left_ankle': 85.0, 'right_ankle': 85.0,
            'left_inner_shoulder': 85.0, 'right_inner_shoulder': 85.0,
            'left_inner_hip': 85.0, 'right_inner_hip': 85.0,
        },
    )
    feedback = TextFeedback.generate_messages(results)

    print("=== Poor Elbow Score ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('Left Elbow' in m and 'elbow' in m.lower() for m in feedback), "Should warn about left elbow"
    assert not any('Left Knee' in m for m in feedback), "Knees are fine"
    print("  PASSED\n")


def test_trajectory_warning():
    """Low trajectory score -> trajectory warning."""
    results = _make_results(
        overall_score=60.0,
        trajectory_score=50.0,
    )
    feedback = TextFeedback.generate_messages(results)

    print("=== Low Trajectory ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('Trajectory' in m for m in feedback), "Should warn about trajectory"
    print("  PASSED\n")


def test_energy_too_low():
    """Student energy much lower than teacher -> energy warning."""
    results = _make_results(
        overall_score=70.0,
        mask_score=55.0,
        energy_details={
            'energy_score': 0.4,
            'per_frame_ratios': np.ones(99, dtype=np.float32) * 0.3,
            'teacher_energy': np.ones(99, dtype=np.float32) * 10.0,
            'student_energy': np.ones(99, dtype=np.float32) * 2.0,
        },
    )
    feedback = TextFeedback.generate_messages(results)

    print("=== Low Energy ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('less energetic' in m for m in feedback), "Should flag low energy"
    print("  PASSED\n")


def test_energy_too_high():
    """Student energy much higher than teacher -> exaggerated warning."""
    results = _make_results(
        overall_score=70.0,
        mask_score=55.0,
        energy_details={
            'energy_score': 0.4,
            'per_frame_ratios': np.ones(99, dtype=np.float32) * 0.3,
            'teacher_energy': np.ones(99, dtype=np.float32) * 5.0,
            'student_energy': np.ones(99, dtype=np.float32) * 15.0,
        },
    )
    feedback = TextFeedback.generate_messages(results)

    print("=== High Energy ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('exaggerated' in m for m in feedback), "Should flag high energy"
    print("  PASSED\n")


def test_worst_moment():
    """Worst frame should be highlighted with timestamp."""
    results = _make_results(
        worst_frames=[(42, 'left_knee', 55.3), (10, 'left_elbow', 30.0)],
    )
    feedback = TextFeedback.generate_messages(results)

    print("=== Worst Moment ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('biggest deviation' in m and 'left knee' in m for m in feedback), "Should highlight worst moment"
    assert any('s:' in m for m in feedback), "Should use seconds, not frames"
    print("  PASSED\n")


if __name__ == '__main__':
    test_excellent_performance()
    test_poor_joint()
    test_trajectory_warning()
    test_energy_too_low()
    test_energy_too_high()
    test_worst_moment()
    print("All tests passed!")
