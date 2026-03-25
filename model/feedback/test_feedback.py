"""
Verification test for feedback generation.

Tests several scenarios with synthetic comparison results to ensure
the rule-based feedback generator produces expected messages.
"""
import numpy as np
from model.feedback.feedback import generate_feedback
from model.config import DEFAULT_COMPARISON_CONFIG


def _make_results(**overrides):
    """Create a baseline results dict with perfect scores, then apply overrides."""
    base = {
        'overall_score': 95.0,
        'skeleton_score': 95.0,
        'trajectory_score': 95.0,
        'mask_score': 95.0,
        'timing_cost': 0.1,
        'alignment_path': [],
        'per_joint_scores': {
            'elbows': 95.0, 'knees': 95.0, 'shoulders': 95.0,
            'wrists': 95.0, 'hips': 95.0, 'ankles': 95.0,
        },
        'worst_frames': [(10, 'elbows', 5.0)],
        'per_frame_shape': np.ones(100, dtype=np.float32),
        'energy_details': {
            'energy_score': 0.95,
            'per_frame_ratios': np.ones(99, dtype=np.float32) * 0.95,
            'teacher_energy': np.ones(99, dtype=np.float32) * 5.0,
            'student_energy': np.ones(99, dtype=np.float32) * 4.8,
        },
    }
    base.update(overrides)
    return base


def test_excellent_performance():
    """All scores high → overall praise, no warnings."""
    results = _make_results()
    feedback = generate_feedback(results, DEFAULT_COMPARISON_CONFIG)

    print("=== Excellent Performance ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('Excellent' in m for m in feedback), "Should have excellent summary"
    assert any('✓' in m for m in feedback), "Should have praise messages"
    assert not any('⚠' in m for m in feedback), "Should have no warnings"
    print("  ✅ PASSED\n")


def test_poor_joint():
    """One joint scores badly → that joint should be flagged."""
    results = _make_results(
        overall_score=72.0,
        skeleton_score=65.0,
        per_joint_scores={
            'elbows': 40.0, 'knees': 90.0, 'shoulders': 85.0,
            'wrists': 80.0, 'hips': 80.0, 'ankles': 85.0,
        },
    )
    feedback = generate_feedback(results, DEFAULT_COMPARISON_CONFIG)

    print("=== Poor Elbow Score ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('Elbows' in m and '⚠' in m for m in feedback), "Should warn about elbows"
    assert not any('Knees' in m and '⚠' in m for m in feedback), "Knees are fine"
    print("  ✅ PASSED\n")


def test_trajectory_warning():
    """Low trajectory score → trajectory warning."""
    results = _make_results(
        overall_score=60.0,
        trajectory_score=50.0,
    )
    feedback = generate_feedback(results, DEFAULT_COMPARISON_CONFIG)

    print("=== Low Trajectory ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('Trajectory' in m and '⚠' in m for m in feedback), "Should warn about trajectory"
    print("  ✅ PASSED\n")


def test_energy_too_low():
    """Student energy much lower than teacher → energy warning."""
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
    feedback = generate_feedback(results, DEFAULT_COMPARISON_CONFIG)

    print("=== Low Energy ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('less energetic' in m for m in feedback), "Should flag low energy"
    print("  ✅ PASSED\n")


def test_energy_too_high():
    """Student energy much higher than teacher → exaggerated warning."""
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
    feedback = generate_feedback(results, DEFAULT_COMPARISON_CONFIG)

    print("=== High Energy ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('exaggerated' in m for m in feedback), "Should flag high energy"
    print("  ✅ PASSED\n")


def test_worst_moment():
    """Worst frame should be highlighted."""
    results = _make_results(
        worst_frames=[(42, 'knees', 55.3), (10, 'elbows', 30.0)],
    )
    feedback = generate_feedback(results, DEFAULT_COMPARISON_CONFIG)

    print("=== Worst Moment ===")
    for msg in feedback:
        print(f"  {msg}")

    assert any('frame 42' in m and 'knees' in m for m in feedback), "Should highlight worst frame"
    print("  ✅ PASSED\n")


if __name__ == '__main__':
    test_excellent_performance()
    test_poor_joint()
    test_trajectory_warning()
    test_energy_too_low()
    test_energy_too_high()
    test_worst_moment()
    print("🎉 All tests passed!")
