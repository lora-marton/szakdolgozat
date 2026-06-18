"""
Unit tests for the Preprocessor helper methods.

Covers array slicing, offset application in both directions, and HDF5
session loading from a temporary directory.
"""

import logging
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests  # noqa: F401

from model.preprocessing.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


def _make_data(n: int, offset: float = 0.0) -> dict:
    """Build a small data dict with landmarks, masks, trajectory + scalars."""
    return {
        "landmarks": np.arange(n * 33 * 4, dtype=np.float32).reshape(n, 33, 4) + offset,
        "masks": np.zeros((n, 32, 32), dtype=np.uint8),
        "trajectory": np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        "fps": 60.0,
        "fixed_scale": 1.25,
    }


def test_slice_data_trims_arrays_and_preserves_scalars():
    """_slice_data should slice numpy arrays but leave scalars untouched."""
    data = _make_data(10)

    sliced = Preprocessor._slice_data(data, 2, 5)

    logger.info("=== Slice Data ===")
    logger.info("  landmarks: %s, fps: %s", sliced["landmarks"].shape, sliced["fps"])
    assert sliced["landmarks"].shape == (3, 33, 4)
    assert sliced["masks"].shape == (3, 32, 32)
    assert sliced["trajectory"].shape == (3, 2)
    assert sliced["fps"] == 60.0
    assert sliced["fixed_scale"] == 1.25
    logger.info("  PASSED\n")


def test_slice_data_end_none_slices_to_end():
    """Passing end=None should slice from start to the array end."""
    data = _make_data(7)

    sliced = Preprocessor._slice_data(data, 3, None)

    logger.info("=== Slice Data End None ===")
    logger.info("  landmarks shape: %s", sliced["landmarks"].shape)
    assert sliced["landmarks"].shape == (4, 33, 4)
    logger.info("  PASSED\n")


def test_apply_offset_positive_trims_teacher():
    """A positive offset trims the leading frames from the teacher."""
    teacher = _make_data(10)
    student = _make_data(10, offset=1000.0)

    t_out, s_out = Preprocessor._apply_offset(teacher, student, offset=3)

    logger.info("=== Apply Offset Positive ===")
    logger.info("  teacher: %s, student: %s", t_out["landmarks"].shape, s_out["landmarks"].shape)
    assert t_out["landmarks"].shape[0] == 7
    assert s_out["landmarks"].shape[0] == 7
    assert np.isclose(t_out["landmarks"][0, 0, 0], 3 * 33 * 4)
    logger.info("  PASSED\n")


def test_apply_offset_negative_trims_student():
    """A negative offset trims the leading frames from the student."""
    teacher = _make_data(10)
    student = _make_data(10, offset=1000.0)

    t_out, s_out = Preprocessor._apply_offset(teacher, student, offset=-2)

    logger.info("=== Apply Offset Negative ===")
    logger.info("  teacher: %s, student: %s", t_out["landmarks"].shape, s_out["landmarks"].shape)
    assert t_out["landmarks"].shape[0] == 8
    assert s_out["landmarks"].shape[0] == 8
    assert np.isclose(s_out["landmarks"][0, 0, 0], 1000.0 + 2 * 33 * 4)
    logger.info("  PASSED\n")


def test_apply_offset_zero_length_matches():
    """A zero offset still truncates to the shorter sequence length."""
    teacher = _make_data(12)
    student = _make_data(8)

    t_out, s_out = Preprocessor._apply_offset(teacher, student, offset=0)

    logger.info("=== Apply Offset Zero ===")
    logger.info("  teacher: %s, student: %s", t_out["landmarks"].shape, s_out["landmarks"].shape)
    assert t_out["landmarks"].shape[0] == 8
    assert s_out["landmarks"].shape[0] == 8
    logger.info("  PASSED\n")


class TestLoadSessionData:
    """Tests for _load_session_data against real HDF5 files."""

    def setup_method(self):
        """Create a temp dir with synthetic teacher HDF5 files."""
        self.tmpdir = tempfile.mkdtemp(prefix="test_preproc_")
        self.n = 5

        landmarks = np.random.rand(self.n, 33, 4).astype(np.float32)
        trajectory = np.random.rand(self.n, 2).astype(np.float32)
        with h5py.File(os.path.join(self.tmpdir, "teacher_data.h5"), "w") as f:
            f.create_dataset("raw", data=landmarks)
            f.create_dataset("trajectory", data=trajectory)
            f.attrs["fps"] = 30.0
            f.attrs["fixed_scale"] = 2.5

        masks = np.zeros((self.n, 16, 16), dtype=np.uint8)
        with h5py.File(os.path.join(self.tmpdir, "teacher_masks.h5"), "w") as f:
            f.create_dataset("masks", data=masks)

    def teardown_method(self):
        """Remove the temp dir after each test."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_session_data_reads_all_fields(self):
        """All arrays and scalar attributes should be read back correctly."""
        data = Preprocessor._load_session_data(self.tmpdir, "teacher")

        logger.info("=== Load Session Data ===")
        logger.info("  keys: %s", sorted(data.keys()))
        assert data["landmarks"].shape == (self.n, 33, 4)
        assert data["trajectory"].shape == (self.n, 2)
        assert data["masks"].shape == (self.n, 16, 16)
        assert data["fps"] == 30.0
        assert data["fixed_scale"] == 2.5
        logger.info("  PASSED\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    test_slice_data_trims_arrays_and_preserves_scalars()
    test_slice_data_end_none_slices_to_end()
    test_apply_offset_positive_trims_teacher()
    test_apply_offset_negative_trims_student()
    test_apply_offset_zero_length_matches()
    t = TestLoadSessionData()
    t.setup_method()
    try:
        t.test_load_session_data_reads_all_fields()
    finally:
        t.teardown_method()
    logger.info("All tests passed!")
