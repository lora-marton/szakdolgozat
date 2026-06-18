# Preprocessor Module — Audio Sync, Motion Trimming & Intersection

Add a preprocessing step that synchronises and trims two dance sequences before DTW alignment. Following the extractor pattern, the logic is split into focused helper modules, with a single `preprocessor.py` orchestrator that collects them.

## Proposed Changes

### Audio Synchronisation

#### [NEW] [audio_sync.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/audio_sync.py)

Provides `compute_audio_offset(video1_path, video2_path, sr=22050) → offset_frames`.

- Loads audio from both videos with `librosa`.
- Cross-correlates the two signals with `scipy.signal.correlate`.
- Returns the lag converted to **frames** (using the target FPS from config), so the rest of the pipeline works in frame indices.

---

### Motion Energy Detection

#### [NEW] [motion_energy.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/motion_energy.py)

Provides:
- `compute_motion_energy(landmarks) → energy` — per-frame motion energy from landmark deltas.
- `find_active_range(energy, threshold_ratio, min_duration_frames) → (start, end)` — first/last frame of sustained movement.

---

### Preprocessor Orchestrator

#### [NEW] [preprocessor.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/preprocessor.py)

Single entry-point function:

```python
def preprocess(teacher_data, student_data, teacher_video, student_video, config=None):
    """
    1. Audio cross-correlation  → compute frame offset between videos
    2. Apply offset              → shift/trim the leading video
    3. Motion energy detection   → find active range in each sequence
    4. Intersection              → keep only the overlapping active region
    Returns trimmed copies of teacher_data and student_data dicts
    (landmarks, masks, trajectory — all sliced to the active intersection).
    """
```

#### Pipeline detail

| Step | Input | Output |
|------|-------|--------|
| Audio offset | two video paths | `offset_frames: int` |
| Apply offset | both data dicts + offset | trimmed-to-shorter data dicts |
| Motion energy | each landmarks array | [(start, end)](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/video_processor.py#12-16) per dancer |
| Intersection | two [(start, end)](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/video_processor.py#12-16) ranges | single [(start, end)](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/video_processor.py#12-16) shared range |
| Final slice | data dicts + shared range | final trimmed data dicts |

---

### Configuration

#### [MODIFY] [config.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py)

Add a `PreprocessorConfig` dataclass with sensible defaults:

```python
@dataclass(frozen=True)
class PreprocessorConfig:
    audio_sample_rate: int = 22050
    motion_threshold_ratio: float = 0.15   # fraction of max energy
    min_active_duration: int = 10          # frames of sustained motion
```

Add `DEFAULT_PREPROCESSOR_CONFIG` instance.

---

### Integration

#### [MODIFY] [comparator.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparator.py)

Insert the preprocessor call **between** data loading and DTW alignment:

```diff
 teacher_data = _load_session_data(output_dir, 'teacher')
 student_data = _load_session_data(output_dir, 'student')

+# --- Phase 0: Preprocessing (sync + trim) ---
+teacher_data, student_data = preprocess(
+    teacher_data, student_data,
+    teacher_video_path, student_video_path,
+    config.preprocessor,
+)

 # --- Phase A: Temporal Alignment (DTW) ---
```

[compare_dances](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparator.py#18-99) will need two additional parameters for the video file paths (needed for audio extraction). These will be passed through from [video_processor.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/video_processor.py):

#### [MODIFY] [video_processor.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/video_processor.py)

Pass the video paths to [compare_dances](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparator.py#18-99) so the preprocessor can extract audio.

## Verification Plan

### Manual Verification

Since there are no automated tests in the project, verification will be done manually:

1. **Unit-level sanity check** — I will create a small scratch script at `/tmp/test_preprocessor.py` that generates synthetic landmark data with known idle periods and verifies the trimming logic produces the expected frame ranges.
2. **I'd like your input on testing with real videos** — Do you want me to test against the existing videos in [videos/](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/video_processor.py#7-41) (e.g., [test_dance1.mp4](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/videos/test_dance1.mp4), [test_dance2.mp4](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/videos/test_dance2.mp4))? If so, I can run a quick end-to-end test through [video_processor.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/video_processor.py) to confirm nothing breaks.
