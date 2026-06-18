# Preprocessor Module — Walkthrough

## What Was Built

A preprocessing pipeline that **synchronises and trims** two dance videos before DTW alignment, removing idle time at the start/end.

### New Files

| File | Purpose |
|------|---------|
| [audio_sync.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/audio_sync.py) | Cross-correlates audio tracks → returns frame offset |
| [motion_energy.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/motion_energy.py) | Detects active dance range via landmark displacement |
| [preprocessor.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/preprocessor.py) | Orchestrator: audio sync → offset → motion trim → intersection |

### Modified Files

| File | Change |
|------|--------|
| [config.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py) | Added [PreprocessorConfig](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py#105-115) dataclass + default instance |
| [comparator.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparator.py) | Added Phase 0 (preprocessing) before DTW; added `teacher_video`/`student_video` params |
| [video_processor.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/video_processor.py) | Passes video paths through to [compare_dances](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparator.py#19-111) |

## Pipeline Flow

```
Video paths ─→ Audio cross-correlation ─→ frame offset
                                              │
Landmark data ─────────────────────────→ Apply offset (trim leading video)
                                              │
                                       Motion energy detection
                                       (per-dancer active range)
                                              │
                                       Take intersection
                                              │
                                       Trimmed data ─→ DTW ─→ Comparison
```

## Verification

- ✅ All imports pass successfully
- ✅ [PreprocessorConfig](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py#105-115) loads with correct defaults (`sr=22050`, `threshold=0.15`, `min_dur=10`)
- ✅ Synthetic motion energy test: 100 frames with idle(0–19), active(20–79), idle(80–99) → detected range **16–83** (expected ~19–80, boundary fuzz is normal)
- ✅ `librosa` dependency installed
