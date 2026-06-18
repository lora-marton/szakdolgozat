# Mask Comparison Pipeline — Walkthrough

## What Changed

Replaced the [compute_iou](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py#7-21) stub in [mask_metrics.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py) with a three-stage pipeline based on your research notes.

### Files Modified

| File | Change |
|---|---|
| [requirements.txt](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/requirements.txt) | Added `pyefd` |
| [config.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py) | Added 7 mask config params to [ComparisonConfig](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py#48-111) |
| [mask_metrics.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py) | Full rewrite: EFD + DTM + optical flow (320 lines) |
| [comparator.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/comparator.py) | Wired [compute_mask_score](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py#272-320), exposed per-frame data |

### Pipeline

```
Raw Mask → EFD smooth (8 harmonics) → DTM (Gaussian σ=10) → Shape Score
                                                                  ↓
Raw Mask → Optical Flow (Farneback) → Energy min/max ratio → Energy Score
                                                                  ↓
                            Combined: 60% shape + 40% energy → Mask Score (0-100)
```

### Key Functions in [mask_metrics.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py)

- [smooth_mask_efd()](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py#19-71) — Strips clothing noise from contours
- [compare_shapes_dtm()](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py#76-143) — Gaussian distance transform overlap scoring
- [compare_mask_energy()](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py#148-222) — Farneback optical flow energy comparison with min/max ratio
- [compute_mask_score()](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py#272-320) — Orchestrator combining all stages

All functions return **per-frame data** for future feedback use.

## Test Results

```
Test 1: Similar masks (circle with slight offset)
  Score: 67.1, Energy score: 0.9479 → PASSED

Test 2: Identical masks
  Score: 99.6 → PASSED

Test 3: Empty masks
  Score: 100.0 → PASSED

Test 4: Very different masks
  Score: 60.0 (lower than similar) → PASSED
```

## Bug Fixed

Fixed a NaN bug in [compare_shapes_dtm](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py#76-143) where filtering `per_frame_scores < 1.0` could produce an empty array. Replaced with a proper `active_mask` boolean tracker.
