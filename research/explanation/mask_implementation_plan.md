# Mask Comparison Pipeline: EFD → DTM → Optical Flow

Replace the [compute_iou](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py#7-21) stub in [mask_metrics.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py) with a three-stage pipeline that compares dancer silhouettes while being robust to clothing differences. Based on research in [notes.txt](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/research/notes.txt) (lines 603–720).

## Library Overview

### 1. `pyefd` — Elliptic Fourier Descriptors

**What it does:** Decomposes a closed contour (the mask outline) into a series of sine/cosine harmonics. Low harmonics = overall body shape; high harmonics = clothing wrinkles/noise.

**Key function:**
```python
from pyefd import elliptic_fourier_descriptors, reconstruct_contour

# Get EFD coefficients (n harmonics × 4 values each)
coeffs = elliptic_fourier_descriptors(contour, order=8, normalize=True)

# Reconstruct a smoothed contour from those coefficients
smoothed = reconstruct_contour(coeffs, locus=(cx, cy), num_points=200)
```

- `contour`: Nx2 array of (x,y) points from `cv2.findContours`
- `order=8`: Number of harmonics to keep (your notes suggest 5–8)
- `normalize=True`: Makes coefficients invariant to rotation/scale/starting point
- The `reconstruct_contour` function rebuilds a clean outline from the coefficients

**Why this library:** It's the standard Python EFD implementation, lightweight (~50 KB), pure NumPy under the hood. No alternative exists with comparable quality.

**New dependency:** `pyefd` (not in [requirements.txt](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/requirements.txt) yet)

---

### 2. `scipy.ndimage.distance_transform_edt` — Distance Transform

**What it does:** Converts a binary mask into a "heat map" where each pixel's value = its Euclidean distance to the nearest edge. Center of a limb = high value, edges = low.

**Key function:**
```python
from scipy.ndimage import distance_transform_edt

# Binary mask → distance map
dist_map = distance_transform_edt(binary_mask)

# Apply Gaussian weighting: center = 1.0, edges → 0
dtm = np.exp(-(dist_map ** 2) / (2 * sigma ** 2))
```

- The raw EDT gives distances in pixels
- We apply Gaussian decay so pixels near the center score higher
- When comparing: multiply student's binary mask by teacher's Gaussian DTM → partial credit for "close but not exact" overlap

**Why this library:** Already in your [requirements.txt](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/requirements.txt) (`scipy`). The `distance_transform_edt` function is highly optimized C code — handles 256×256 masks in <1ms.

---

### 3. `cv2.calcOpticalFlowFarneback` — Dense Optical Flow

**What it does:** Computes a motion vector (dx, dy) for *every* pixel between two consecutive grayscale frames. We use this inside the mask to measure "energy" (how hard the dancer hits a move).

**Key function:**
```python
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, curr_gray,
    flow=None,
    pyr_scale=0.5,    # image pyramid scale
    levels=3,          # pyramid levels
    winsize=15,        # averaging window size
    iterations=3,      # refinement iterations
    poly_n=5,          # polynomial expansion neighborhood
    poly_sigma=1.2,    # Gaussian for polynomial smoothing
    flags=0,
)
# flow.shape = (H, W, 2) — dx, dy per pixel
magnitude = np.linalg.norm(flow, axis=-1)  # speed per pixel
```

- Computed on full frames, then **masked** to only measure pixels inside the segmentation mask
- Average magnitude within mask = "energy" of that frame's movement
- Compare teacher vs student energy profiles across frames

**Why this library:** Already in your [requirements.txt](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/requirements.txt) (`opencv-python`). Farneback is the standard dense flow algorithm — good balance of speed and accuracy for 256×256 masks.

---

## Proposed Changes

### Dependencies

#### [MODIFY] [requirements.txt](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/requirements.txt)

Add `pyefd` to the dependency list.

---

### Configuration

#### [MODIFY] [config.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py)

Add mask-specific parameters to [ComparisonConfig](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py#48-111):

| Parameter | Default | Purpose |
|---|---|---|
| `mask_binary_threshold` | `128` | Threshold for binarizing uint8 masks |
| `efd_harmonics` | `8` | Number of EFD harmonics (low = smoother) |
| `efd_contour_points` | `200` | Points in the reconstructed contour |
| `dtm_sigma` | `10.0` | Gaussian decay width for distance transform (pixels) |
| `flow_winsize` | `15` | Farneback window size |
| `weight_shape` | `0.60` | Shape sub-weight (DTM score) within mask's 20% |
| `weight_energy` | `0.40` | Energy sub-weight (flow score) within mask's 20% |

---

### Mask Metrics Implementation

#### [MODIFY] [mask_metrics.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py)

Replace the [compute_iou](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py#7-21) stub with:

**`smooth_mask_efd(mask, n_harmonics, n_points)`**
- Binarize mask → find largest contour → compute EFD coefficients → reconstruct smooth contour → fill to create clean binary mask
- Returns: smoothed binary mask (same size as input)

**`compare_shapes_dtm(teacher_masks, student_masks, sigma)`**
- For each frame: generate Gaussian-weighted DTM from teacher's smoothed mask → multiply by student's smoothed mask → calculate overlap score
- Skip frames where both masks are empty
- Returns: `mean_score` (0–1), `per_frame_scores` array

**`compare_mask_energy(teacher_masks, student_masks, winsize)`**
- For each consecutive frame pair: compute Farneback optical flow → mask flow to only dancer pixels → calculate mean magnitude
- Compare teacher vs student energy using `min/max` ratio (same approach as trajectory speed similarity)
- Skip frame pairs where both have near-zero energy
- Returns: `energy_score` (0–1)

**`compute_mask_score(teacher_masks, student_masks, config)`**
- Top-level orchestrator replacing [compute_iou](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py#7-21)
- Calls EFD smoothing → DTM comparison → flow energy comparison
- Combines: `score = weight_shape * dtm_score + weight_energy * energy_score`
- Returns: `score` (0–100), `per_frame_dtm` array

---

### Comparator Integration

#### [MODIFY] [comparator.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/comparator.py)

- Replace [compute_iou(...)](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/mask_metrics.py#7-21) call with `compute_mask_score(..., config)`
- Remove the `mask_score * 100` conversion (new function returns 0–100 directly)
- Update the import

---

## Verification Plan

### Automated Test

A synthetic data test script that validates the pipeline end-to-end without real videos:

```
python -c "
import numpy as np
from model.comparison.mask_metrics import compute_mask_score
from model.config import DEFAULT_COMPARISON_CONFIG as cfg

# Create synthetic 256x256 masks (10 frames)
# Teacher: circle moving right over 10 frames
# Student: same circle, slightly offset
masks_t = np.zeros((10, 256, 256), dtype=np.uint8)
masks_s = np.zeros((10, 256, 256), dtype=np.uint8)
for i in range(10):
    cv2.circle(masks_t[i], (100 + i*10, 128), 40, 255, -1)
    cv2.circle(masks_s[i], (105 + i*10, 130), 38, 255, -1)

score, per_frame = compute_mask_score(masks_t, masks_s, cfg)
print(f'Score: {score}, frames: {len(per_frame)}')
assert 50 < score < 100, 'Similar masks should score high'
assert len(per_frame) == 10

# Identical masks should score ~100
score2, _ = compute_mask_score(masks_t, masks_t, cfg)
assert score2 > 95, f'Identical masks should score near 100, got {score2}'
print('All assertions passed!')
"
```

### Manual Verification

> [!IMPORTANT]
> Since there are no existing unit tests in the project, I suggest the user later tests with real HDF5 data from a session directory. This would confirm the full [comparator.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/comparator.py) pipeline works end-to-end with real masks.
