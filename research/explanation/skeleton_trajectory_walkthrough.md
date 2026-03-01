# Walkthrough — Preprocessor, Skeleton & Trajectory Metrics

## Preprocessor Module

Created `preprocessing/` sub-package: audio cross-correlation for sync, motion energy for trim, intersection for shared range.

---

## Skeleton Metrics

### Joint Angles (2D)

For each [(parent, joint, child)](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/controller/fileGetter.py#55-58) triplet, we compute the angle at [joint](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/dtw.py#51-64):

```
vec_a = parent_pos - joint_pos
vec_b = child_pos - joint_pos
angle = arccos(dot(a, b) / (|a| × |b|))
```

Uses only x, y (2D) since the normalization pipeline is 2D-focused and MediaPipe's z is less reliable.

### Center of Gravity

Weighted average of joint positions using biomechanical segment weights (defined in config). Captures balance and weight transfer differences.

### Exponential Decay Scoring

```
if error ≤ tolerance:  score = 100
else:                  score = 100 × exp(-((error - tol) / σ)²)
```

- Small overshoot → gentle penalty (style variation)
- Large overshoot → harsh penalty (clear mistake)
- `σ = 25°` for angles, `σ = 0.05` for CoG (normalized coords)

Skeleton score = 80% angle score + 20% CoG score.

### Verification

| Test | Expected | Got |
|------|----------|-----|
| 90° elbow | 90.0° | 90.0° |
| Identical sequences | 100 | 100 |
| 80° error | near 0 | 0.2 |
| CoG shift 3% | moderate | 48.7 |

---

## Trajectory Metrics

### The Problem

Two dancers in different rooms with different cameras — absolute positions are meaningless. We compare **movement patterns**: are they moving in the same direction at the same speed?

### Method

[trajectory_metrics.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/trajectory_metrics.py) computes frame-to-frame velocity vectors from hip positions, then scores two aspects:

#### 1. Direction Similarity (75% weight)

Per-frame **cosine similarity** between teacher and student velocity vectors:

```
cosine = dot(teacher_vel, student_vel) / (|teacher_vel| × |student_vel|)
```

Cosine similarity produces values in `[-1, 1]`, which we map to `[0, 1]`:

| Cosine | Mapped | Meaning |
|--------|--------|---------|
| 1.0 | 1.0 | Same direction |
| 0.0 | 0.5 | Perpendicular |
| -1.0 | 0.0 | Opposite direction |

When **one dancer is stationary but the other moves**, direction score = 0 (they should be moving too).

#### 2. Speed Similarity (25% weight)

Per-frame **ratio** of velocity magnitudes:

```
speed_score = min(|teacher_vel|, |student_vel|) / max(|teacher_vel|, |student_vel|)
```

| Ratio | Meaning |
|-------|---------|
| 1.0 | Same speed |
| 0.5 | One moves twice as fast |
| 0.0 | One is stationary |

#### 3. Stationary Frame Filtering

Frames where **both** dancers are nearly still (velocity below threshold) are excluded entirely. This avoids noisy cosine similarity from tiny floating-point velocity vectors when nobody is moving.

#### Combined Score

```
trajectory_score = (0.75 × mean_direction + 0.25 × mean_speed) × 100
```

### Verification

| Test | Score | Direction | Explanation |
|------|-------|-----------|-------------|
| Identical paths | 100.0 | 1.0 | Perfect match |
| Reversed path | 56.7 | 0.55 | Mostly opposite movement |
| Perpendicular | 62.5 | 0.5 | Right vs up → cosine = 0 → maps to 0.5 |
| Both stationary | 100.0 | 1.0 | All frames filtered → treated as match |
