# Feedback Generation — Walkthrough

## Changes Made

| File | Change |
|------|--------|
| [feedback.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/feedback.py) | **[NEW]** — 7 rule-based feedback categories in [generate_feedback()](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/feedback.py#10-69) |
| [config.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py) | Added 4 tunable feedback thresholds to [ComparisonConfig](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py#48-134) |
| [comparator.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/comparator.py) | Removed `_generate_feedback` stub, added `worst_frames` to return dict |
| [video_processor.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/video_processor.py) | Added Phase 3: calls [generate_feedback()](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/feedback.py#10-69) after comparison, attaches to results |
| [test_feedback.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/test_feedback.py) | **[NEW]** — 6 test scenarios with synthetic data |

## Feedback Categories

1. **Overall summary** — "Excellent / Good / Decent / Needs practice" based on `overall_score`
2. **Joint warnings** — flags joints below 70% with specific tips per joint
3. **Worst moment** — highlights the single worst frame (joint + angle error)
4. **Trajectory** — warns if floor movement direction differs (< 70%)
5. **Silhouette** — warns if body shape score is low (< 60%)
6. **Energy mismatch** — detects if student is too slow (< 0.6× energy) or too fast (> 1.6×)
7. **Positive reinforcement** — praises components scoring ≥ 90%

## Test Results

All 6 scenarios pass: excellent performance, poor joint, trajectory warning, low energy, high energy, worst moment.
