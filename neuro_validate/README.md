# Neuro Validate

`neuro_validate/` is the real-data scientific validation layer for Emotiv Learn.

Its job is to answer one question:

> Does EEG improve cognitive-state estimation beyond behavior/task signals alone?

This module is intentionally separate from the tutoring simulator. The tutoring
demo is an application surface; `neuro_validate/` is the neuroscience anchor.

## Initial 3A Benchmark

The first benchmark targets `COG-BCI` N-Back workload estimation.

- dataset: COG-BCI
- task: N-Back
- sample unit: aligned EEG windows
- target: workload level
- feature families:
  - behavior/task only
  - EEG only
  - fused EEG + behavior/task
- evaluation: subject-held-out grouped cross-validation

## Deliverables

The benchmark should produce:

- `metrics.json`
- `summary.csv`
- a single bar chart comparing behavior-only vs EEG-only vs fused

## Module Layout

- `configs/`
  Benchmark configs.
- `src/schema.py`
  Canonical benchmark sample / result schemas.
- `src/ingest_cog_bci.py`
  Dataset indexing and metadata discovery.
- `src/align_nback_windows.py`
  Align EEG windows to N-Back task structure and labels.
- `src/eeg_features.py`
  Interpretable EEG feature extraction.
- `src/behavior_features.py`
  Non-neural baseline feature extraction.
- `src/datasets.py`
  Build matrices for behavior-only, EEG-only, and fused models.
- `src/models.py`
  Lightweight baseline models.
- `src/evaluate.py`
  Grouped evaluation and metric aggregation.
- `src/plots.py`
  Benchmark figures.
- `src/run_workload_benchmark.py`
  Main entrypoint.

## Important Framing

This module does **not** claim tutoring efficacy.

It validates a narrower but crucial premise:

> EEG contributes useful cognitive-state information beyond sparse observable
> non-neural signals.

