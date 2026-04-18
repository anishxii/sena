# System 3A Window Schema Notes

This note summarizes what we verified directly from the downloaded COG-BCI files on 2026-04-17.

## Dataset Layout

Each subject archive contains:

- `sub-XX/ses-S1`
- `sub-XX/ses-S2`
- `sub-XX/ses-S3`

Each session contains:

- `behavioral/*.mat`
- `eeg/*.set`
- `eeg/*.fdt`

For the N-back benchmark, the relevant task files are:

- EEG:
  - `zeroBACK.set` / `zeroBACK.fdt`
  - `oneBACK.set` / `oneBACK.fdt`
  - `twoBACK.set` / `twoBACK.fdt`
- behavior:
  - `0-Back.mat`
  - `1-Back.mat`
  - `2-Back.mat`

## What A Training Example Is

The current 3A benchmark is **not** a sequence model.

Each training example is a **single rolling EEG window** extracted from one continuous N-back recording.

Current config:

- window length: `4.0 s`
- window step: `2.0 s`
- sample rate: `500 Hz`

So each window spans:

- `2000` samples per channel

## Labels

We are currently using workload labels from the N-back condition itself:

- `ZeroBack -> 0`
- `OneBack -> 1`
- `TwoBack -> 2`

This makes 3A a workload classification benchmark.

## Real EEG Recording Properties

Verified from `sub-01 / ses-S1 / zeroBACK.set`:

- channels: `63`
- samples: `198288`
- trials: `1`
- sample rate: `500 Hz`
- duration: about `396.6 s`
- event rows: `203`

This confirms that each task is a continuous recording, not a pre-windowed dataset.

## Behavioral Side

The exported behavioral CSVs are trial-level tables.

Example: `sub-01__ses-S1__0-Back.csv`

- rows: `144`
- columns include:
  - `rt`
  - `hittrials`
  - `miss`
  - `error`
  - `mistake`
  - `correct`
  - `outlier`

Observed pattern for `sub-01 / ses-S1`:

- `0-Back` mean RT about `394.5 ms`
- `1-Back` mean RT about `461.7 ms`
- `2-Back` mean RT about `607.6 ms`

That is a useful sanity check that the behavioral signal tracks increasing task difficulty.

## Current Window Payload Contract

The benchmark currently creates a `WindowedTrial` with:

- subject/session/task identity
- workload label
- window start/end time
- EEG sample slice location
- rolling behavior features up to that window

Verified payload keys include:

- `trial_progress_norm`
- `rolling_accuracy`
- `rolling_rt_percentile`
- `lapse_rate`
- `behavior_correct_rate`
- `behavior_hit_rate`
- `behavior_miss_rate`
- `behavior_error_rate`
- `behavior_mistake_rate`
- `behavior_outlier_rate`
- `behavior_rt_mean_norm`
- `behavior_rt_median_norm`
- `session_index_norm`
- `subjective_workload_rsme`
- `sample_start`
- `sample_end`
- `srate`
- `nbchan`
- file provenance paths

## Current Practical Interpretation

This means the 3A benchmark claim is:

> Given a rolling task window, does EEG add predictive value for estimating workload beyond behavioral/task features alone?

That is cleaner and more defensible than trying to claim the tutor simulator itself proves neural realism.
