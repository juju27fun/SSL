# SSL

Masked waveform reconstruction with the **U-Net used by the MAD4d experiment**, plus preprocessing and latent-space evaluation. Architecture code belongs to the companion **Models** repository; this package imports `smi_models`. There is no second copy of the U-Net or Conv1DGAP-S here.

This is a faithful extraction of a development protocol, not a claim of independent accuracy or biological generalisation. It includes no private recordings, trained checkpoints, notebooks, remote jobs or tracking-service dependency.

## Install

Use Python 3.12 and uv. From this SSL checkout:

```bash
uv sync --locked
uv run pytest -q
```

Models is fetched automatically at the immutable Git commit declared in `pyproject.toml` and `uv.lock`; no sibling checkout or original workspace is required. The architecture revision is pinned so another checkout cannot silently change the experiment. Update the declared revision and regenerate the lock only after validating compatibility. CI runs automatically on pushes and pull requests, and can also be triggered manually.

`uv build` emits a wheel depending on `smi-research-models==0.1.0`. For wheel-only installation, install the Models wheel built from that declared revision alongside the SSL wheel. The Git pin belongs to the reproducible uv environment; ordinary wheel metadata identifies the package version.

## First complete run on CPU

These commands exercise the entire software path using 18 generated toy waveforms. They do **not** reproduce the original experiment's data, training budget or performance.

```bash
uv run smi-ssl demo-data --output .cache/demo/events.npz
uv run smi-ssl train --data .cache/demo/events.npz --output .cache/demo/run --profile smoke
uv run smi-ssl encode --data .cache/demo/events.npz --checkpoint .cache/demo/run/checkpoints/latest.pt --output .cache/demo/embeddings.npz
uv run smi-ssl cluster --embeddings .cache/demo/embeddings.npz --output .cache/demo/held-out.json
uv run smi-ssl cluster --embeddings .cache/demo/embeddings.npz --output .cache/demo/transductive.json --protocol transductive
```

Use a new output path to repeat a command; existing outputs are never silently replaced. The toy third split stands in for the real-validation interface and is synthetic too. The displayed losses/ARI are software diagnostics, not scientific results.

The smoke profile runs one epoch and caps real-validation monitoring at two rows per supplied class label; unlabeled rows (`-1`) form one group for this cap. Full monitoring is uncapped. The full preset has 30 epochs, batch size 32, AdamW learning rate 0.0003, weight decay 0.01 and gradient clipping at 1.0, matching the retained U-Net run's override. CPU smoke uses one PyTorch thread; full training can select `--device cuda` with a suitable PyTorch/CUDA installation. GPU execution and research-scale convergence are not validated by the CPU smoke.

## Signal path and ownership

1. **Start from already delimited events.** Detection belongs to MAD_detection. The retained real bead loader starts from upstream-preprocessed acquisition traces; it does not apply another bandpass. Do not filter an already filtered trace a second time.
2. **Crop for the U-Net.** The `prepare` command takes a 4,096-sample window centred on `center_sample`, clamped inside the source trace. At 2 MHz this is 2.048 ms. There is no resampling. Source event bounds are transformed into crop coordinates and constrained to leave an 8% background margin at both ends, matching the original MAD4d loader.
3. **Normalize each window.** `normalize_signal(..., mode="window_zscore")` uses `(x-mean(x))/max(std(x),1e-6)` independently for each complete window, before masking. This is the original local normalization, including the samples subsequently hidden. It is not a global train-set normalization and does not fit on validation populations.
4. **Choose hidden samples.** P25 samples isolated patch-aligned windows; CYCLIC25 advances a deterministic balanced cycle over each event and background. Boolean True means hidden, both for the model's visibility input and the loss target. Each retained training pass selects 1,024 of 4,096 samples using 16-sample windows. CYCLIC25 uses a candidate stride of eight, with 32 event and 32 background windows per pass. Short supports are expanded with context by the original mask builder.
5. **Reconstruct.** Models' U-Net receives `(B,1,4096)` signals plus a `(B,4096)` mask. It zeroes hidden amplitudes, concatenates the mask as a second channel and returns `(B,1,4096)` predictions.
6. **Optimize and validate.** The retained B0 objective is masked waveform MSE; original derivative/energy loss components and B1–B3 settings remain available explicitly in the configuration. Validation compares against zero, visible mean, nearest-neighbour and linear-interpolation baselines. The retained preset uses P25 for standard validation; changing `masking.evaluation_policy` to CYCLIC25 now executes complete cycles for initial, per-epoch, final simulation and real validation. Unknown policies are rejected. Training and validation masks therefore have distinct, documented roles.
The CLI also restores the retained **matched monitoring**: select up to 2,048 rows from each prepared train/val population using class-proportional quotas and the original SHA-256 ranking of `seed|split_tag|sample_id`. The same subsets are evaluated before training and after every epoch under the training mask policy (CYCLIC25 by default), using all unique passes per sample. `metrics.json.monitoring` records counts, class quotas and selected-ID hashes; each history entry contains `matched_monitor`. Smaller smoke populations are used in full. Labels 0/1/2 map to 2um/4um/10um for this selection; other labels (including -1) are explicit additional strata, without inferring a physical class. Row order still defines mask sample indices, so preserve archive order for exact reruns.

7. **Keep the fixed final checkpoint.** `latest.pt` is saved every epoch and `best.pt` records the best P25 validation MSE, but the retained preset selects the final epoch. No early stopping or best-checkpoint substitution is introduced.
8. **Extract and evaluate.** Mean-pooled U-Net bottleneck features have 96 dimensions. The clustering modes below state which population fits every transform and the label mapping.

`signal_preprocessing.py` also preserves the existing optional P1 bandpass/saturation-quality helpers as an explicit API. `conv_preprocessing.prepare_event_signal` preserves the supervised Conv1DGAP-S comparison path: reflect-pad a 4,096-point crop, average consecutive blocks of eight to 512 points, optionally augment, then use the original `(std+1e-8)` normalization. These are separate processing profiles; neither is silently applied to U-Net inputs. The older yeast protocols and their different normalisation/resampling conventions are outside the retained MAD4d training recipe.

## Supply your events

Create an event CSV with these columns:

```csv
id,path,split,center_sample,start_sample,end_sample,group,label
sample-001,acquisition-001.npy,train,8192,7600,8700,acquisition-001,0
```

Provide rows for all three splits `train`, `val` and `real_val`. The single row above only illustrates the schema and is not a complete dataset. Bounds use source sample indices, `[start_sample,end_sample)`; time is converted using 2 MHz. `path` is relative to the supplied signal root. Labels are integer class IDs for downstream interpretation; they do not enter the reconstruction loss. Use `-1` if labels are unavailable (training works; clustering requires valid labels only on its fit/scored populations). A `group` must identify the whole original acquisition/noise carrier; all its derived events belong to one split. If omitted, the source file checksum is used as the group.

```bash
uv run smi-ssl prepare --manifest events.csv --signal-root acquisitions --output prepared.npz
uv run smi-ssl train --data prepared.npz --output runs/unet --profile full --device cuda
```

Input traces must be finite 1-D arrays with at least 4,096 samples and the intended upstream filtering already applied. This command does not repair saturation or infer event boundaries. It reproduces the event cropping stage on explicit user inputs.

Alternatively supply an NPZ archive directly:

| Key | Shape / type | Meaning |
|---|---|---|
| `signals` | `(N,4096)`, finite float | Cropped samples before per-window normalization |
| `event_masks` | `(N,4096)`, boolean | Event support used by CYCLIC25, not hidden-target masks |
| `split` | `(N,)`, string | `train`, `val` or `real_val` |
| `ids` | `(N,)`, unique string | Stable event IDs |
| `groups` | `(N,)`, nonempty string | Source groups that cannot cross splits |
| `labels` | `(N,)`, integer | Class interpretation only |
| `sampling_frequency_hz` | scalar | Exactly `2000000` for this preset |

The reader rejects malformed arrays, absent splits, empty event/background support, duplicate IDs and source-group overlap. Source-group correctness is the data supplier's responsibility. Synthetic source rows in the original research were split by their noise carrier; the portable archive must preserve that grouping. The original synthetic support uses ±2 tau (with asymmetric side scaling when present) and an 8% background margin; provide those masks when preparing equivalent synthetic arrays. The toy example only tests this interface.

## Configuration and checkpoints

```bash
uv run smi-ssl config --output unet-config.json
uv run smi-ssl train --data prepared.npz --config unet-config.json --output runs/configured --profile smoke
```

The packaged `presets/unet_mad4d.json` is flattened from the original inherited YAML configurations, with the verified 30-epoch/CYCLIC25 run override. `study` dataset IDs preserve provenance only; no registry lookup is performed. The independent CLI takes explicit input/output paths. See `configuration.py` for the supported contract, `training.py` for the original loop, `masking.py` for masks and `losses.py` for objectives.

Checkpoints retain the original dictionary keys `model_state_dict`, `config`, `epoch` and validation metrics. `load_checkpoint` uses strict parameter loading and PyTorch's restricted `weights_only=True` loader. Changing architecture shape makes incompatible checkpoints fail. The retained source checkpoint format contains no optimizer/RNG state: **loading supports inference and inspection, not exact interrupted-training resume**. No checkpoint is shipped; reproducing trained representations requires authorized weights or your own training.

## K-means and Hungarian interpretation

**Held-out** is the default: fit StandardScaler, row L2 normalization and K-means on train representations. Fit the cluster-to-class Hungarian assignment on train labels only, then predict and score validation. PCA is fitted on train for a 2-D view, while K-means uses the full normalized latent space. ARI uses raw cluster IDs; the Hungarian mapping affects class-labelled confusion only. Seeds 41/42/43 and `n_init=20` follow the existing evaluator.

**Transductive** fits these transforms and clusters on validation itself and applies Hungarian interpretation there too. It describes that population; it cannot establish held-out generalisation. The output JSON names the protocol, fit/scored populations and claim boundary. `real_val` is used for reconstruction monitoring, not silently combined into either clustering partition. Train and validation must remain available and non-overlapping even for this explicit control. Held-out requires class labels on train and val; transductive requires them on val only. `real_val=-1` never blocks either clustering protocol.

## Outputs, tests and limits

A training run writes `checkpoints/latest.pt`, `checkpoints/best.pt`, `history.json`, `metrics.json` and bounded reconstruction examples. `encode` writes an NPZ with embeddings, IDs, groups, splits and labels. `cluster` writes per-seed ARI, Hungarian mapping, predictions and confusion summaries.

```bash
uv run pytest -q
uv build
```

Tests cover original masking/preprocessing/clustering numerics, hidden-input invariance, train/validation group exclusion, local normalization, a CPU optimization step and checkpoint roundtrip. `PROVENANCE.json` records source commits, file hashes and extracted ranges, including the actual U-Net worktree. The loop is copied rather than recreated; legacy registry-backed constructors are replaced by prepared datasets and architecture imports point to Models.

No research corpus, final-test split, cloud service or remote GPU is touched by these checks. Exact scientific reproduction additionally requires the original data preparation, authorized inputs, checkpoints or full training, and matching numerical/hardware conditions. The source experiment is exploratory and single-seed; running this code does not expand that evidence. Licensing/redistribution rights remain those of the source owners.

The installed distribution includes `PROVENANCE.json` and this README under `share/smi_ssl` in the installation prefix.
