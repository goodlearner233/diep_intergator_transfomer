# Fresh DIEPformer training aligned with the official DIEP protocol

This is a separate training entry for the **local DIEPformer model**. It starts a new experiment from random initialization and leaves the original training and Gadi recovery scripts untouched. It does not submit a scheduler job or modify the existing recovery run.

The directory contains:

| File | Role |
|---|---|
| `train.py` | Model construction, AdamW training, validation, checkpoint selection and final test/export |
| `data_pipeline.py` | Input validation, filtering, reference energies, cached graphs, splits and atom-budget batch sampling |
| `telemetry.py` | Streaming per-rank batch diagnostics and rank-zero optimizer diagnostics |

Share this directory **together with the repository's `src/matgl` code**. The Python entry selects that local source tree explicitly; this is not a standalone replacement for an arbitrary installed MatGL or official DIEP package. No PBS, Slurm or shell launcher is supplied here. Resource allocation and the appropriate launcher on the receiving supercomputer should be arranged separately.

The latest code review and validation scope are recorded in [AUDIT_2026-10-01.md](AUDIT_2026-10-01.md).

## What is aligned, and what is retained

The reference is the official DIEP repository at commit [`3bf9cd4cf4fe43081fa37b89edff28eeb2ef9cca`](https://github.com/materialsalchemist/diep/tree/3bf9cd4cf4fe43081fa37b89edff28eeb2ef9cca), especially its [launch settings](https://github.com/materialsalchemist/diep/blob/3bf9cd4cf4fe43081fa37b89edff28eeb2ef9cca/train.sh), [training entry](https://github.com/materialsalchemist/diep/blob/3bf9cd4cf4fe43081fa37b89edff28eeb2ef9cca/scripts/train_mp_pes_pyg.py) and [batch sampler](https://github.com/materialsalchemist/diep/blob/3bf9cd4cf4fe43081fa37b89edff28eeb2ef9cca/src/diep/pyg/graph/data.py).

| Setting | Default in this entry |
|---|---|
| Initialization | Fresh model, seed **42** |
| Structures retained | At most **150 atoms per structure** |
| Batch size rule | At most **1000 atoms per GPU microbatch**, whole structures |
| Data split | Official-style filter first, then approximately **90% / 5% / 5%**, seed 42 |
| Element offsets | Supplied MatPES-PBE isolated-atom reference energies |
| Output scale | Force-vector RMS calculated using **training atoms only** |
| Optimizer | AdamW, AMSGrad enabled |
| Adam coefficients | Beta one **0.9**, beta two **0.999** |
| Weight decay / epsilon | **0.00001** / **0.00000001** |
| Loss | Mean Huber, delta **1.0**, E/F/S weights **1.0 / 1.0 / 0.1** |
| Gradient clipping | Global L2 norm upper bound **2.0** after accumulation/synchronization |
| Learning rate | **0.001** initially, cosine descent to **0.00001** |
| Training campaign | **200 epochs**, with cosine period also **200** |
| Default parallelism | **4 GPUs**, accumulation **1** per GPU |
| Tensor precision | FP32; PyTorch float32 matrix-multiplication precision set to `high` |

The 200-epoch campaign is an explicit adaptation: the pinned official launcher requests 1000 epochs. This entry matches the cosine period to the requested campaign length instead of stopping after only the first 200 epochs of a 1000-epoch schedule.

The model remains **121-dimensional DIEP grid features → three M3GNet blocks → one Transformer layer → atomic linear contributions summed to total energy**. Node and edge embeddings have 64 channels; the Transformer has four heads, feed-forward width 128 and dropout 0.0. Pair and three-body cutoffs remain 5 and 4 angstroms. The local pair smoothing implementation is retained.

The pinned official launcher selects `sum` features with eight channels. **That descriptor change is deliberately not copied here.** This experiment changes the training protocol while retaining the existing DIEPformer representation and Transformer. It is not an exact reproduction of the official model or a single-variable causal test of loss spikes.

## 1. Prepare the environment and two input files

Use the existing environment that can already import and run this repository's PyG DIEPformer, including its PyTorch, Lightning, torch-geometric, NumPy, pymatgen and ASE dependencies. This entry also imports `filelock`. There is no separate speculative dependency list or automatic package installation. Select CUDA-compatible packages for the receiving machine as part of the normal repository setup.

Provide:

1. The raw **MatPES-PBE** structure dataset as a JSON array or JSONL file. Each record must contain `structure`, total `energy`, per-atom `forces` and `stress`.
2. The matching **MatPES-PBE isolated-atom reference file**, conventionally `MatPES-PBE-atoms.json`. The loader accepts the official list format or an explicit element-symbol-to-energy JSON mapping. Required reference energies must exist for every element present in any retained split.

The data contract is total energy in eV, force components in eV/angstrom and **raw MatPES stress in kbar**. The preparation step converts raw stress by multiplying by **−0.1** to match the repository's GPa convention; six-component Voigt stress is expanded to a 3 by 3 matrix. Do not supply already-converted GPa stresses, or apply this conversion beforehand. No conversion is applied to the raw energy or force labels.

Preparation happens automatically at the beginning of the Python command. A preparation lock protects the shared cache; a separate lifetime lock allows only one training invocation to own the output directory through fit, test and export. Workers join the same verified launch. It validates the labels, filters structures, builds graphs and writes a manifest with source hashes, original sample IDs, excluded IDs, atom counts, split membership, reference energies and the computed scale. Cache contents are SHA256-verified during preparation/resume, including same-size file changes. Invalid labels or missing required reference energies produce an error rather than being silently discarded. Version records support either the pymatgen umbrella distribution or pymatgen-core.

The scale is:

```text
force_rms = sqrt(sum over training atoms of (Fx² + Fy² + Fz²) / number of training atoms)
```

It is one fixed force-vector RMS, not a per-batch statistic, a centered standard deviation or an RMS divided by three times the atom count. The potential applies this scale to the network energy and adds element reference contributions. Differentiation therefore scales forces and stress consistently. All reported errors remain in physical units.

Preparation streams raw JSON but still constructs retained structures/labels for the existing graph builder in host memory. Ranks subsequently use memory-mapped graph tensors while loading labels into host memory. This is not a guarantee that full-dataset preparation or four-rank loading fits a particular RAM allocation; measure on the target environment.

### Split policy and previous experiments

`--split-policy official` is the default: filter first, create a seeded permutation of the retained structures, assign rounded 5% validation and 5% test counts, and use the remainder for training. This produces a **new holdout relative to the old unfiltered experiment**, even though both use seed 42. Old M0/M1/old-DIEPformer numbers are therefore not a strictly matched comparison to this new test result.

The optional `--split-policy legacy-membership` recreates the old full-source seed-42 permutation with its original integer 90%/5% boundaries, then filters each split without moving retained structures across split boundaries. It is meaningful only with the **same original file order** as the old experiment. It does not load a previous split file or infer the identity of a reordered dataset. Choose the policy before starting, save the resulting original sample IDs, and use the same policy and members for controlled baseline comparisons.

## 2. Start a fresh campaign

Run from the repository root **after the requested compute resources have been allocated**. Substitute actual input and output paths. This example requests the script's four-GPU configuration; it does not allocate GPUs or submit a scheduler job:

```text
python training_official_aligned/train.py --data /path/to/MatPES-PBE-2025.2.json --element-refs /path/to/MatPES-PBE-atoms.json --output-dir /path/to/runs/diepformer_official_aligned_seed42 --seed 42 --epochs 200 --devices 4 --accumulate-grad-batches 1
```

Use a new output directory for a new experiment. Without `--resume`, an output directory already registered to an earlier invocation is rejected. The original training files and old checkpoints are not overwritten or automatically loaded. The command checks that the requested GPUs are visible and does not silently reduce four GPUs to one.

For a separately identified single-GPU adaptation:

```text
python training_official_aligned/train.py --data /path/to/MatPES-PBE-2025.2.json --element-refs /path/to/MatPES-PBE-atoms.json --output-dir /path/to/runs/diepformer_aligned_single_gpu_seed42 --seed 42 --epochs 200 --devices 1 --accumulate-grad-batches 4
```

Four GPUs with accumulation 1 process four local microbatches before an update; one GPU with accumulation 4 processes four sequential microbatches before an update. Those arrangements can approximate the same gradient averaging, but do not guarantee identical floating-point results, final incomplete accumulation windows, pack ordering or stochastic behavior. Atom-budget batches have variable structure and force-component counts. Four batches no longer imply exactly 128 structures, nor do four GPUs guarantee a fourfold speedup.

The sampler shuffles the structure order once to construct fixed greedy packs. It shuffles pack order each epoch using seed plus epoch, then distributes packs across ranks. A structure is never split across batches. To equalize training step counts, distributed training may repeat a few complete packs at the end; those repeated training visits contribute to optimization and training metrics.

Validation and test also pad to equal forward counts, but mark repeated packs with zero metric weight. **Padding samples are excluded from the reported validation/test metrics.** Telemetry retains their rows with explicit `metric_weight` and `padding_sample_count` fields so they can also be excluded from later analysis.

Useful optional arguments are `--num-workers`, `--max-atoms-per-batch`, `--no-progress-bar` and `--no-telemetry`. Worker count defaults to zero for portability; choose a measured setting on the target machine. Lowering the atom budget changes the experiment's batch composition and update count, so use a separate run identity. The 1000-atom limit is not a memory guarantee for a 121-channel Transformer model.

## 3. Follow the training and validation sequence

After preparation and initialization, Lightning runs two sanity-validation batches, then repeats:

1. Predict total energy and obtain coordinate derivatives for forces and strain derivatives for stress.
2. Compare energy **per atom**, all force components and all stress components with references using mean Huber losses.
3. Form `energy_loss + force_loss + 0.1 * stress_loss` and accumulate/backpropagate gradients.
4. Synchronize gradients in DDP, reject nonfinite gradients, clip the global norm to 2, and apply AdamW with AMSGrad.
5. After the final training batch, Lightning advances the cosine scheduler once for the next epoch. It validates the current parameters, updates the best-validation selection and saves epoch checkpoints with that advanced scheduler state. No training callback manually steps the scheduler.

The entry explicitly uses the mathematical attention backend to support the derivative path required by force/stress training. It stops on a nonfinite loss or gradient before proceeding with the corresponding parameter update. This does not promise to prevent every finite but excessively large update or every loss spike.

### Metric definitions and checkpoint selection

`train_Total_Loss`, `val_Total_Loss` and `test_Total_Loss` are weighted E/F/S Huber losses, not energy MAE. For the primary metrics, a batch's mean metric is weighted by the number of real structures when forming the epoch value, following the official-style reporting convention. Force means inside a batch include all atomic Cartesian components; stress means include the full tensors. Primary RMSE is the structure-count-weighted average of **batch RMSE values**.

The entry additionally writes `*_pooled_Energy_*`, `*_pooled_Force_*`, `*_pooled_Stress_*` and `*_pooled_Total_Loss`. These first sum errors over the entire phase and then divide by the relevant component count; pooled RMSE takes its square root only after that global division. In particular, pooled force errors weight all force components equally across the phase. With variable-sized batches, primary and pooled values need not match. Do not compare one convention from this experiment against the other convention from another experiment without labeling the difference.

The best checkpoint is selected by **minimum `val_Total_Loss`**, not by a pooled metric, test error or minimum training error. Epoch numbers in checkpoint names/logs are zero-based; `completed_epochs` gives the human-readable count.

## 4. Inspect outputs and diagnostic cost

| Output | Contents |
|---|---|
| `run_config.json` | Effective settings, seed, official reference commit, code/data signature and package versions |
| `source_snapshot/` | Copy of the Python source used for this run |
| `prepared_data/manifest.json` and cache `split_indices.json` | Input/reference identity, graph-cache configuration, original IDs, filtering, splits and scaling |
| `logs/version_*/metrics.csv` | Lightning epoch metrics and learning-rate logging |
| `epoch_history.jsonl` | One consolidated rank-zero epoch record with completed epoch count and metrics |
| `checkpoints/best-*.ckpt` | Best validation-selected full checkpoint |
| `checkpoints/epoch-*.ckpt` and `checkpoints/last.ckpt` | Full epoch checkpoints and latest epoch save |
| `checkpoints/emergency/step-latest.ckpt` | Rolling full diagnostic snapshot, overwritten every 500 optimizer updates; not accepted for training resume |
| `fit_*_rank*/`, `test_*_rank*/` | Unique compressed telemetry sessions; a resume creates additional sessions |
| `training_status.json` | Status written after a successful return from `fit`; not a live scheduler-status file |
| `final_test.json`, `best_potential/` | Final selected-checkpoint test results and exported potential after campaign completion |

Every rank records batch IDs, sample IDs, atom/edge/triplet counts, per-structure energy/force/stress errors, batch Huber/MAE/RMSE, and CUDA allocator allocated/reserved/peak bytes. The triplet observer uses the actual forward geometry and the model's ordered source-sharing edge-pair rule within the three-body cutoff. CPU runs record null CUDA values. Host elapsed time is labeled **unsynchronized** and is not GPU kernel timing, utilization or complete dataloader timing. Allocator values exclude other processes and some non-PyTorch device allocations; observer work contributes a small amount to the measured peak.

Rank zero records each actual optimizer update: synchronized gradient norms before/after clipping, true parameter change, relative change, AdamW settings, raw first/second moments, AMSGrad maximum moments and the bias-corrected denominator used by the optimizer. A deterministic set of up to 128 scalar parameter coordinates records matched gradients, parameters, moments and denominator values before/after an update. The analytical weight-decay contribution and the remaining actual change are recorded separately; the remainder includes floating-point rounding. These are parameter coordinates, not 128 training structures.

The observer also records **module-level summaries by default** (telemetry schema 2). The session metadata maps every observed parameter to its name-prefix group, including individual graph-block indices. Each update records the group's parameter norm, pre/post-clip gradient norms, actual update norm, relative update and nonfinite counts. Before/after Adam summaries include bias-corrected first/second-moment norms, the square-root second-moment norm, the actual AMSGrad denominator minimum/maximum, inverse-denominator norm/maximum and optimizer-step range. Each parameter uses its own optimizer-group settings and state step. Uninitialized state is explicitly excluded; existing state with no current gradient is retained and separately counted. These summaries help localize an anomaly, but are not activation, Hessian or causal measurements.

Module summaries reduce one parameter tensor at a time and reuse the existing actual-update vector. They do not add a second full-model snapshot or retain past tensors. The Python callback accepts `module_statistics=False` for controlled overhead comparisons; the training entry leaves the default enabled. Coordinate-probe selection is unchanged and does not guarantee that all small modules receive probes; the module summaries cover all observed trainable parameters independently of those probes.

No checkpoint is written for every batch. No full gradient history or Hessian is saved. The observer retains one temporary parameter snapshot per observed update; compressed logs are streamed with bounded buffering. **Total disk usage still grows with the number of batches, structures, updates and retained epoch checkpoints.** Measure bytes and runtime per epoch during a short representative run before estimating a 200-epoch allocation; no fixed full-run RAM, GPU-memory or disk requirement is asserted here.

Telemetry flushes regularly and attempts durable flushing at epoch boundaries and exceptions. A failed-loss batch retains its per-structure errors; a nonfinite-gradient stop retains pending scalar/module/coordinate evidence without serializing the temporary full-parameter tensor. Validation and test failures are synchronized across ranks before proceeding. A hard walltime/process kill can still lose recent rows or leave a gzip stream without its final footer; recovery readers may need to retain complete decompressed lines and tolerate a truncated final stream. Diagnostic values provide clues about spikes, not proof that Adam, large structures or a particular module caused them.

### Measured module-observer overhead (2026-10-01)

The independent benchmark uses the actual 411,628-parameter DIEPformer layout with deterministic **synthetic gradients**, FP32 AdamW/AMSGrad and an RTX 4060 Laptop GPU (PyTorch 2.1.2). It isolates optimizer observation; it does not execute graph construction or E/F/S forward/backward. Three fresh-process repetitions per mode measured five updates each, with identical final model and optimizer states in all modes.

| Comparison | CUDA allocated peak difference | Sampled host RSS growth difference, including first observer use |
|---|---:|---:|
| Full observer versus no observer | 19.38 MB (18.49 MiB) | Approximately 244 MB (232 MiB) |
| Added module summaries versus previous global/coordinate observer | No measured peak increase | Approximately 35.5 MB (33.9 MiB) |

Both observer modes reached the same absolute allocator peak, 29.37 MB, because the existing global snapshots dominate this small isolated test. This does not mean the new reductions allocate no temporary buffers. One complete FP32 parameter copy is 1.65 MB; the largest single parameter occupies 48 KiB. These statistics are collected only on rank zero; other ranks retain their batch-observation work.

A follow-up after three observer warm-up updates measured about 0.5 MiB of further sampled host RSS growth over five updates in either observer mode. The larger first-use footprint includes runtime/kernel initialization and remains part of the process footprint; the warm result does not cancel it. This bounded-window observation is not a universal memory guarantee for a different GPU, runtime or full-dataset job.

The warmed isolated optimizer/update time was approximately 0.037 seconds with the previous observer and 0.382 seconds with module summaries, an added 0.345 seconds per update on this machine. Thus low incremental GPU memory does not imply negligible compute overhead. This is not total training-step time or a throughput prediction for the target cluster. Module fields added roughly 1.8 KB compressed per update in the short synthetic test, plus session metadata; real values, batching and compression will change disk growth.

Results: `测试笔记本/official_aligned_training_20260930/module_benchmark_gpu_20261001/summary.json` and `module_benchmark_gpu_warm_20261001/{base,modules}/result.json`. The reproducible benchmark is `benchmark_module_telemetry.py` in that test directory.

## 5. Resume only this new experiment

Prefer `checkpoints/last.ckpt` or an explicit full epoch checkpoint from this output directory. Repeat the same command and training configuration, adding:

```text
--resume /path/to/runs/diepformer_official_aligned_seed42/checkpoints/last.ckpt
```

Keep `--epochs 200`; it defines the full cosine campaign even if a job only completes part of it. The full checkpoint restores model parameters, AdamW/AMSGrad state, scheduler and Lightning counters. The script validates the recorded format and source/data/configuration signature. It rejects old Adam recovery checkpoints and changed code, seed, split policy, atom budgets, GPU count, accumulation or other signed training settings. Do not delete metadata to bypass this protection. Changes intentionally excluded from the training signature include telemetry verbosity, worker count, progress display and an early stop limit.

For a controlled short segment without shortening the 200-epoch cosine schedule, use `--stop-after-epochs 2` while retaining `--epochs 200`. This stops after two **total completed epochs**, not two additional epochs after every resume. Remove the early-stop limit when continuing the full campaign. Use a small separately identified dataset/output for functional checks; do not treat a subset run as the full MatPES experiment.

The rolling 500-update snapshot is **diagnostic-only**, including a snapshot taken after the final training batch but before validation and the epoch-end save. Partial-epoch restore does not currently restore this entry's phase accumulators and sampler progress reliably, so the entry explicitly refuses it instead of silently replaying incomplete state. Resume from the latest completed-epoch checkpoint; an interrupted partial epoch is repeated from that saved boundary. A stop limit below the checkpoint's completed count is also rejected. Resuming an already-completed campaign performs no extra optimizer or scheduler updates and can run its final test/export.

Use `last.ckpt` for ordinary continuation. Explicitly loading an older completed-epoch checkpoint rewinds model/optimizer/scheduler state and can append repeated epoch numbers to the current output's history; it must not be analyzed as an uninterrupted continuation. This entry has no separate historical-fork command. Fixed seed and deterministic pack ordering do not by themselves guarantee identical trajectories across GPU counts, hardware or software versions. There is no automatic job resubmission chain.

## 6. Complete the campaign, then test and export

After all 200 epochs have completed, the default behavior explicitly reloads the best checkpoint selected by validation loss, evaluates it on the held-out test split and writes `final_test.json`. It then exports `best_potential/`, including the element offsets and force-RMS scale needed to reproduce energy/force/stress predictions. An early-stopped segment does not run the final test. `--skip-final-test` suppresses the automatic final test/export when deliberately requested.

Use validation to compare training settings; reserve test results for the final evaluation. A good test result still does not establish suitability for a particular Li–LGPS interface or stable molecular dynamics. Those application checks remain separate experiments. Likewise, if this combined protocol improves the training curve, isolating the reason requires controlled comparisons rather than attributing improvement to AdamW or AMSGrad alone.

## Local validation completed

Validation scripts and outputs are kept under `测试笔记本/official_aligned_training_20260930/`.

- Nine preparation/sampler tests cover filtering, reference coverage, raw IDs, stress conversion, train-only RMS, splits, same-size cache tampering and core-only pymatgen metadata.
- Twelve observer tests include CPU and RTX4060 runs. Enabling telemetry leaves toy-model weights and every AdamW state tensor exactly unchanged; clipping, AMSGrad denominators, decay, workload counts, module grouping, mixed optimizer settings/state steps, missing gradients, uninitialized/nonfinite states and failure-evidence preservation are checked.
- A 24-real-structure MatPES E/F/S integration trains one epoch, resumes to two, selects the best checkpoint, tests and exports. It verifies exact real-model weights/AdamW equality with telemetry on/off, scheduler continuity, current epoch records and refusal to overwrite or resume incompatible settings.
- The full entry also completed a one-epoch real E/F/S run on one RTX4060 and a two-process CPU Gloo fit/validation/test run, including zero-weight evaluation padding.
- An independent real E/F/S two-rank Gloo update matches sequential averaging of the same two unequal batches exactly for gradients, parameters and optimizer state. Gradient clipping was exercised.
- Independent metric tests verify pooled errors across different pack sizes and rank counts. An actual Lightning loop with a tiny synthetic E/F/S predictor runs 200 epochs/400 optimizer updates with exactly 200 scheduler advances. A 73-epoch interruption/resume produces bitwise-identical final weights, optimizer state and scheduler state to an uninterrupted run.
- Two-rank Gloo fault injection checks one-rank nonfinite validation/test loss and gradients: both ranks stop, diagnostic evidence is retained and AdamW does not update parameters.
- Five subprocess admission checks cover duplicate invocation rejection, worker admission, stale metadata and lock cleanup. Resume boundary checks cover completed runs, preserved earlier best models, invalid stop limits and rejected partial-epoch snapshots.

These are functional checks on small inputs. Four-GPU NCCL operation, full-dataset memory requirements, throughput and long-run training behavior still need measurement on the target supercomputer. `epoch_history.jsonl` records its `lr` after that epoch's scheduler advance; the epoch-start console line and update telemetry show the learning rate actually used for updates.
