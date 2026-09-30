# Independent training-entry validation

Run from the repository root in the environment used for this PyG DIEPformer. Tests are separate from the three production Python files. They write generated results beside themselves; do not commit the resulting caches, logs or checkpoints.

For CPU checks, set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and, when needed by the local MKL environment, `MKL_THREADING_LAYER=SEQUENTIAL`. Windows Gloo tests use localhost TCP and `USE_LIBUV=0`. Python UTF-8 mode (`python -X utf8`) supports this directory name.

## Tests without external datasets

Run each file with Python:

- `test_data_pipeline.py`: synthetic structures, references, splits/RMS, cache integrity and package metadata.
- `test_telemetry.py`, `test_module_telemetry.py`, `test_telemetry_failures.py`: observer formulas, CPU/GPU noninterference, module summaries and failure evidence. GPU cases run when CUDA is available.
- `test_metric_invariance.py`: pooled E/F/S aggregation, evaluation padding and checkpoint identity guards.
- `test_run_admission.py`: real subprocess launch ownership and worker admission.
- `test_distributed_failures.py`: two-process nonfinite-loss/gradient rejection through production code.
- `test_lightning_200_epoch_schedule.py`: actual Lightning control flow for uninterrupted and resumed 200-epoch campaigns using a tiny predictor.
- `test_emergency_resume_schedule.py`: explicit rejection of partial-epoch snapshots, including the last training batch before validation.

`test_resume_boundaries.py` takes the result directory printed by `test_lightning_200_epoch_schedule.py` as its positional argument. The emergency-resume regression checks rejection of partial-epoch snapshots. These are control-flow tests, not scientific accuracy experiments.

## Real E/F/S checks

`test_training_entry.py` and `test_ddp_efs_equivalence.py` expect:

- `测试笔记本/官方训练入口对照_20260930/sample_24.json`: a 24-record subset in the raw MatPES PBE schema (stress still in kbar).
- `测试笔记本/official_aligned_training_20260930/MatPES-PBE-atoms.json`: matching isolated-atom reference energies.

These data files are not included in this source delivery. Use the already prepared local fixture, or provide an equivalent deliberately identified small MatPES fixture. Tests assert their own fixed fixture's batch/update counts, so arbitrary replacement data can require adapting those assertions; never interpret these fixtures as the full experiment.

`benchmark_module_telemetry.py` is an optional isolated optimizer-observer benchmark with synthetic gradients. It does not measure complete E/F/S training throughput. See the production README for measured scope and limitations.
