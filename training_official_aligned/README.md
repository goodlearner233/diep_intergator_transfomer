# DIEPformer training

After comparing our previous training setup with the [official DIEP training entry](https://github.com/materialsalchemist/diep/blob/3bf9cd4cf4fe43081fa37b89edff28eeb2ef9cca/scripts/train_mp_pes_pyg.py), we decided to retrain DIEPformer from scratch, adopting the following practices and adding diagnostics to investigate training spikes:

- Fixed-size batches → atom-budget batches with a per-structure size limit.
- No element reference offsets and unit output scale → MatPES atomic references and training-set force RMS scaling.
- Adam → AdamW with AMSGrad and weight decay.
- A cosine schedule designed for 1000 epochs → a schedule matching the proposed 200-epoch run, plus batch/update diagnostics to investigate spikes.

The 121-channel DIEP → M3GNet → Transformer → linear-sum model is retained. The new Python entry is in this folder; the original training files are preserved.

**Settings below are proposed defaults. Final training parameters, computing resources and the launch script will be agreed with Sherif after discussion.** Seed 42 is retained. No new cluster launcher is supplied yet.

## 1. What is recorded

- **Each epoch:** training/validation energy, force and stress MAE, RMSE, Huber loss and learning rate.
- **Each batch:** structure IDs, atom/edge/triplet counts, per-structure errors, GPU memory and elapsed time.
- **Each optimizer update:** gradients before/after clipping, actual parameter changes and AdamW/AMSGrad statistics, including module summaries and 128 fixed parameter probes.
- **Checkpoints and setup:** best model, every completed epoch, latest completed epoch, configuration, data splits and source snapshot.

## 2. How to use the results

All files are under --output-dir:

| File | Use |
|---|---|
| logs/version_*/metrics.csv | Plot learning curves and find the lowest val_Total_Loss (weighted Huber loss). |
| fit_*_rank*/batches.jsonl.gz | Identify which structures/batches have large errors or memory use. |
| Rank-zero optimizer_updates.jsonl.gz and coordinates.jsonl.gz inside the fit folder | Check gradients, parameter updates and optimizer/module behavior around a spike. |
| final_test.json and best_potential/ | Final test results and the exported validation-best model. |

To investigate a spike: **find the epoch → inspect its batches → inspect the corresponding optimizer updates**. Keep logs from every rank together. These records provide clues; they do not by themselves prove the cause.

## 3. Start training

Proposed four-GPU launch command, to confirm with Sherif. From the repository root, activate its working Python environment and run this **after allocating four GPUs**. Replace the two input paths. Supply raw MatPES stress in kbar; the script converts it.

~~~bash
python training_official_aligned/train.py \
  --data /path/to/MatPES-PBE-2025.2.json \
  --element-refs /path/to/MatPES-PBE-atoms.json \
  --output-dir runs/diepformer_aligned_seed42 \
  --epochs 200 --seed 42 \
  --devices 4 --accumulate-grad-batches 1
~~~

Defaults: AdamW + AMSGrad; learning rate **0.001 → 0.00001**; gradient clipping **2**; at most **150 atoms per structure** and **1000 atoms per GPU batch**. Diagnostics are enabled automatically.

**One GPU:** use --devices 1 --accumulate-grad-batches 4 and a separate output directory.

**Resume:** repeat the original command and add:
~~~bash
--resume runs/diepformer_aligned_seed42/checkpoints/last.ckpt
~~~
Keep --epochs 200. Use a completed-epoch checkpoint; the rolling emergency/step-latest.ckpt is for diagnosis only.

The Python command does not submit a cluster job; the site's PBS/Slurm header is configured separately.

[Detailed code audit](AUDIT_2026-10-01.md)
