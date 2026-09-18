# DIEP + Transformer + Linear: Gadi training

Target: **200 cumulative epochs**, from a fresh DIEP model. Start with the
2-epoch pilot below, then resume the same run after checking GPU memory, speed,
and the remaining project allocation. No automatic job chaining is enabled.

## 1. Clone into a new folder (Gadi login terminal)

```bash
mkdir -p /scratch/xq82/bd1444/diep_intergator_transformer_work
cd /scratch/xq82/bd1444/diep_intergator_transformer_work
git clone --branch main https://github.com/goodlearner233/diep_intergator_transfomer.git repo
cd repo
git log -1 --oneline
```

For an existing checkout, inspect `git status` before using `git pull --ff-only`.
Keep the source revision unchanged between segments of a training run. Do not
pull into a checkout while its training job is running.

## 2. Check the existing environment, data, queue and allocation

The PBS file reuses the existing CUDA environment and dataset. Do not reinstall
or upgrade PyTorch simply to run this fork; `PYTHONPATH` selects this checkout.

```bash
test -f /scratch/xq82/bd1444/diepformer_dgl_work/miniforge3/etc/profile.d/conda.sh
test -x /scratch/xq82/bd1444/matglformer_linear_work/envs/matgl-linear/bin/python
test -r /scratch/xq82/bd1444/matglformer_linear_work/data/MatPES-PBE-2025.2.json
nci_account -P xq82
qstat -Qf gpuhopper
bash -n gadi/train_diep_200_segment.pbs
```

Stop and inspect any failed prerequisite before submission. Paths and project
names in this PBS file are for this existing installation; other users must
adapt them to their account.

## 3. Submit the first segment from the repository root

```bash
qsub gadi/train_diep_200_segment.pbs
qstat -u "$USER"
```

Record the job ID returned by `qsub`. The pilot requests 1 GPU, 12 CPUs, 256 GB
RAM and 4 hours in `gpuhopper`, and ends at cumulative epoch 2. This is not a
guarantee that first-time full-dataset preprocessing and two epochs fit in four
hours. Check the output log before requesting the next segment.

Defaults:

| Setting | Value |
| --- | --- |
| Model | DIEP pair/triplet → 3 M3GNet blocks → Transformer → Linear atomic sum |
| Batch / gradient accumulation | 32 / 4 on one GPU (usually 128 structures per parameter update) |
| Transformer | 4 heads, 1 layer, FFN 128, dropout 0 |
| Pair / triplet cutoff | 5 Å / 4 Å |
| DIEP grid | [-5, 5] in each direction, spacing 1; 121 channels |
| Loss | Huber(E/N) + Huber(F) + 0.1 Huber(stress) |
| Optimizer / learning rate | Adam / 0.001 |
| Scheduler | Cosine, 1000 epochs to 0.00001; advances once per epoch |
| Data / split | MatPES-PBE-2025.2 / 90:5:5, seed 42 |

The entry point is `train_transformer_atomic_sum_full_matpes.py`. It uses the
existing `PotentialLightningModule → Potential → M3GNet` chain. The additional
`EpochScheduleCheck` validates scheduler progress; it does not advance it.

## 4. Inspect and resume

The new run lives under `runs/diep_bs32_acc4_200ep/`. Inspect the PBS output,
CSV/TensorBoard metrics, `segment_results.json`, and `checkpoints/last.ckpt`.
Check finite energy/force/stress metrics and scheduler count, as well as runtime.
The initial `nvidia-smi` output is not a measurement of peak training memory.

After the previous job completes, choose an endpoint and walltime from the
measured speed and actual queue limits. For example only:

```bash
qsub -l walltime=24:00:00 -v END_EPOCH=20 gadi/train_diep_200_segment.pbs
```

`END_EPOCH=20` means train **up to epoch 20**, not 20 additional epochs. The PBS
script automatically loads this run's `last.ckpt`, restoring weights, Adam,
scheduler, and progress. It refuses concurrent jobs writing to the same run.
Use this same mechanism until `END_EPOCH=200`; only that final segment runs the
held-out test set using the best validation checkpoint.

Do not resume old M3GNet-basis weights or the local smoke-test checkpoint.
Checkpoints are saved at complete epochs; a walltime kill can lose the partial
epoch. Keep enough time for preprocessing, validation and checkpoint writing.

## Validation scope

Local CPU checks have exercised the actual training entry point for two epochs,
then restored the checkpoint and continued to epoch three with final testing.
Scheduler counts were 2 and 3 respectively; parameters, optimizer state and
energy/force/stress results were finite. Those were small-sample tests, not a
claim about trained accuracy or GPU batch32 memory. The known second-coordinate
derivative issue in triplet projection has not been changed in this version.
