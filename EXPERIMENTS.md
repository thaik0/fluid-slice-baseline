# Diagnose before changing the formulation

The first real-data pilot optimized but generated too little large-scale energy,
excess fine-scale variation, and consistent component-mean drift across sampling
seeds. Shifted patches greatly reduced seams while lowering energy. We need to
find which part of the pipeline causes these failures. No geometric or spectral
modification is introduced in this screen.

## Stage 1: existing checkpoint, ten sampling arms

`diagnose.py` runs these treatments with identical initial noise for each seed:

| Question | Treatments | Decision |
| --- | --- | --- |
| Is Euler integration too coarse? | fixed Euler 32,64,128 | If results converge but stay wrong, more steps will not repair this checkpoint. |
| Does solver order matter at matched network evaluations? | Heun32 vs Euler64; Heun64 vs Euler128 | Improvement at equal NFE supports changing the solver. Compare drift and spectra, not loss alone. |
| Does resolution of the shifted integration matter? | all-shift Euler32,64,128 | Separates integration refinement from the number of geometry changes. |
| When are shifts useful/harmful? | early-half, late-half, all, fixed at Euler64 | Look for a schedule that reduces seams without further large-scale energy loss. |

Shifts are piecewise constant on 32 physical-time bins. All integration grids
align with those boundaries. The 32-step all-shift arm reproduces the old
sampler. Refining to 64/128 does NOT increase the number of shifts. Heun uses
the same geometry for both function evaluations within an interval. Its endpoint
evaluation uses the left geometry at discontinuities, avoiding cross-boundary
averaging. Only fixed geometry gets the solver-order comparison in this screen.

Sixteen batches of random periodic crops from TRAIN and VALIDATION are probed
at t = 0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1. The same crops/noise are reused across
times within each batch, with dedicated seeded RNGs. Report per-component MSE,
mean signed error, and squared error of spatial means. These diagnose errors on
the true noise/data interpolation path, not accumulated generated-state error.
Endpoint t=1 is an extrapolative diagnostic of a model trained at continuous
random t, not a separately trained endpoint. Integrated t-loss is not estimated
by naively averaging these deliberately uneven probe locations.

The sampler records physical component means, total energy, and plane-centered
energy at nine times. This locates when drift appears. Do not remove sample
means as an unreported fix: real slice means are allowed to vary. Grouped real
validation statistics preserve snapshot membership for future uncertainty work;
we do not claim generated seeds or correlated real planes give independent
training replicates.

Each arm reports NFE, warmed runtime, allocated CUDA memory, spectrum errors,
seams, energies, and paired per-seed differences against fixed Euler64. Timing
includes nine trace collections (with device synchronization effects) and is
not directly comparable to the previous sampler's uninstrumented runtime.
Ranges across generated seeds are descriptive, not confidence intervals.
Seam phase scanning uses a common 64-cell period on full 384-square planes,
including across window-size models; periodic wrap jumps are separately saved.
A low seam score alone cannot win a screen.

```bash
# On Perlmutter, from the repository, with module and virtual environment ready:
export PYTHON_BIN="$PWD/.venv/bin/python3"
export SLICE_RUN="$PWD/runs/pilot"
sbatch --account=YOUR_GPU_ACCOUNT --export=ALL scripts/diagnostics.slurm
```

Queue choice is left to the user. Command-line sbatch flags can override QOS and
time directives; the account and current queue rules must be checked for your
allocation. This script does not retrain or read test.npy. Output goes to a new
`diagnostics-JOBID` directory. Bring back diagnostics.json, diagnostics.png,
summary.md, source-commit.txt, environment.txt and the job log. Full fields are
optional via --save-fields; they are not required for this first screen.

## Stage 2: controlled training screen

A job array caps concurrency at one GPU job by default. All runs start from
scratch with the SAME training seed, batch size 8, optimizer, learning rate and
prepared dataset. Run directories include array ID, arm and training seed.

| Index | Name | Width / parameters (3 channels) | Training window | Updates |
| --- | --- | --- | --- | --- |
| 0 | baseline replication | 32 / 473,187 | 64 | 1,000 |
| 1 | longer | 32 / 473,187 | 64 | 5,000 |
| 2 | wider | 64 / 1,886,403 | 64 | 5,000 |
| 3 | context128 | 32 / 473,187 | 128 | 5,000 |
| 4 | wholeplane | 32 / 473,187 | 384 | 5,000 |
| 5 (optional) | large | 192 / 16,939,587 | 64 | 5,000 |

The default array is 0–4. Index 5 requires an explicit --array override and is
not needed to establish the small-model trend. Each run is followed by the same
checkpoint diagnostics (four generated seeds and eight probe batches for this
screen). Checkpoints are saved every 1,000 updates and at the end. Intermediate
checkpoints can be diagnosed separately. Exact interrupted-training resume is
not implemented; periodic saves preserve partial experimental value, not an
automatic restart guarantee. Longer/wider cases isolate duration/width against
the appropriate control. Compare their 1,000-update checkpoints if needed.

**Context caveats:** increasing crop area changes pixel exposure, compute,
padding effects and GroupNorm's spatial statistics, not just context. The
convolutional receptive field does not become the full plane simply by using a
larger window. The whole-plane case tests removal of internal inference tiling
under matched training/inference extent; it is not a global-attention model.
Sampling shifts for a whole-plane window roll the domain before applying the
network, so they still alter where zero-padding acts. Spectra remain comparable;
seam scores must also be interpreted relative to each model's real boundaries.

Window-size comparisons are matched on updates/images, NOT compute or pixels.
Report parameter counts, runtime, memory, window area and updates. If context
helps, follow with a pixel-/compute-budget control before claiming its cause.
Different widths do not share exactly the same initialization/noise draws;
training seeds are a blocking convention, not a claim of identical stochastic
paths across architectures.

```bash
export PYTHON_BIN="$PWD/.venv/bin/python3"
export SLICE_RUN="$PWD/runs/pilot"
sbatch --account=YOUR_GPU_ACCOUNT --array=0-4%1 --export=ALL scripts/training_screen.slurm
# Optional near-17M arm only:
# sbatch --account=YOUR_GPU_ACCOUNT --array=5 --export=ALL scripts/training_screen.slurm
```

Start stage 1 first. Then screen stage 2 (or a smaller array subset), using its
fixed solver/schedule as a common control across ALL models. Do not compare one
model's best schedule with another model's untuned control. Keep the held-out
test data untouched. Use the screen to choose a hypothesis, then repeat the
relevant comparison with TRAIN_SEED=43 and 44 and at least eight generated seeds.
Training and sampling seeds are different sources of uncertainty.

## How this changes the next decision

- Coarse solver causes drift/spectral errors: improve sampling before retraining.
- Low-noise or high-noise path errors dominate: test time embedding/objective
  changes only after confirming the pattern across batches and checkpoints.
- Longer training fixes the problem: establish duration and checkpoint selection
  first, rather than attributing an undertrained control to capacity.
- Wider training helps validation/path error AND generated statistics: pursue
  capacity with replicated seeds and measured compute budgets.
- Larger windows help while widening does not: prioritize cross-patch/global
  information, while isolating padding and normalization in follow-ups.
- All controls retain similar drift: investigate normalization layers, time
  representation and mean/low-mode prediction with a precise ablation.

No result here justifies assuming the proposed collaborator method, inventing a
geometric input, or adding a spectrum penalty without a defined hypothesis.

After the array finishes, collect completed arms with a common sampler:

```bash
python3 collect_screen.py runs/pilot/screen-ARRAY_JOBID
```

This writes screen-summary.json and screen-summary.md and refuses to combine
reports with different manifest hashes. Bring these plus per-run diagnostics.json,
diagnostics.png and training.json. Inspect failed array tasks in their logs;
the collector includes completed arms only and does not imply missing arms passed.

## Replicate the window comparison before redesigning the model

Submit `scripts/replicate_windows.slurm` to run six tasks, at most one at a time:

| Array task | Training seed | Training window | Screen arm |
| --- | --- | --- | --- |
| 0 | 43 | 64 | longer |
| 1 | 43 | 128 | context128 |
| 2 | 43 | 384 | wholeplane |
| 3 | 44 | 64 | longer |
| 4 | 44 | 128 | context128 |
| 5 | 44 | 384 | wholeplane |

Every task uses width 32 (473,187 parameters), 5,000 updates, batch 8 and the
existing dataset normalization/split. It invokes the existing screen script
without changing Slurm's array IDs. Four identical generation seeds and the
same ten sampler settings match the original seed-42 screen; generated seeds
are not additional independent training replicates. Time probes also retain
the original eight batches. A later larger generation ensemble can refine
sampling variability, but does not replace multiple training seeds.

```bash
export PYTHON_BIN="$PWD/.venv/bin/python3"
export SLICE_RUN="$PWD/runs/pilot"
sbatch --account=YOUR_GPU_ACCOUNT --export=ALL scripts/replicate_windows.slurm
# Once ALL SIX tasks finish, replace ARRAY_JOBID below:
python3 collect_screen.py runs/pilot/replication-ARRAY_JOBID
```

Outputs go under a new replication-ARRAY_JOBID directory, leaving prior screens
intact. Existing training seed 42 plus these runs gives three training seeds
per window. Use common sampler controls and compare each window contrast
within each training seed, not a different selected sampler per model. Three
training seeds remain a small exploratory sample, not a precise population
estimate. Do not change the held-out test set or tune new objectives on it.

Check all six Slurm task exit states: the collector only lists completed
reports. Package their training.json, diagnostics.json and diagnostics.png for
joint comparison with the earlier seed-42 archives. Queue overrides remain
user-specific, as with the original screen. A dry-run dispatch check is available
via REPLICATION_DRY_RUN=1 without allocating a GPU or loading a module.

## Whole-plane normalization × boundary factorial

The window replication found persistent seed-dependent component means and
spectral errors even without internal patch tiling. These observations motivate
a hypothesis test, not a conclusion about the cause. The next factorial trains:

| Treatment | Model normalization | 3×3 convolution padding |
| --- | --- | --- |
| Control | GroupNorm | Zero |
| Boundary only | GroupNorm | Circular |
| Normalization only | Identity (no normalization layer) | Zero |
| Combined | Identity | Circular |

All treatments use full 384-square planes, width 32, batch 8, 5,000 updates,
AdamW at 1e-4, the same prepared data and training-only data normalization.
Removing model GroupNorm does NOT remove dataset standardization. The control
has 473,187 parameters; identity variants have 471,907 because the GroupNorm
affine parameters are removed. Within a training seed, all convolution weights
start identically across the four treatments and training crops/noise/time draws
are matched. This is possible because GroupNorm/Identity initialization does not
consume random draws. Defaults and old checkpoint state-dict keys are preserved.

The array has twelve tasks, one at a time: tasks 0–3 run the four treatments in
table order with seed 42; tasks 4–7 use seed 43; tasks 8–11 use seed 44. Run names
include both factors and seed. Existing run directories are rejected to prevent
accidental overwrite. Checkpoints are saved every 1,000 updates and at the end.
Every run receives the same ten sampling diagnostic arms, four generation seeds
4242–4245, and eight training/validation probe batches. Test data remain untouched.

**Physical boundary guard:** circular padding is only allowed when the training
and inference patch is the entire square plane. Arbitrary local crops are not
periodic simulations. All 3×3 convolutions, including coarse feature maps, use
the selected padding; 1×1 output convolution needs no padding. Pooling and
upsampling remain unchanged. Two stride-two pooling layers mean arbitrary
one-cell rolls can still change the downsampling phase, even with circular
padding. Tests verify roll covariance for shifts divisible by four, not full
translation covariance. Thus shift-policy effects need not vanish in this arm.

Primary comparisons use FIXED Euler64, isolating the boundary treatment without
randomly rolling the domain. All-shift and late-shift Euler64 are secondary,
predefined comparisons with the same schedule across treatments. Effects on
mean drift, centered/uncentered energy, full spectra and scale bands matter
jointly. A reduction in pooled component mean alone can hide cancellation; we
also compare mean-flow energy `total energy - plane-centered energy` against the
same statistic on real planes. Nonzero real plane means are legitimate: this
is not a constraint forcing every sample's mean to zero.

`collect_factorial.py` requires all twelve completed runs, identical reference,
manifest, optimizer/training settings and generation-seed/sampler combinations.
It produces per-run outcomes, ranges over training seeds and within-seed main
effects/interaction contrasts. Main effects average each factor's change over
both settings of the other factor; interaction is the difference of the padding
effects. These are descriptive contrasts for three training seeds, not significance
claims. No automatic winner or new loss/mean correction is selected.

The same optimizer is held fixed to make a controlled screen. If removing
normalization destabilizes training, that does not establish its impossibility:
report the failure and separately plan optimization controls. Nonfinite training
loss fails explicitly instead of silently producing a successful report. Circular
CUDA training under the existing deterministic setting is not verified locally;
if it errors, retain the log rather than silently disabling reproducibility.

```bash
export PYTHON_BIN="$PWD/.venv/bin/python3"
export SLICE_RUN="$PWD/runs/pilot"
# Choose your account and queue explicitly. Example for an exhausted GPU project:
sbatch --account=YOUR_GPU_ACCOUNT --qos=shared_overrun \
  --time-min=00:15:00 --export=ALL scripts/norm_boundary.slurm
# After all twelve tasks finish:
python3 collect_factorial.py runs/pilot/factorial-ARRAY_JOBID
```

The job has a 15-minute requested maximum per task based on prior full-plane
runs of about six minutes. New treatments may run differently; timeout is an
execution failure, not a scientific result. If timeout occurs, inspect the log
and increase the next submission's time budget explicitly. Source/environment
records and per-treatment training time/memory are saved. Bring back
factorial-summary.json/md and each treatment's training.json, diagnostics.json,
and diagnostics.png. Existing screens remain available for control replication.

## Learned mean-flow branch (after the boundary factorial)

Use the GroupNorm + periodic whole-plane model as the control. The hypothesis is
that explicitly modeling the zero spatial Fourier mode can improve the
seed-dependent sample-mean distribution without sacrificing centered structure.
It does not itself add a mechanism for the other missing low-frequency modes.

Two architectures are trained at seeds 42,43,44:

- `implicit`: unchanged model, 473,187 parameters.
- `learned`: same backbone plus a 4→64→64→3 SiLU MLP (4,675 parameters), total
  477,862. Inputs are the three normalized component spatial means of x_t and
  scalar time t. Its output is the normalized mean of the flow velocity.

For learned mode, the output is `backbone(x,t) - spatial_mean(backbone(x,t)) +
mean_head(spatial_mean(x),t)`. The backbone still receives the original state;
only its output mean is separated. This is a learned branch, not a constraint
forcing generated fields to zero mean and not a postprocessing correction.
Whole-plane training and inference are enforced for this ablation. Local-patch
mean heads would predict a different quantity and are rejected by the CLI and
sampler guard.

Keep the ORIGINAL ordinary pixelwise MSE objective and optimizer. Algebraically,
its value is the sum of centered-field MSE and spatial-mean MSE. There is no new
loss weight, spectrum penalty or clipping. A new branch and output decomposition
change representation/optimization together; a successful result would motivate
further isolation rather than prove why GroupNorm failed. The extra parameter
count is reported. No exact reproduction of another fluid method is claimed.

Within each seed, backbone initialization AND the following training noise/time
RNG stream match. The new MLP is initialized in a forked CPU RNG scope; it does
not advance the parent state. It is still randomly initialized, not trained in
advance or manually fitted to validation data. Old checkpoints load with the
implicit default.

The six-task array maps 0/1 to implicit/learned seed42, 2/3 to seed43, 4/5 to
seed44. Every run uses width 32, GroupNorm, circular padding, full 384-square planes,
batch 8, 5,000 updates, AdamW at 1e-4 and checkpointing every 1,000 updates. Standard
diagnostics remain ten arms × four generation seeds, with eight probe batches.
Fixed Euler64 is the primary comparison. Test data remain untouched.

Additionally, each run samples 64 fixed-Euler64 fields with seeds 4242–4305.
The four existing fields' means are reused, and only the remaining 60 need extra
sampling. No extra FFTs or field files are needed for this ensemble. Save all
per-component means and their mean/std/covariance, 5/50/95 percentiles and
mean-flow energy. Compare against all 128 validation plane means with their
snapshot-time and plane labels. Marginal one-dimensional Wasserstein distances
are reported in dataset velocity units and normalized by TRAINING component
std (avoiding division by near-zero reference mean variation). Distances are
descriptive: generated samples and real correlated planes have finite, unequal
sample counts. Marginal agreement cannot establish the correct joint law or
coupling to spatial structure. We do not give confidence intervals from planes
or force the reference distribution to zero mean.

Success requires repeatable improvement in the mean distribution across training
seeds while preserving centered spectra, large-scale energy and fine-scale
behavior. Correct mean-flow energy with collapsed sample means is not success.
If the head helps only one component/seed or worsens centered structure, report
the tradeoff. If it fails, possible explanations include limited head inputs,
time representation or optimization—not proof that explicit mean prediction
cannot work. Keep all comparisons against the same data and sampler.

```bash
export PYTHON_BIN="$PWD/.venv/bin/python3"
export SLICE_RUN="$PWD/runs/pilot"
sbatch --account=YOUR_GPU_ACCOUNT --qos=shared_overrun \
  --time-min=00:15:00 --export=ALL scripts/mean_branch.slurm
# After all six tasks finish:
python3 collect_mean_branch.py runs/pilot/mean-branch-ARRAY_JOBID
```

The requested maximum is 30 minutes per task, with a 15-minute minimum to leave
room for the larger generation ensemble. Actual assigned walltime and runtime
must be checked on Perlmutter. The collector requires all six completed models,
matching data/references, training settings and sampling seeds. It separates the
four-seed spectral screen from the 64-seed mean ensemble and reports per-seed
learned-minus-control contrasts. New outputs go under mean-branch-ARRAY_JOBID;
prior factorial outputs are preserved. Bring back mean-summary.json/md plus each
run's training.json, diagnostics.json and diagnostics.png.
