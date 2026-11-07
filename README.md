# Fluid slice baseline

A small experimental generator of planar velocity cuts from periodic 3D fields.
This is an independently implemented, single-resolution flow-matching control.
It is not a reproduction of another fluid method or of Pyramid Flow. No dataset,
weights, or unpublished collaborator details belong in this repository.

## First experiment and physical meaning

Use full `xy` planes, with tensor order `[channels,y,x]` and channels `u,v,w`.
Confirm that `u,v,w` mean velocity along physical `x,y,z`, respectively. Start
with four equally spaced z planes per snapshot, at z indices 0,96,192,288 for
384 cubed data. Treat these planes as correlated observations, not independent
simulations. Later compare `u,v` with all else fixed; its energy omits `w` and
must not be compared directly with three-component energy.

A slice is not a simulation of two-dimensional turbulence. Although the volume
may satisfy `du/dx + dv/dy + dw/dz = 0`, a plane can have nonzero
`du/dx + dv/dy = -dw/dz`. Do not project it onto a divergence-free 2D field or
penalize this quantity toward zero. `dv/dx-du/dy` is only the normal component of
3D vorticity; full vorticity and dissipation require missing normal derivatives.

Full planes inherit periodic boundaries. Small training crops do not become
periodic across their own opposite edges. We use ordinary zero-padded patch
convolutions, random crop origins, and modular extraction at the domain edge.
Global inference rolls the full periodic plane, tiles nonoverlapping windows,
assembles predictions, reverses the roll, and makes ONE global Euler update.
There is no boundary blending or separate per-patch noise.

## Why this starting point

The inspected 3D project's loader uses NetCDF `u,v,w` and nominal ten-timestep
files, inferring duplicate boundary snapshots from consecutive filenames. Its
reports show shifting can reduce seams while lowering energy. Its reported
short training run also spent most wall time outside GPU computation. Therefore
we export a bounded set of planes once, retain provenance, and test seams and
energy together. The earlier index ranges 200–327 and 0–15 are historical
context, not an approved split for this implementation. The earlier volume
normalization is deliberately not reused.

The control learns a vector field from noise to data:
`x_t=(1-t)noise+t*data`, loss `mean((v_theta(x_t,t)-(data-noise))**2)`.
Sampling integrates from t=0 to t=1 with uniform Euler steps. This follows the
straight Gaussian-path idea in [Flow Matching](https://arxiv.org/abs/2210.02747).
Shifting nonoverlapping windows is motivated by
[SpotDiffusion](https://arxiv.org/abs/2407.15507), whose original application is
panorama generation. Our two-axis, periodic fluid sampler is an adaptation of
that idea, not the original diffusion pipeline. We defer pyramid stages,
geometric conditioning, and spectral losses to keep this control interpretable.

The velocity regressor is a small two-level encoder/decoder with time as an
extra channel, group normalization and nearest-neighbor upsampling. Width is a
controlled capacity knob. There is no pretrained model, coordinate embedding,
attention, incompressibility constraint, or physics loss.

## Local setup and tests

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
.venv/bin/python3 -m pytest -q
MPLCONFIGDIR=runs/matplotlib .venv/bin/python3 scripts/smoke.py
```

The smoke command creates synthetic random data, trains three steps, and runs
paired samplers. It validates plumbing only. Generated fields and checkpoints
are ignored by Git. Load only checkpoints you trust.

## Data inventory and provisional split

Run on the machine holding the data:

```bash
export FLUID_DATA_ROOT=/your/dataset/directory
mkdir -p runs/pilot
python3 inspect_data.py "$FLUID_DATA_ROOT" --out runs/pilot/inventory.json
```

Inspect numeric physical time coordinates, units, dimensions, and missing files.
The initial adapter expects numeric times; date-based coordinates require an
explicit conversion convention first. Do not infer equal time spacing from
file numbers or renumber available files to close gaps. Select one canonical
record for each duplicated physical time across files. Confirm whether distinct
coordinates can still contain duplicate physical snapshots.

Create `runs/pilot/manifest.json` using this schema (illustrative times and
filenames only; replace with verified inventory values):

```json
{
  "status": "provisional_pending_time_and_lab_review",
  "axes": ["z", "y", "x"],
  "dims": ["time", "z", "y", "x"],
  "channels": ["u", "v", "w"],
  "shape": [384, 384],
  "slice_indices": [0, 96, 192, 288],
  "spacing": 0.016362462547794956,
  "minimum_time_gap": 10.0,
  "snapshots": [
    {"file":"fluid_a.nc", "time_index":0, "time":0.0, "split":"train"},
    {"file":"fluid_b.nc", "time_index":0, "time":20.0, "split":"val"},
    {"file":"fluid_c.nc", "time_index":0, "time":40.0, "split":"test"}
  ]
}
```

`dims` maps physical `[time,z,y,x]` to the actual NetCDF dimension names;
`axes` declares the physical interpretation. Spacing is one uniform physical
in-plane grid spacing, in dataset length units. The example spacing came from
the prior workflow and needs lab confirmation; it is not inferred here.

Prefer chronological contiguous train, validation, and test time blocks, with
excluded buffer intervals between them. Every slice/crop/orientation from a
snapshot stays in its snapshot's split. Select the guard gap from temporal
correlation measurements and lab practice, rather than treating the example
gap as sufficient. A single trajectory cannot establish independence across
runs. Keep the status provisional until these choices are approved. Do not
use the test block repeatedly to choose models or schedules.

The manifest rejects duplicate times/records and cross-split gaps smaller than
`minimum_time_gap`; it checks each requested time against the actual file
coordinate. It cannot discover duplicates with misleading coordinates or prove
independence. Export records the manifest hash and computes per-channel means
and population standard deviations from training planes only, in float64.
Validation/test use those same statistics. Disk arrays contain normalized data;
all reported physical metrics undo normalization first.

```bash
python3 run.py prepare --manifest runs/pilot/manifest.json \
  --root "$FLUID_DATA_ROOT" --out runs/pilot/data
```

## Perlmutter pilot

Use your existing PyTorch module or an environment containing requirements.txt.
Set your own account when submitting; no account or private paths are embedded.
Before submitting, run the tests and the synthetic smoke command in that environment.

```bash
module load pytorch/2.6.0
python3 -m pytest -q
MPLCONFIGDIR=runs/matplotlib python3 scripts/smoke.py
export SLICE_RUN="$PWD/runs/pilot"
# Prepare data using the manifest command above before submitting.
sbatch --account=YOUR_GPU_ACCOUNT --export=ALL scripts/perlmutter.slurm
```

The job trains 1,000 updates, width 32, 64-square patches, batch 8, seed 42;
then samples eight pairs of 384-square fields with 32 steps. Both arms share
the exact checkpoint and initial noise; shift randomness has a separate RNG.
CUDA synchronization surrounds runtime measurement; peak allocated CUDA memory
is reset per arm. Times are end-to-end sampler times, including tiling. They
are not a rigorous throughput benchmark; first-call warmup may affect them.

For a manual run or a larger step-budget check:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 run.py train --data runs/pilot/data --out runs/pilot/train \
  --device cuda --steps 1000 --width 32 --patch 64 --batch 8
python3 run.py evaluate --data runs/pilot/data \
  --checkpoint runs/pilot/train/checkpoint.pt --out runs/pilot/eval \
  --device cuda --steps 32 --samples 8 --seed 4242
```

Bring back `inventory.json` (remove sensitive filenames if needed), the manifest
and time-unit/axis confirmation, `normalization.json`, `training.json`,
`evaluation.json`, `comparison.png`, `environment.txt`, and Slurm wall time/job
status. No data or checkpoints need to be committed. The repository's local
virtual environment and the Perlmutter module may differ; preserve each run's
package versions and source commit. CUDA results are not promised bitwise equal
to CPU results.

## Decisions attached to measurements

1. **Data gate:** verify axes, physical times, plane dimensions, training mean
   near zero/std near one after normalization, and full-plane wrap differences
   comparable with neighboring differences. Periodic does not mean first and
   last stored points have equal values. Stop before real training if unclear.
2. **Optimization gate:** compare training and fixed validation flow loss. The
   validation probe uses fixed crops/noise at t=0.5, so it is a diagnostic rather
   than a full validation integral. Train longer or overfit a small subset if
   optimization fails; do not diagnose architecture capacity from three steps.
3. **Patch gate:** compare fixed and shifted pairs. A phase-scanning seam ratio
   takes the largest mean squared neighboring jump over patch phases, divided
   by the overall interior jump mean, separately along x and y. Check wrap jumps
   too. A lower seam score is useful only alongside acceptable component
   variance and energy relative to real planes. Real fields also have nonunit
   finite-sample seam ratios; compare against their distribution.
4. **Scale gate:** compare per-component std, total and fluctuation kinetic
   energy, low/high spectral bands, the radial spectrum, and second-order x
   increments. The shell-summed planar FFT spectrum uses angular wavenumber and
   Parseval normalization: its sum equals fluctuation energy. It is a planar
   spectrum of three velocity components, NOT a 3D spherical-shell HIT spectrum
   or evidence of a 2D cascade. Use identical spacing, planes, channels, and FFT
   convention for real/generated comparisons. The high band starts at one
   quarter of grid width in integer radial bins; low band is bins 1–4. These
   are explicit screening bands, not universal physical thresholds.
5. **Capacity gate:** only after adequate optimization, compare widths 16,32,64
   at matched data, patch size, seed set, steps, batch size and sampler settings;
   also report wall time and consider matched-compute comparisons. Parameter
   counts are saved for every training run; `python3 scripts/count_parameters.py`
   lists the available width scales. Use multiple training seeds before
   conclusions. A larger model helping validation and low-frequency energy
   supports further capacity work; lower training loss alone does not.
6. **Formulation gate:** if energy is still low, compare with whole-plane
   inference as a diagnostic and with larger training windows before adding
   conditioning or spectral losses. Larger inference windows alone are out of
   training distribution. Local sampling cannot guarantee domain-scale
   correlations; shifted boundaries are not a global-context mechanism.

In-plane divergence and normal vorticity RMS are descriptive slice statistics,
not full 3D incompressibility or enstrophy claims. Current outputs include
component means/std, energies, spectra, seam/wrap ratios and second-order
increments; velocity PDFs and uncertainty intervals are follow-up analysis.
Aggregate and bootstrap by snapshot, not by plane, when reporting reference
uncertainty. Eight generated pairs are a pilot, not a conclusive study. Paired
PSNR/SSIM is omitted because unconditional outputs have no corresponding target.

## Details needed before reproducing a proposed fluid method

Ask its author for a paper/preprint and approved code or pseudocode; exact
noise/data path and objective; architecture and parameter accounting; definition
of geometric input (coordinates, distances, positional encoding or other);
where conditioning enters; patch overlap/shift/boundary rules; global context
exchange; meaning of full-spectrum behavior and its evaluation; training data,
normalization, time split, sampling schedule and compute. Obtain permission
before recording or publishing unpublished details. A model-size anecdote is
not evidence of the right capacity, and this baseline makes no reproduction
claim. Keep any private meeting notes outside version control.
