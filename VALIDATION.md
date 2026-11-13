# Implementation validation

Local CPU validation on 2026-11-07:

- Eight tests passed: training-only normalization, split guards, coordinate
  mismatch rejection, exact periodic roll/tile/stitch identity, global sampler
  updates and seeded shifts, model gradients, spectrum Parseval normalization,
  and detection of relocated seams.
- Synthetic NetCDF smoke completed preparation, three training updates, and two
  fixed/shifted seed pairs. All four samples and their physical-unit metrics,
  per-plane reference diagnostics, paired changes and comparison PNG were saved.
- The smoke model had 30,171 parameters (width 8). Validation loss at t=0.5 was
  2.4543 initially and 2.4411 at update three. This is an execution check only;
  three updates on random data establish no learning or fluid-quality result.
- Shell syntax check passed for the Perlmutter job.
- Exact local Python packages are in requirements-local-tested.txt. Perlmutter
  PyTorch 2.6/CUDA compatibility is not yet tested; preserve its environment and
  rerun tests there before training.

Architecture counts with three velocity channels:

| Width | Parameters |
| --- | ---: |
| 16 | 119,091 |
| 32 | 473,187 |
| 64 | 1,886,403 |
| 128 | 7,532,931 |
| 192 | 16,939,587 |

No real turbulence data were available locally. Physical axis interpretation,
time-coordinate ordering/units, guard-gap adequacy, real-data input speed,
CUDA memory/runtime, and generated turbulence statistics remain unverified.
No real-data split or scientific conclusion is approved by this validation.

Follow-up validation on 2026-11-09:

- Ten tests pass, including deterministic time-coordinate deduplication/bin
  selection and a physically labeled x,y,z storage-to-y,x plane test with
  coordinate-spacing validation.
- Synthetic preparation/training/paired sampling smoke passes after adding
  preparation statistics. Normalized training means are near zero and standard
  deviations near one.
- Manifest construction was checked against a supplied inventory for both a
  small data check and a larger candidate pilot. This uses metadata only; no
  real velocity arrays were read locally. Dataset-specific manifests and the
  supplied inventory remain outside version control.

Diagnostic implementation validation on 2026-11-13:

- Fifteen tests pass. New coverage checks physical-time shift policies, exact
  compatibility with the original 32-step shifted sampler, known-ODE Heun
  accuracy, deterministic time probes and finite seam metrics for single windows.
- All ten diagnostic arms ran end to end on a synthetic-data checkpoint,
  including reports, paired differences, traces, time probes and PNG generation.
  Reports serialize with nonfinite JSON values forbidden in the verification.
- A two-step synthetic training run saved both periodic checkpoints. The screen
  collector consumed a completed synthetic report successfully.
- Both new Slurm scripts pass bash syntax validation. The five-arm training
  array and real-data diagnostic sweep have not been run on Perlmutter here.
  CUDA compatibility, resource requirements and scientific outcomes remain to
  be checked there. Synthetic execution is not evidence of turbulence fidelity.

Window replication launcher validation:

- All six dry-run dispatch tests verify the expected seed/window-arm pairing
  and the shared replication output root without invoking Slurm or training.
- The existing training launcher still defaults to its original array mapping;
  an explicit SCREEN_ARM override lets the replication array reuse it without
  rewriting Slurm environment identifiers.
- Bash syntax checks pass. GPU execution and scientific replication outcomes
  remain to be verified on Perlmutter.
