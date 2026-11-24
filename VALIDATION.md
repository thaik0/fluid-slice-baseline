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

Normalization/boundary factorial implementation validation:

- Forty-one tests pass, including all four forward/backward variants, legacy
  state-dict/default compatibility, identical convolution initialization across
  treatments, periodic edge impulse behavior and roll covariance on the
  four-cell downsampling sublattice, crop-domain guard, twelve array mappings,
  incomplete/mismatched factorial rejection and known interaction arithmetic.
- All four variants completed two synthetic training steps and the full ten-arm
  diagnostic pipeline with finite reports and correct saved model configurations.
- An existing pre-ablation local smoke checkpoint loads strictly in the updated
  model. New launcher shell syntax and Python compilation pass.
- Real 384-square training, 5,000-update stability, circular-padding CUDA backward
  behavior under deterministic mode, GPU memory/runtime and scientific treatment
  effects have not been validated here. No actual factorial was submitted locally.

Learned mean-branch implementation validation:

- Fifty-one tests pass, including legacy defaults/checkpoints, identical backbone
  initialization and parent RNG state across mean modes, exact predicted-output
  mean ownership, ordinary-MSE mean/centered decomposition, finite gradients,
  full-plane sampler guard, six array mappings and collector completeness/settings
  checks. A distribution test detects collapse even at correct mean-flow energy.
- Both implicit and learned variants completed three synthetic training updates,
  all ten sampling diagnostics and an eight-sample mean ensemble. Reports are
  finite, reference plane labels align, and saved configurations load strictly.
- Parameter counts verified: 473,187 control; 477,862 learned branch. Python
  compilation and launcher shell syntax pass.
- The 384-square 5,000-update GPU comparison and 64-sample real-data ensembles
  have not been run locally. Scientific effects, stability, CUDA behavior and
  runtime must be checked on Perlmutter. The synthetic smoke establishes execution
  only; it is not evidence of improved turbulence generation.

Budgeted coarse-pilot implementation validation:

- Fifty-four tests pass. New tests verify Fourier amplitude/phase and DC,
  high-frequency and target-Nyquist rejection, common-shell energy conservation,
  physical spacing/domain preservation, coarse training-only normalization,
  validation-change isolation, provenance, reserved-test isolation, overwrite
  rejection and DC-free low-pass visualization masks.
- A synthetic 384→96 preparation completed; common physical shells1–16 matched
  to relative error below6e-16 before float32 storage. A three-update width8
  training smoke and eight fixed64 samples completed with finite reports,
  two saved generated fields and both visualization artifacts.
- Launcher shell syntax/Python compilation pass. Execution without an explicit
  funded account, or with a different submitted account, fails before module
  loading/training. No job was submitted and no GPU allocation was used locally.
- Real source preparation,5,000-update CUDA execution, actual resource footprint
  and scientific low-k recovery remain to be checked on Perlmutter. This is a
  new coarse single-resolution task, not a validated pyramid or a pure
  resolution-only architectural attribution.

Coarse replication launcher validation:

- Fifty-six tests pass. The new dry-run test checks the two-seed sequential,
  one-GPU,15-minute TOTAL budget plan. Collector tests verify complete matched
  three-seed aggregation and refusal of mismatched manifests/incomplete outputs.
- Shell syntax and Python compilation pass. Missing/mismatched account checks
  fail before module loading or GPU work. No actual GPU job was submitted here.
- The launcher reuses the previously verified training/evaluation settings; real
  seed43/44 outcomes and total job runtime remain to be measured on Perlmutter.

Real-coarse conditional refinement validation:

- All59 tests pass, including analytic amplitude/phase/DC preservation,
  Fourier interpolation/decimation round trips, projector orthogonality and
  differentiability, forced low-mode removal during sampling even for an
  intentionally unsuitable velocity model, conditioning-shape errors, and an
  end-to-end synthetic train/evaluate run. Fine source tampering is rejected;
  the tests run without a reserved test array.
- A separate384→96 synthetic run completed CPU geometry checks, two width4
  training updates and fixedEuler sampling. Maximum stored-context error was
  below5e-8 dataset units; roundtrip/decomposition errors were below7e-16;
  normalized low-mode sampling error was below3e-7. These establish numerical
  geometry and execution only, not learned turbulence quality.
- Launcher shell syntax and Python compilation pass. One GPU, one seed,
  15-minute total cap; no automatic retries. No GPU job was submitted locally.
  CUDA runtime/memory at width32/batch8, convergence after5,000 updates, and
  scientific residual quality remain unverified until the remote pilot runs.
