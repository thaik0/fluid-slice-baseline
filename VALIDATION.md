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
