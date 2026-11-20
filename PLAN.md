# Study plan

Amos Bunde. Written 20 November 2026, before any workload was measured and before the
simulator was written. Results are added at the end of this file once the study has run.
Anything I change after seeing validation results goes under "Amendments".

## Question

Can the runtime of a tensor workload on a device be predicted from an operator trace and a
small description of the device, without running the workload there? And when can such a
prediction be trusted: only for workloads like the ones already measured, or also for
larger ones, different architectures and device configurations that were never measured?

## What will be built

1. **Trace capture.** A PyTorch model and an input shape become a list of operators with
   shapes, dtype, parameter count, FLOPs and bytes read and written. Capture uses
   `torch.fx`. Training steps are derived from the forward trace plus a parameter update.
2. **Device descriptor.** A small JSON file: cores, cache sizes, memory bandwidth, and for
   each operator class a peak rate, a per-call overhead, a serial fraction for thread
   scaling and a measured backward-to-forward cost ratio. It is fitted from
   microbenchmarks of single operators. It never sees a whole workload.
3. **Executor.** A discrete-event replay of the trace against the descriptor. Each
   operator costs its call overhead plus the larger of a compute time and a memory time.
   The output is a predicted time per operator, a total, and a split of the total into
   compute, memory and overhead.

The details of the device model may change while I build it against the microbenchmarks.
It is frozen, and committed, before the validation matrix is measured.

## Workloads and device configurations

Four architecture families, each written from primitive operators so the trace is complete:
an MLP, a residual CNN, a transformer encoder and an MLP-Mixer. Each family has a grid of
about 24 workloads that varies width, depth, batch size and sequence length or image size.
Every workload is measured as a forward pass and as a training step (forward, backward and
an SGD update).

The only hardware is one laptop CPU with four physical cores and no GPU. A "device
configuration" is that CPU with 1, 2 or 4 threads and float32 or float64 arithmetic, so six
configurations. A workload in one mode on one configuration is a "cell".

Each cell is timed after warm-up, repeated at least three times, and summarised by the
median. A subset of cells is measured a second time later in the run to estimate how much
the measurement itself moves. That figure is the noise floor for every error below.

## Predictors

- **Simulator.** Calibrated on single-operator microbenchmarks only.
- **Simulator with fitted corrections.** The same, with one multiplicative correction per
  operator class fitted on the training workloads of each test design.
- **FLOPs-proportional.** Time is a constant times total FLOPs, one constant per device
  configuration and mode.
- **Gradient boosting.** A gradient-boosted tree regressor on trace features (totals of
  FLOPs, bytes, parameters and operator counts by class, plus threads, dtype and mode),
  predicting log time.
- **Log-linear regression.** A ridge regression on the logs of the same features. Trees
  cannot predict outside the range of their training targets, so this is the fairer
  black-box opponent when the test asks for extrapolation.

## Test designs

- **(a) Random.** Five-fold cross-validation over workloads. All cells of a workload stay
  together.
- **(b) Size.** In each family, fit on the smaller half of workloads by FLOPs and test on
  the larger half. The simulator's microbenchmarks are cut off, for each operator class,
  at the largest operator seen in the training half, so it does not get to see large
  operators either.
- **(c) Family.** Leave one architecture family out, four folds.
- **(d) Thread count.** Fit on cells with 1 and 2 threads, test on cells with 4 threads.
  The simulator is calibrated only on 1- and 2-thread microbenchmarks.

## Metrics

- Primary: mean absolute percentage error (MAPE) of predicted against measured time over
  test cells.
- Rank agreement: Spearman correlation over test cells, and Spearman correlation across
  the six device configurations within each workload.
- Decision quality: for each workload and mode, the predictor chooses the thread count it
  expects to be fastest at float32. Regret is the measured time of that choice over the
  measured best, minus one.

Intervals are 95% bootstrap intervals that resample workloads, not cells, because the
cells of one workload are not independent. Differences between two predictors are paired.

## Hypotheses

Dated 20 November 2026. "Simulator" means the version without fitted corrections.

- **H1. The black box wins at interpolation.** In design (a), gradient boosting has a lower
  MAPE than the simulator, and the interval of the paired difference excludes zero.
- **H2. The simulator wins at size extrapolation.** In design (b), the simulator has a
  lower MAPE than gradient boosting, and the interval of the paired difference excludes
  zero. I also report the comparison with log-linear regression. If log-linear regression
  matches the simulator here, the advantage comes from functional form and not from
  modelling the machine, and I will say so.
- **H3. The simulator wins on an unseen architecture.** In design (c), the simulator has a
  lower MAPE than gradient boosting in at least three of the four families.
- **H4. The simulator wins on an unseen thread count.** In design (d), the simulator has a
  lower MAPE than both gradient boosting and log-linear regression.
- **H5. The simulator is good enough to choose a configuration.** Its chosen thread count
  is within 10% of the best measured time for at least 90% of workload-mode pairs.
- **H6. FLOPs alone are a poor predictor.** The FLOPs-proportional baseline has the highest
  MAPE of all predictors in every design, and over 50% in design (a).
- **H7. Absolute accuracy.** Over all cells the simulator's MAPE is below 30%.

Either answer to H2 to H4 is a finding. If the black-box regressors extrapolate as well as
the mechanistic model, that is worth knowing.

## Forward-looking demonstration

After validation I will use the simulator to project runtimes on devices I do not have: one
with twice the memory bandwidth and ones with 8 and 16 cores. These projections cannot be
checked here and will be labelled as unvalidated.

## Limits known in advance

- One CPU. Nothing will be measured on a GPU or any other accelerator, and nothing will be
  claimed about them. The descriptor format is meant to extend to accelerators but that is
  untested.
- Other jobs share the machine and it is a laptop part with thermal limits, so timings are
  noisy. Hence the medians, the repeats and the noise floor.
- The microbenchmarks cover the operator classes the four families use. A workload built
  from other operators is outside what was calibrated.
- This plan is not registered anywhere. Its commit date is the only evidence of order.

## Budget

At most four threads at a time, and under about 45 minutes of measurement in total.

## Amendments

### Made before the validation matrix was measured (20 November 2026)

These changes came from building the simulator against the microbenchmarks and from a dry
run of the analysis code on synthetic timings generated from a made-up descriptor. No
measured workload time had been looked at when they were made. The hypotheses are
unchanged.

1. **The device is two pinned cores, not four free ones.** Other jobs kept the machine
   busy throughout. Unpinned, a four-thread operator that normally took 36 microseconds
   sometimes took 3 milliseconds, because it had to wait for the scheduler to free four
   cores at once. The process now pins itself to both hardware threads of two physical
   cores. The thread counts are still 1, 2 and 4, but 4 threads now means two cores with
   hyperthreading. This changes what design (d) asks: it becomes "predict the hyperthreaded
   configuration from the one- and two-thread configurations". The descriptor gained a
   per-class hyperthread yield. Under design (d) that yield cannot be calibrated and is set
   to zero, so the simulator will predict that 4 threads are no faster than 2.
2. **A guard against interference.** Each measurement records how busy other processes
   kept the pinned CPUs while it ran. If that exceeds 0.4 CPUs the measurement is repeated,
   up to four attempts, and the attempt with the least interference is kept.
3. **Four convolution classes instead of one.** In the microbenchmarks the speed per FLOP
   of a convolution differed severalfold between 1x1 kernels, layers with three input
   channels, and 3x3 kernels with and without a stride.
4. **A first-touch term.** An output above the allocator's 32 MiB threshold is mapped
   afresh on every call, and the microbenchmarks showed the cost. The descriptor has a
   rate for it. The memory-bandwidth and first-touch benchmarks describe the machine, not
   an operator class, so design (b) does not cut them off by size.
5. **Log-linear regression uses whole-trace totals only.** With per-class totals it
   produced absurd predictions in the dry run when a class was absent from training. The
   restricted version is the stronger opponent, so the change does not favour the
   simulator.

### Made after the validation matrix was measured

6. **A second analysis on the cells measured at full machine speed.** The second
   measurement pass showed that the machine as a whole ran 1.2 to 2.7 times slower from
   about minute 12 of the first pass until its end. On all cells, two measurements of the
   same cell disagree by 28% on average, which is as large as the errors being compared.
   I therefore ran every analysis twice: on all 1,152 cells as planned, and on the 495
   cells measured in the first ten minutes of the first pass or in the second pass, where
   repeats disagree by 15%. Cells were visited in random order, so the 495 are a random
   subset. The ten-minute cut was chosen after looking at the drift, not before.
7. **A recheck of the calibration.** After seeing the simulator over-predict, I re-timed
   78 microbenchmark cases. They had taken a median of 1.41 times longer during
   calibration. As an exploratory analysis, labelled as such, I report the simulator with
   every prediction divided by that one constant.
8. **The measurement stopped once at a 40-minute limit** with 99 cells and the whole
   second pass still to do, and was resumed. Timed runs took about 65 minutes in total
   (14.5 for microbenchmarks, 50.8 for workloads), over the 45 I had budgeted, not
   counting two earlier microbenchmark runs that were discarded because of interference.
9. **Figures show the simulator both ways**, as calibrated and with fitted corrections.
   The plan did not say which predictors would be drawn.

## Outcomes

Verdicts are computed in `src/tracesim/stats.py` from the committed predictions. "Planned"
is all 1,152 cells. "Clean" is the 495-cell analysis of amendment 6. MAPE in percent.

| | Hypothesis | Planned analysis | Clean windows |
|---|---|---|---|
| H1 | Gradient boosting beats the simulator at interpolation | Not supported: 37.9 against 41.0, interval of the difference includes zero | Supported: 26.2 against 60.8 |
| H2 | The simulator beats gradient boosting from small to large | Not supported: 53.9 against 54.7, interval includes zero | Failed: 83.8 against 55.3 |
| H3 | The simulator beats gradient boosting on at least 3 of 4 unseen families | Supported: 3 of 4 | Failed: 2 of 4 |
| H4 | The simulator beats both regressors on an unseen thread count | Not supported: 32.1 against 31.1 for gradient boosting | Failed: 48.5 against 19.5 |
| H5 | The simulator's thread choice is within 10% of the best in 90% of cases | Failed: 54.2% of 192 choices | Failed: 50.0% of 16 choices |
| H6 | FLOPs-proportional is the worst predictor in every design | Failed: gradient boosting is worse on unseen families | Failed: FLOPs is the worst only from small to large |
| H7 | The simulator's MAPE over all cells is below 30% | Failed: 41.0 | Failed: 60.8 |

One of seven held as planned, and that one did not survive the cleaner data. The
hypotheses were all about the simulator calibrated from microbenchmarks alone. What I did
not predict is in the report: the same simulator with one fitted correction per operator
class was the most accurate predictor under every design.
