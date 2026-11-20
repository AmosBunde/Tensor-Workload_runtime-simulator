# Tensor-Workload_runtime-simulator

**Can you predict how long a neural network will take on a machine without running it
there, and when should you believe the prediction?**

This repository holds a trace-driven simulator and a study of it. The simulator turns a
PyTorch model into a list of operators, prices each one against a small description of a
device (cores, cache, memory bandwidth, a rate per operator class) and adds up the time.
The study measures 96 workloads on a real CPU and asks whether that mechanistic model
predicts workloads unlike the ones it was fitted on better than a black-box regressor
does. It was planned in [PLAN.md](PLAN.md) before anything was measured, and most of what
I predicted there turned out wrong.

## What was found

Four model families (MLP, residual CNN, transformer, MLP-Mixer), 24 sizes each, timed as
forward passes and training steps on six configurations of one CPU: 1,152 cells. Other
jobs slowed the machine for most of the run, so the numbers below use the 495 cells
measured while it ran at full speed. Two measurements of the same cell differ by 15% on
average there, and no error below is meaningful beyond that. The full write-up, including
the analysis of all cells, is in [REPORT.md](REPORT.md).

Mean absolute percentage error of predicted against measured time:

| Test cells are... | Simulator, microbenchmarks only | Simulator, corrections fitted | Gradient boosting | Log-linear regression | FLOPs-proportional |
|---|---|---|---|---|---|
| random workloads | 60.8% | 19.9% | 26.2% | 34.1% | 56.7% |
| larger than any training workload | 83.8% | 27.6% | 55.3% | 46.7% | 174.3% |
| an architecture family never fitted on | 60.8% | 31.2% | 93.7% | 37.8% | 56.0% |
| a thread count never fitted on | 48.5% | 19.2% | 19.5% | 34.0% | 41.4% |

![Prediction error under four test designs](docs/img/headline.png)

- **The simulator calibrated from operator microbenchmarks alone was poor.** It
  over-predicted by about half. A later recheck found that the microbenchmarks had been
  taken while the machine was running a median of 1.41 times slower than normal. A device
  description is only as good as the state of the device when it was written.
- **The same simulator with one fitted number per operator class was the best predictor
  under every design.** Its error rose from 19.9% on random workloads to 27.6% on
  workloads up to 587 times larger than any it was fitted on. Gradient boosting went from
  26.2% to 55.3%; its predictions for the large workloads averaged about a third of the
  measured time.
- **The black-box regressors broke on an unseen architecture.** Gradient boosting's error
  on MLPs it had never seen was 240%.
- **On an unseen thread count nothing separated the simulator from gradient boosting**
  (19.2% against 19.5%). The unseen count was four threads on two cores, and neither had
  any way to know what hyperthreading adds.
- **Every predictor ranked workloads almost perfectly and none could choose a thread
  count.** Rank correlation over test cells was 0.98 or more for the simulator in every
  design. But on all cells its chosen thread count was within 10% of the best in only
  54% of 192 cases, and in 8 of the 16 cases the clean cells allow.
- Of the seven hypotheses in the plan, one held on the planned analysis and none on the
  cleaner data. The hypotheses were about the uncorrected simulator. The report gives
  each one.

This is one laptop CPU, pinned to two cores, shared with other jobs. Nothing was measured
on a GPU or any other accelerator.

## What is in the repository

Written from scratch around PyTorch, NumPy, SciPy and scikit-learn.

| Part | File | What it does |
|---|---|---|
| Workloads | `src/tracesim/zoo.py` | Four model families built from primitive operators, and the grid of 96 workloads |
| Trace capture | `src/tracesim/trace.py` | Turns a module and an input into an operator trace with `torch.fx`: classes, shapes, FLOPs, bytes |
| Device descriptor | `src/tracesim/device.py` | The JSON description of a device, and edited copies for what-if questions |
| Executor | `src/tracesim/simulator.py` | Prices each operator and replays the trace as a discrete-event simulation |
| Microbenchmarks | `src/tracesim/microbench.py` | Times single operators across sizes, dtypes and thread counts |
| Calibration | `src/tracesim/calibrate.py` | Fits a descriptor to microbenchmark points, or to a restricted set of them |
| Timing | `src/tracesim/timing.py`, `quiet.py` | Median-of-repeats timer, CPU pinning, a guard against interference |
| Measurement | `src/tracesim/measure.py` | Resumable runner for the grid; selects the cells measured at full machine speed |
| Baselines | `src/tracesim/baselines.py` | FLOPs-proportional, gradient boosting, log-linear regression, fitted simulator corrections |
| Test designs | `src/tracesim/evaluate.py` | The four hold-out designs; writes one row per prediction |
| Statistics | `src/tracesim/stats.py` | MAPE, rank correlation, decision regret, bootstrap intervals over workloads |
| Projections | `src/tracesim/whatif.py` | Replays traces against devices that were not measured |
| Figures | `src/tracesim/figures.py` | Every figure in this README and the report |
| The measured device | `devices/host.json` | The fitted descriptor of the machine used here |

## Try it

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). No GPU is used.

```bash
git clone https://github.com/AmosBunde/Tensor-Workload_runtime-simulator
cd Tensor-Workload_runtime-simulator
uv sync --extra dev          # installs CPU PyTorch, SciPy, scikit-learn
uv run pytest -q             # 45 tests, a few seconds
```

Predict one workload on the committed descriptor, without timing anything:

```bash
uv run tracesim capture --workload transformer-d256-d2-s128-b8
uv run tracesim predict --workload transformer-d256-d2-s128-b8 --mode train --threads 2
```

The first prints the operator trace. The second prints a predicted time, its split into
compute, memory, call overhead and threading loss, and the time by operator class.
`uv run tracesim capture --list` gives the 96 workload ids.

## Reproduce the study

The result files from my run are committed, so steps 5 to 7 work straight after cloning.
Steps 1 and 3 time real code and need a quiet Linux machine with at least two physical
cores; they pin themselves to two of them. Times are from my run.

| Step | Command | Time | Writes |
|---|---|---|---|
| 1. Time single operators | `uv run tracesim microbench` | about 15 min | `results/microbench.jsonl` |
| 2. Fit the device descriptor | `uv run tracesim calibrate` | seconds | `devices/host.json`, `results/calibration.json` |
| 3. Time every workload | `uv run tracesim measure` | about 50 min | `results/measurements.jsonl` |
| 4. Recheck the calibration | `uv run tracesim microbench --recheck` | 15 s | `results/microbench_recheck.jsonl` |
| 5. Run the four test designs | `uv run tracesim evaluate` | about 2 min | `results/predictions.csv`, `results/predictions_clean.csv` |
| 6. Project unmeasured devices | `uv run tracesim whatif` | 10 s | `results/whatif.json` |
| 7. Statistics and figures | `uv run tracesim summarise && uv run tracesim figures` | 20 s | `results/summary.json`, `docs/img/` |

Steps 1 and 3 skip anything already in their output file, so they can be stopped and
restarted; delete the file to start again. Step 5 captures 384 traces on first use and
caches them in `data/` (9 MB, not in git).

Timings will not match mine. They depend on the machine and on what else it is doing, and
on mine a repeat of the same cell differed by 15% at best. Expect the same ordering of
predictors if the machine is quiet, not the same numbers. The "clean window" selection in
step 5 is specific to my run: it keeps the first ten minutes of the first pass and the
whole second pass. On a new set of measurements, read `drift` in `results/summary.json`
before trusting either analysis.

## Reading the results yourself

- `results/summary.json`: every number quoted here and in the report. `planned` is the
  analysis of all cells, `clean` the analysis of the full-speed cells.
- `results/predictions_clean.csv`, `results/predictions.csv`: one row per test cell and
  design, with the measured time and each predictor's prediction.
- `results/measurements.jsonl`: one row per timing, with quartiles, the number of
  attempts and how busy other processes kept the pinned CPUs.
- `results/microbench.jsonl`: the 966 operator timings the descriptor was fitted to.

## Limits

- **CPU only.** One Intel i7-10610U, pinned to two cores. The descriptor format (rates
  per operator class, memory bandwidth, a serial fraction) was designed with accelerators
  in mind but has not been tested on one, and I claim nothing about GPUs or TPUs.
- **A noisy, shared machine.** Timings use warm-up, repeats and medians, and each cell
  was repeated when other processes were on its CPUs. That did not catch a machine-wide
  slowdown that lasted most of the run. The clean-window analysis was not planned; I chose
  its ten-minute cut after seeing the data.
- **The calibration was taken on a slowed machine**, which is the likely reason the
  uncorrected simulator over-predicts. I did not recalibrate, so I cannot say how good a
  well-calibrated simulator would have been.
- **Small numbers.** 96 workloads; 48 in the size test; four families, so the
  unseen-architecture test has four folds. The thread-choice result on clean data rests
  on 16 cases.
- **The projections for unmeasured devices are not validated**, and the simulator they
  come from over-predicts the measured host.
- The plan was written first but was not registered anywhere, and I also ran the study.

## Licence

MIT. See [LICENSE](LICENSE).
