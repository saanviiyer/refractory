# Refractory

## Research question

When we say that a world model's dynamics "use" a coordinate, is that a fact about the model, or a fact about which part of the model's state we chose to examine?

The project started on 31 August 2026. It builds on `used_coordinates` and does not change it. The estimator from `used_coordinates` is copied without edits into `src/vendor/`, with checksums in [`MIGRATION_MANIFEST.md`](MIGRATION_MANIFEST.md).

## Spiking networks are the instrument

The name suggests a spiking-networks project, so this point comes first. Spiking networks are the instrument here, and they are not the subject. No claim in this repository is about spikes. The claim boundary below forbids every claim about efficiency, energy, spike count or hardware.

A spiking core is the simplest architecture that forces the analyst to make a visible choice. A GRU has one hidden vector, so the question "which coordinates do the dynamics use" goes to that vector and there is nothing to decide. A current-based leaky integrate-and-fire (LIF) core carries four components of equal standing: membrane potential `v`, synaptic current `i`, spike trace `r`, and the emitted vector `s`. A probing paper could treat any one of them as "the representation". The choice was always being made. A one-vector architecture only hid it.

The project's own controls support this framing. The effect appears in the rate twin (same architecture, same parameters, continuous emission). In the spiking cell it was uninformative both times it was measured. The result is therefore about any model whose state is more than one vector, which describes most models.

## Why the question is open

`used_coordinates` showed that a variable can be read out of a latent state while the model's own transition dynamics ignore it. Its criterion asks the "use" question directly, without labels.

That criterion assumes that there is one latent state to ask about. The parent project never had to name a state variable, because a GRU has only one. The assumption was invisible, and nobody had justified it. This project makes the choice explicit and asks whether the answer survives it.

## Sub-questions and status

**Q1. Does the criterion transfer to a second architecture?** Unresolved. The answer depends on the environment. The spiking model fails the validity gate on the wrapping arm (`v6`, 1 of 8 seeds beat frame persistence). It passes on the bounded arm (`v7`, 8 of 8, ratio 0.4569, replicated on fresh seeds at 0.4749). But on `v7` the criterion's primary view returns nothing, so this project cannot claim that the criterion transfers.

**Q2. Is the verdict a property of the model or of the chosen state variable?** This is the central question. Answer: of the chosen state variable. The test was pre-registered (A4), used fresh seeds and a clean gate, and was scored by code. On the same frozen model and the same actuator, the `s` view returns translation on 8/8 seeds while the `i` view returns rotation on 6/8. Two equal-sized components of one state give opposite answers about which group the action induces.

**Q3. Is that dependence learned, or an artefact of the representation?** Answer: learned. In the untrained control (A5), every view that fits at all agrees on translation at random initialisation. The `s` view fits nothing before training (residual 0.9047) and fits at 0.2821 after. Training creates the disagreement. This excludes the bounded-code explanation, which was the leading alternative.

## What follows

"The dynamics use this coordinate" is under-specified unless the state component is named. A model does not simply have or lack this property. The parent's criterion is still correct on a one-vector architecture, where the choice is forced and the question is well-posed. Its scope is narrower than it looked. Any use of it on a model with structured state must report which component it examined.

The obvious component is the wrong one. The `r` view is the state the decoder reads and the closest analogue of the GRU hidden vector, so most analysts would pick it. It finds nothing anywhere. Its winning residuals range from 0.7495 to 0.9514 across every cell measured, against a no-transform reference of 1.0.

## Controls

**Rate twin.** `RateArmWM` is the same model with the Heaviside step replaced by the surrogate's own forward function. Parameter count, initialisation, gradient form and the four-component state are identical. An effect that appears equally in the twin comes from having a multi-component state and has nothing to do with spiking. `PREREG.md` fixes in advance how large the spiking-minus-twin difference must be before anything is called a spiking result.

**Two budgets.** `snn` matches the GRU's hidden width (128). Every view then has the dimension the criterion was calibrated on, and the variance-matched random-plane null comes from a space of the same size. It has 10.5% fewer parameters than the GRU. `snn_wide` matches the parameter count instead. With 211 neurons it has 554,497 parameters against the GRU's 554,720, a gap of 0.04%. Neither comparison is fair alone, so both are run.

## The validity gate

The parent project lost a 96-model sweep, and then a day of analysis, to a regulariser whose clamp was always active. The term gave no gradient. The model still trained, converged and produced numbers, and no statistic of the term could tell "never engaged" apart from "engaged and converged".

A spiking model can fail in the same way. If the surrogate gradient's support does not reach the neurons' operating margin, it carries nothing, and the loss curve still looks fine. With a fixed threshold of 1.0 at this encoder's scale, the untrained membrane distribution has a standard deviation of about 0.16, and no neuron ever fires. The recurrent path is dead from step one. `spiking.calibrate_threshold` fixes this before the first gradient step. `tests/test_spiking.py` first asserts the silence, so the calibration cannot be dropped without a test failure.

A better statistic of the surrogate cannot settle the question, because no statistic of the term can. `surrogate_diag.py` reports raw margins and firing rates, and every line of its output says that it is not a gate. The gate is `gate_blackout.py`. It compares prior-path prediction against frame persistence and runs between training and analysis. `spike_torus.py` and `spike_family.py` refuse to run on a checkpoint family that did not pass it.

### Gate calibration

On 31 August 2026 the gate was run against the parent's own checkpoints, where the answer is known (`runs/gate_calibration.json`):

| parent checkpoints | expected | got | beating persistence | median ratio |
|---|---|---|---|---|
| conv GRU, decoupled wrapping arm | PASS | PASS | 8/8 | 0.653 |
| conv GRU, bounded arm | PASS | PASS | 8/8 | 0.301 |
| RSSM, decoupled wrapping arm | FAIL | FAIL | 0/8 | 2.122 |
| RSSM, bounded arm | FAIL | FAIL | 0/8 | 1.578 |

All four verdicts match. The evaluation window is drawn independently of the parent's, so per-seed ratios differ by a few per cent (0.5905 against the parent's 0.5873 on `arm_gru_dark_s0`). A gate that could not reproduce a known-good and a known-bad family would be worthless for any new architecture.

## Status

As of 31 August 2026 there are five pre-registered amendments (A1 to A5), 48 tests and 436 generated macros. [`FINDINGS.md`](FINDINGS.md) has the full results and their caveats. [`PREREG.md`](PREREG.md) has the bands they are read against, each fixed before its run.

| | |
|---|---|
| gate calibrated against the parent's known-good and known-bad checkpoints | 4/4 |
| `v6` (wrapping joints), all three models | fail the gate |
| `v7` (bounded joints), `snn` and `rate` | pass, on discovery and on fresh seeds |
| H1, H2 as pre-registered | never ran (they need a passing `v6` cell) |
| H4 (family on the primary view) | failed |
| A4 (view-dependence, confirmatory) | confirmed |
| A5 (untrained control) | learned, confound excluded |

Three corrections are recorded in the files, with no silent edits. The first is an overgeneralisation that the architecture fails the gate, which `v7` refuted. The second is a convergence range read from three seeds when eight existed. The third is the framing of A4 around the A3 floor. A5 showed that the floor does not separate trained from untrained models. The floor checks that a transform beats doing nothing. It is not evidence that the criterion found a learned coordinate.

### The missing decisive test

To learn whether the disagreeing views track the dynamics or something else, the project needs an environment whose correct family is rotation, which means wrapping joints. Every such environment here is `v6`, and no `v6` cell has passed the gate. Until one does, Q2 and Q3 have answers but the mechanism behind them does not.

## Claim boundary

Q2 and Q3 are established only within these limits:

- 16x32 synthetic RGB, two degrees of freedom, and a current-based LIF core with a fast-sigmoid surrogate. Nothing about deeper spiking networks, other neuron models, other surrogates, real video or real robots.
- No efficiency claim of any kind. The project measures no energy, no latency, no spike count per inference and nothing about hardware. Spiking serves only as a second architecture with a multi-component state.
- No claim that a spiking core is more biologically faithful, or that anything here models neurons.
- No claim that the disagreeing views track the dynamics. The dependence of the verdict on the chosen component is established. What each component's verdict tracks is unknown and needs a wrapping-joint environment that passes the gate. None exists yet.
- No claim about `v6`. Nothing there passed the gate, so the criterion has never run on a wrapping arm in this project.
- The parent's limits carry over. The criterion tests for a planar rotation or a translation along a direction, which are two one-parameter group actions among many. No comparison with the probing, disentanglement or activation-patching literature has been run.

Language rule (inherited): write "action-conjugate coordinate". Do not write "the neurons encode joint angle". Say what was measured, and on which view.

Number rule (inherited): `src/make_numbers.py` generates every number in prose from run JSON. The parent project had one hand-typed number, and it was the one that turned out wrong. A number here that is not a generated macro is a defect.

## Reproduce

Python 3 with `torch`, `numpy`, `scipy`, `scikit-learn` and `pytest`. Gate calibration reads the parent checkpoints from the sibling `used_coordinates` folder.

```bash
export PYTHONPATH=src:src/vendor OMP_NUM_THREADS=1
python3 -m pytest -q
python3 src/gate_blackout.py --calibrate --output runs/gate_calibration.json
./run_pilot.sh v6
```

`run_pilot.sh` trains, applies the gate, stops on failure, and only then runs the analysis. The other `run_*.sh` scripts run the amendments (`run_confirm_a4.sh`, `run_untrained_a5.sh`, `run_v7_family.sh` and others).

Long runs detach with `python3 src/vendor/detach.py <log> <cmd>`, which does a real double fork with `os.setsid`. Confirm that the process has ppid 1 and a `??` TTY. Re-parenting to init alone is not enough. That mistake cost a 96-model sweep on 27 August 2026.

## Layout

    src/            spiking core, view-restricted criterion, gate, macros
    src/vendor/     the parent's estimator, unedited and checksummed
    tests/          structural invariants and the gate's refusal
    runs/           result JSON files
    paper/          generated numbers

Model checkpoints (`*.pt`) and logs are not in the repository.
