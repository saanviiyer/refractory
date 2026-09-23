# Refractory

## The research question

> **When we say a world model's dynamics "use" a coordinate, is that a fact
> about the model --- or a fact about which part of the model's state we chose to
> look at?**

Started 31 August 2026. Builds on `used_coordinates`, which is left unmodified;
the estimator is vendored verbatim and checksummed in
[`MIGRATION_MANIFEST.md`](MIGRATION_MANIFEST.md).

### This is not a spiking-networks project

The name is misleading and the distinction matters, so it is stated first.
Spiking networks here are the **instrument, not the subject.** No claim in this
repository is about spikes, and none may be: see the claim boundary below, which
forbids every efficiency, energy, spike-count and hardware claim outright.

A spiking core was chosen because it is the simplest architecture that **forces
the analyst to make a choice they cannot avoid noticing.** A GRU has one hidden
vector, so "which coordinates do the dynamics use" is asked of that vector and
there is nothing to decide. A current-based LIF core carries four components of
equal standing --- membrane potential `v`, synaptic current `i`, spike trace `r`,
and the emitted vector `s` --- and a probing paper would treat any of them as
"the representation". The choice was always being made; a one-vector
architecture just hid it.

The project's own controls confirm the framing. The effect appears in the
**rate twin** --- same architecture, same parameters, continuous emission --- and
is uninformative in the spiking cell, both times it was measured. Whatever this
is about, it is not about spikes. It is about any model whose state is not one
vector, which is most of them.

### Why the question is not already answered

`used_coordinates` established that **decodability is not use**: a variable can
be read out of a latent state while the model's own transition dynamics ignore
it. Its criterion asks the "use" question directly, without labels.

That criterion carries an unexamined presupposition: **that there is one latent
state to ask about.** The parent never had to name a state variable, because a
GRU offers only one. The presupposition was invisible rather than justified.

This project makes the choice explicit and asks whether the answer survives it.

### Three sub-questions, and where each stands

**Q1. Does the criterion transfer to a second architecture?**
Unresolved, and the answer is environment-dependent rather than yes or no. The
spiking model fails the validity gate on the wrapping arm (`v6`, 1 of 8 seeds
beat frame persistence) and passes it on the bounded arm (`v7`, 8 of 8, ratio
0.4569, replicated on fresh seeds at 0.4749). But on `v7` the criterion's
primary view returns nothing, so "the criterion transfers" is not something this
project can assert.

**Q2. Is the verdict a property of the model or of the chosen state variable?
--- the central question.**
**Answered: of the chosen state variable.** Pre-registered (A4), fresh seeds,
clean gate, scored by code: on the same frozen model, the `s` view returns
translation 8/8 while the `i` view returns rotation 6/8, for the same actuator.
Two equally-sized components of one state, opposite answers about which group
the action induces.

**Q3. Is that dependence learned, or an artefact of the representation?**
**Answered: learned.** The untrained control (A5) shows that at random
initialisation every view that fits at all agrees --- all say translation --- and
that the `s` view fits nothing before training (residual 0.9047, against 0.2821
after). Training *creates* the disagreement. The bounded-code alternative, which
was the leading rival explanation, is excluded.

### What follows, stated carefully

"The dynamics use this coordinate" is **under-specified** unless the state
component is named. It is not a property a model simply has. This does not make
the parent's criterion wrong --- on a one-vector architecture the choice is
forced and the question is well-posed. It means the criterion's scope was
narrower than it looked, and that any application of it to a model with
structured state must report which component it was pointed at.

The obvious component is the wrong one. The `r` view --- the state the decoder
reads, the closest analogue of the GRU hidden vector, and the choice any
reasonable analyst would make --- **finds nothing anywhere**: winning residuals
0.7495 to 0.9514 across every cell measured, against a no-transform reference of
1.0.

## The controls that decide what any of it means

**The rate twin.** `RateArmWM` is the same model with the Heaviside replaced by
the surrogate's own forward function. Identical parameter count, identical
initialisation, identical gradient form, identical four-component state. Any
effect that appears equally in the twin is not about spiking; it is about
having a multi-component state. `PREREG.md` fixes in advance how large the
spiking-minus-twin difference must be before anything is called a spiking
result.

**Two budgets.** `snn` matches the GRU's hidden width (128), so every view has
the dimension the criterion was calibrated on and the variance-matched
random-plane null is drawn from a space of the same size; it comes in 10.5%
under the GRU's parameter count. `snn_wide` matches the parameter count
instead: 211 neurons is 554,497 parameters against the GRU's 554,720, a gap of
0.04%. Neither is a fair comparison alone, so both are run.

## The gate, and why it is not optional

The parent lost a 96-model sweep and then a day of belief to a regulariser
whose clamp always bound: the term contributed no gradient, the model trained,
converged and produced numbers, and no statistic of the term could tell "never
engaged" from "engaged and converged".

A spiking model has the same failure in a new costume. A surrogate gradient
whose support does not reach the neurons' operating margin carries nothing, and
the loss curve looks fine. Left at a fixed threshold of 1.0 against this
encoder's scale, the untrained membrane distribution has standard deviation
about 0.16 and **not one neuron ever fires** --- the recurrent path is dead from
step one. `spiking.calibrate_threshold` fixes that before the first gradient
step, and `tests/test_spiking.py` asserts the silence first so the calibration
cannot quietly be dropped.

But the lesson is not "add a better statistic of the surrogate". It is that no
statistic of the term can settle it. `surrogate_diag.py` reports raw margins
and firing rates and says on every line that it is not a gate. The gate is
`gate_blackout.py`: prior-path prediction against frame persistence, wired
between training and analysis, and `spike_torus.py` and `spike_family.py`
**refuse to run** on a checkpoint family that did not clear it.

### The gate is calibrated

Run 31 August 2026 against the parent's own checkpoints, where the answer is
already known (`runs/gate_calibration.json`):

| parent checkpoints | expected | got | beating persistence | median ratio |
|---|---|---|---|---|
| conv GRU, decoupled wrapping arm | PASS | PASS | 8/8 | 0.653 |
| conv GRU, bounded arm | PASS | PASS | 8/8 | 0.301 |
| RSSM, decoupled wrapping arm | FAIL | FAIL | 0/8 | 2.122 |
| RSSM, bounded arm | FAIL | FAIL | 0/8 | 1.578 |

Four of four. The evaluation window is drawn independently of the parent's, so
per-seed ratios differ by a few per cent (0.5905 against the parent's 0.5873 on
`arm_gru_dark_s0`) while every verdict matches. A gate that could not
reproduce a known-good and a known-bad family would be worthless in both
directions, whatever it said about a new architecture.

## Status

31 August 2026. Five pre-registered amendments (A1-A5), 48 tests, 436 generated
macros. Full results and their caveats in [`FINDINGS.md`](FINDINGS.md); the
bands they are read against, fixed before each run, in [`PREREG.md`](PREREG.md).

| | |
|---|---|
| gate calibrated against the parent's known-good and known-bad checkpoints | 4/4 |
| `v6` (wrapping joints), all three models | fail the gate |
| `v7` (bounded joints), `snn` and `rate` | pass, on discovery **and** fresh seeds |
| H1, H2 as pre-registered | never ran --- they need a passing `v6` cell |
| H4 (family on the primary view) | **failed** |
| A4 (view-dependence, confirmatory) | **confirmed** |
| A5 (untrained control) | **learned**, confound excluded |

Three corrections are recorded rather than quietly edited: an
overgeneralisation about the architecture failing the gate, refuted by `v7`; a
convergence range read off three seeds when eight existed; and the framing of
A4 around the A3 floor, which A5 showed does not separate trained from
untrained models. The floor is a check that a transform beats doing nothing ---
not evidence that the criterion found a learned coordinate.

### The decisive test that is still missing

Whether the disagreeing views track the *dynamics* or something else needs an
environment whose correct family is **rotation** --- wrapping joints. Every such
environment here is `v6`, and no `v6` cell has ever cleared the gate. Until one
does, Q2 and Q3 are answered and the mechanism behind them is not.

## Claim boundary

Q2 and Q3 are established, within these edges, and must not be written past
them:

- 16x32 synthetic RGB, two degrees of freedom, a current-based LIF core with a
  fast-sigmoid surrogate. Nothing about deeper spiking networks, other neuron
  models, other surrogates, real video or real robots.
- **No efficiency claim of any kind.** This project measures no energy, no
  latency, no spike count per inference and no hardware anything. Spiking is
  used here as a second architecture and a multi-component state, and that is
  the only thing it may be said to be.
- No claim that a spiking core is more biologically faithful, or that anything
  here is a model of neurons.
- **No claim that the disagreeing views track the dynamics.** That the verdict
  depends on the chosen component is established; *what* each component's
  verdict is tracking is not, and needs a wrapping-joint environment that
  clears the gate. None exists.
- No claim about `v6`. Nothing there cleared the gate, so the criterion has
  never been run on a wrapping arm in this project at all.
- The parent's boundaries carry over: the criterion tests for a planar rotation
  or a translation along a direction, which are two one-parameter group actions
  among many; and no comparison against the probing, disentanglement or
  activation-patching literatures has been run.

Language rule, inherited: write "action-conjugate coordinate", never "the
neurons encode joint angle". Say what was measured, on which view.

Number rule, inherited: every number in prose is generated from run JSON by
`src/make_numbers.py`. The parent shipped exactly one typed number and it was
the one that turned out to be wrong. An un-macroed number here is a defect.

## Layout

    src/            spiking core, view-restricted criterion, gate, macros
    src/vendor/     the parent's estimator, verbatim and checksummed
    tests/          structural invariants and the gate's refusal
    runs/           checkpoints and result JSON
    paper/          generated numbers, and a manuscript once there is one

## Reproduce

```bash
cd refractory && export PYTHONPATH=src:src/vendor OMP_NUM_THREADS=1
python3 -m pytest -q
python3 src/gate_blackout.py --calibrate --output runs/gate_calibration.json
./run_pilot.sh v6
```

`run_pilot.sh` trains, gates, refuses on failure, and only then analyses. Long
runs detach with `python3 src/vendor/detach.py <log> <cmd>`, which does the real
double fork with `os.setsid`. Confirm ppid 1 **and** a `??` TTY. Re-parenting to
init alone is not enough; that mistake cost this line of work a 96-model sweep
on 27 August 2026.
