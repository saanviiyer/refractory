# Pre-registration

Fixed 31 August 2026, before any model in this repository was trained. The only
run that existed at the time of writing is the gate calibration, which is a
check on the instrument and not a hypothesis test.

Anything below that turns out to need changing gets changed **in a dated
amendment at the bottom of this file**, not by editing the band.

## Reference values

The bands are set against the parent's published conv-GRU results on the
decoupled wrapping arm (`v6`), dark condition, `mode=search`, eight seeds,
read from `used_coordinates/runs/arm_torus_search_v6.json`:

| quantity | plane 1 | plane 2 |
|---|---|---|
| conjugacy residual, median (range) | 0.205 (0.082-0.326) | 0.232 (0.166-0.583) |
| matched-null residual, median | 0.993 | 0.983 |
| best basis | (1,0) 8/8 | (0,1) 8/8 |
| concentration against that basis, median (range) | 0.949 (0.919-0.988) | 0.958 (0.881-0.978) |
| principal angle between the two planes | 90.0 deg, all seeds | |

And on the bounded arm (`v7`), from `INSTRUMENT.md`: rotation 8/8 on the
wrapping parity arm, translation 8/8 on the bounded one.

## G0: instrument calibration. RUN, PASSED.

The gate must reproduce the parent's known verdicts before its verdict on a new
architecture means anything: conv GRU PASS on `v6` and `v7`, RSSM FAIL on both.

Result, `runs/gate_calibration.json`, 31 August 2026: 4/4 agreeing, `calibrated`
true. Median ratios 0.653 / 0.301 (GRU, pass) and 2.122 / 1.578 (RSSM, fail).

## G1: validity gate. Runs before any analysis, on every cell.

For each cell (model in {`snn`, `rate`, `snn_wide`, `rate_wide`}, variant in
{`v6`, `v7`}), `gate_blackout.py` compares blackout prediction against frame
persistence on eight dark-condition seeds.

- **PASS** (8/8 beat persistence): the analysis for that cell runs.
- **PARTIAL** (1-7/8): the analysis runs only with `--allow-partial`, which is
  recorded in the output JSON, and the dropped seeds and the reason are written
  into `FINDINGS.md`. This is a declared deviation, not a silent one.
- **FAIL** (0/8): nothing downstream runs, and the failure **is** that cell's
  reported result. The parent's RSSM is the precedent: a negative gate is
  publishable and a coordinate verdict read past it is not.

`spike_torus.py` and `spike_family.py` call `require_pass` before they read a
checkpoint and raise rather than warn. `tests/test_gate.py` asserts that
ordering by inspecting their source.

## H1: does the criterion transfer to a spiking core?

Cell: `snn`, `v6`, dark, `mode=search`, primary view `r` (the state the decoder
reads, and so the analogue of the GRU hidden vector), eight seeds.

**H1 holds** if all four of:

1. plane 1 best basis is (1,0) and plane 2 best basis is (0,1) on >= 7/8 seeds;
2. median conjugacy residual < 0.60 on both planes;
3. `beats_null` on >= 7/8 seeds for both planes;
4. median best concentration > 0.70 on both planes.

The bands sit well above the parent's GRU values (0.205/0.232 residual, 0.95
concentration) because a different architecture is allowed to be worse, and well
below the null (0.99, and chance concentration around 0.26 as measured on an
untrained checkpoint) because it is not allowed to be nothing.

**H1 fails** if median residual >= 0.85 on either plane, or if the bases are not
recovered on >= 5/8 seeds. Then the reported result is that the criterion does
not transfer to a spiking core at this budget, written as a result and not as a
debugging note. The parent's limitation stands and this project says so.

Between the two: report the numbers, claim neither. Declared here so it cannot
be resolved after the fact.

## H2: is the verdict a property of the model or of the chosen state variable?

This is the project's own claim. Same cell as H1, all five views
(`v`, `i`, `r`, `s`, `all`).

Primary statistic, per view: `agree(view)` = the number of (seed, plane) pairs
out of 16 (8 seeds x 2 planes) on which that view's `best_basis` **and** its
`beats_null` both match the primary view `r`. Computed by
`make_numbers.agreement_macros`.

- **H2a, invariance:** every view scores `agree >= 14/16`. The criterion reads a
  property of the model. Report as robustness; the architecture axis is
  unblocked with no caveat about state choice.
- **H2b, dependence:** at least one view scores `agree <= 8/16` while itself
  beating its own matched null on >= 6/8 seeds for at least one plane. The
  verdict depends on which state variable the criterion is pointed at, and a
  claim of the form "the dynamics use this coordinate" needs the view named.
  The second qualifier matters: a view that finds nothing at all disagrees
  trivially and is not evidence of anything.
- **9/16 to 13/16 inclusive is declared uninformative.** Report the number,
  claim neither H2a nor H2b, and say in the write-up that the arm was
  inconclusive.

## H3: is any of it about spiking?

The rate twin shares the four-component state, the parameter count, the
initialisation and the gradient form, and differs only in that its emission is
continuous.

A spiking-specific claim requires, for the view in question,
`agree_snn(view) - agree_rate(view)` to differ by more than 4/16 in the
direction claimed. Below that, the finding is reported as a property of
multi-component recurrent state, which is the honest description and still a
finding. It would mean the criterion's under-specification has nothing to do
with spikes and applies to any model whose state is not one vector.

## H4: the family detector on bounded joints

Cell: `snn`, `v7`, dark, primary view `r`, eight seeds. The parent's GRU says
translation 8/8 on both joints.

**H4 holds** if the spiking model also returns translation on >= 7/8 seeds for
both joints on the primary view. The same `agree` statistic is computed across
views, with the family verdict in place of the basis, and read against the same
H2a/H2b/uninformative bands.

## Budget control

`snn_wide` and `rate_wide` (211 neurons, 554,497 parameters against the GRU's
554,720) repeat H1 and H2 on `v6`. If the width-matched and parameter-matched
models disagree on H1, neither is reported as the architecture result and the
disagreement is the finding.

## Controls, all declared in advance

- **Untrained.** `--untrained` rebuilds the architecture at random init. No
  plane may beat its matched null. A 6-step checkpoint already reads residual
  1.00 against null 1.00 with concentration 0.26, which is the expected shape.
- **Shuffled actions.** Trained with actions shuffled in time. This is the
  parent's dissociation case, where heading decoded at 20-24 degrees while the
  conjugate plane sat 69-83 degrees away.
- **Lit.** No blackout.
- **Matched-variance random planes**, `--n-null 4`, drawn from the orthogonal
  complement of the discovered plane and matched on both dimension and variance
  removed.
- **Intervention mode.** `view-only` is primary: it is the literal analogue of
  what the parent does, since a GRU has one block and rotating it is view-only
  by construction. `--coherent`, which re-derives the spike vector from the
  rotated potential, is a declared secondary and is run on the `v` and `all`
  views, where it is the only place re-derivation is defined. The output
  records `coherent_effective` so a no-op cannot be mistaken for a null result.

## What would invalidate the measurement

Listed before any model was trained, in the parent's format.

1. **The model does not predict the arm during a blackout.** There is then no
   latent worth probing. Gate G1.
2. **The population is degenerate.** A checkpoint with
   `frac_neurons_never_fire > 0.5` or `frac_neurons_always_fire > 0.5`
   (`surrogate_diag.py`) carries a state that cannot turn, and is excluded and
   reported as excluded. This is a declared exclusion rule, not a gate: it is
   applied to individual checkpoints, and if it removes more than two of eight
   seeds in a cell the cell is reported as degenerate rather than analysed.
3. **Discovery touches labels.** Structurally prevented and asserted:
   `discovery_uses_joint_labels` is recorded false in every output.
4. **Fitted gains rail at the grid edge.** `gain_at_grid_edge` is recorded per
   fit. If more than 25% of fits in a cell rail, the grid was too narrow, the
   cell is re-run on a wider one, and both runs are kept.
5. **The two budgets disagree**, per the budget control above.

## Amendments

### A1: a graded control. 31 August 2026, written before the run.

**What prompted it.** H3's rate twin was built to isolate discreteness, and the
v6 pilot showed it does not do that cleanly. Only 23.8% of its emitted values
fall strictly between 0.02 and 0.98 (median across eight seeds; best seed
40.3%), because `sigmoid(4 * margin)` saturates at the margins the model learns.
Roughly three quarters of the "continuous" control is a spike in all but name,
so the pilot's "both fail, therefore not discreteness" reading is supported but
not established.

**What is added.** A third model kind, `rate_soft`: the same architecture,
parameter count and initialisation, with the emission `sigmoid(w_e * margin)`
at `w_e = 0.25` and **exact autograd through that sigmoid**, with no surrogate.

**What it therefore tests, stated plainly.** `rate_soft` removes three things at
once relative to `snn`: the discrete forward, the surrogate/forward mismatch,
and the weak emission gradient. It is *not* a clean isolation of discreteness
alone; it is a clean separation of **spiking machinery** from **architecture**.
That is the more useful question now, because the pilot's open hypothesis is
architectural: both existing models reset the membrane multiplicatively by the
emitted value every step and route the whole recurrence through that bounded
emission.

- If `rate_soft` **fails** the gate, the failure is the four-component LIF
  architecture, and nothing about spikes, surrogates or gradients explains it.
- If `rate_soft` **passes**, the spiking machinery is implicated, and which of
  the three removed components matters is a further experiment.

**Validity condition, fixed now.** `rate_soft` counts as a graded control only
if its median `frac_emission_graded` is **>= 0.70**. Below that it has
saturated the way the first twin did, its result is reported as another
partially-saturated model, and it does not license the separation above.

**What it does not do.** It does not replace the original twin. `gateRateV6*`
and the H3 section of `FINDINGS.md` stand as recorded whatever
`rate_soft` returns. No parameter of `snn` or `rate` is changed.

### A2: `v7` proceeds despite the `v6` gate failure. 31 August 2026.

`PLAN.md` already sequences S5 as not blocked on S4, so this records rather than
changes the decision. The `v6` cells failed the gate, and `v7` is run anyway
because a failure confined to one environment and a failure across both are
different results, and the second is only available by running it. `v7` is
judged against the same G1 bands, with H4 downstream of its own gate.

### A3: H4 needs a quality floor. 31 August 2026, mid-run, partial peek disclosed.

**Disclosure first.** This is written after seeing 2 of 8 checkpoints per cell
in the running H4 analysis (a structural check that the fits were well-formed) and before any cell completed. What those two showed: winning residuals in
the range 0.42 to 0.99, and several family verdicts decided by margins of about
0.05 between two families that both fit badly. No per-seed verdict is used
below and the amendment is anchored to a band this file already contains.

**The gap.** H4 as written holds if the primary view returns translation on
>= 7/8 seeds. It says nothing about whether the winning family *fits*. A
residual near 1.0 means the fitted transform is no better than applying no
transform, so a model could satisfy H4 by coin-flipping between two families
that both explain nothing. H2b already guards against this ("a view that finds
nothing at all disagrees trivially and is not evidence of anything") and H4
was written without the equivalent.

**The floor, anchored not invented.** H4 additionally requires the winning
family's **median residual to be below 0.60**, which is the band H1 already
fixes for the conjugacy residual, and which sits well above the parent's own
worst winning residual on this instrument (0.360, rotation on the bounded arm,
`INSTRUMENT.md`) and well below the no-transform reference of 1.0.

- Both conditions met: H4 holds.
- Family recovered on >= 7/8 seeds but median winning residual >= 0.60:
  **unresolved**, not a pass. Report the verdict counts and the residuals, and
  state that the instrument found no family that fits.
- Family not recovered: H4 fails, as already written.

**Note added 31 August 2026, after A5.** The floor does **not** separate a
trained model from an untrained one. In the A5 control, 8 of 10 (view, joint)
fits on a randomly initialised `rate` model clear it, one of them at 0.2946,
a better fit than the same view achieves after training. The floor is a check
that a fitted transform beats doing nothing; it is **not** evidence that the
criterion found a learned coordinate, and must not be quoted as if it were.
Anything already written in that form is a framing error and is corrected in
`FINDINGS.md`.

**Consequence for H2-style view comparisons on this data.** The `agree`
statistic counts a view as agreeing when its verdict matches the primary
view's. Where no view clears the floor, agreement between views is agreement
between coin flips and licenses nothing in either direction. Any view
comparison on a cell that fails the floor is reported as uninformative
regardless of where the count lands.

### A4: confirmatory test of view-dependence. 31 August 2026, before any fresh model was trained.

**What is being confirmed.** H4's post-hoc observation: on `v7`, two
128-dimensional components of the same frozen state, each clearing the A3
residual floor, returned opposite family verdicts. Specifically, on **joint 1**
the `s` view returned translation 8/8 (residual 0.2672, margin 0.2277) while the
`i` view returned rotation 7/8 (residual 0.4977, margin 0.1963). That comparison
was reconstructed after the fact, because the pre-registered `agree` statistic
references the `r` view and `r` fits nothing.

**Design.** Eight **fresh seeds, 8 to 15**, of `snn` and `rate` on `v7`.
Environment, budget, loss, optimiser, schedule, architecture, estimator, grids
and the A3 floor are all unchanged; only the seeds are new. Checkpoints are
written to `runs/conf_v7/` and are not mixed with the discovery run. Re-reading
the discovery checkpoints would confirm nothing, so they are not touched.

**Primary prediction, sharp and directional.** In any confirmatory cell where
`s` and `i` both clear the A3 floor (median winning residual < 0.60):

- `s` returns **translation on >= 6/8 seeds for joint 1**, and
- `i` returns **rotation on >= 6/8 seeds for joint 1**.

Both hold -> **view-dependence is confirmed**: the same criterion, on the same
frozen model, gives opposite answers about which group the actuator induces
depending on which equally-sized component of the state it is pointed at.

Both views clear the floor but do not show opposite majorities on joint 1 ->
**refuted**. The discovery observation was a seed artefact and gets written up
as one.

Fewer than both views clear the floor -> **uninformative**. Not a failure and
not a pass; the instrument found too little on this cell to compare, and the
question stays open.

Joint 1 and this view pair are named because that is where the discovery
observation sat. A different pair disagreeing on a different joint would be a
new observation, not a confirmation, and is recorded as such if it occurs.

**Secondary predictions**, reported whatever the primary does:

- **P1:** `s` returns translation on >= 7/8 seeds on **both** joints, in both
  cells. The discovery run gave 32/32 across cells.
- **P2:** the primary view `r` does **not** clear the A3 floor, in either cell.
  The claim that the state the decoder reads is the wrong place to point the
  criterion depends on this, and it is the prediction most likely to fail.

**Gate discipline is unchanged.** Each confirmatory cell is gated on its own
`gate_blackout` verdict. FAIL means that cell is not analysed. PARTIAL is
analysed only under `--allow-partial`, recorded in the JSON. The discovery run
had `snn` PASS 8/8 and `rate` PARTIAL 7/8; there is no expectation that the
fresh seeds reproduce those verdicts, and the verdicts are reported as they come.

**What this test cannot do.** It cannot say the effect is about spiking. The
discovery comparison was in the rate twin, and H3 already fixes that reading. It
cannot separate a dynamical explanation from a representational one: `s` is
bounded to [0,1] and non-negative, and a translation family may fit such a code
better for reasons of what it can represent. That confound is untouched by this
test and stays in the write-up.

### A5: the untrained control, to test the bounded-code confound. 31 August 2026, before it was run.

**What it tests.** A4 confirmed that the `s` and `i` views return opposite
family verdicts, and named the alternative explanation it could not exclude:
`s` is bounded to [0, 1] and non-negative, so a translation family may fit it
better because of what that code can represent rather than because of what the
dynamics do. If that is the whole story, the verdict is a fact about the
representation's geometry and is already present **before any learning**.

**Design.** The architecture rebuilt at random initialisation, no gradient step
taken, and the family detector run on it in the same way as on the trained models: same
environment (`v7`), same seeds (8-15), same estimator, same grids, same A3
floor.

**One necessary departure, stated because it decides the control's validity.**
The random-init model is given the **same pre-gradient threshold calibration the
trainer applies**, because that calibration happens before the first gradient
step and is part of initialisation rather than of learning. Without it the
spiking population never fires at all (the default threshold of 1.0 sits far
outside an untrained membrane distribution of standard deviation about 0.16),
and the control would return "nothing fits" by construction, which would look
like the reassuring answer while being an artefact. A control that cannot fail
is not a control.

**The gate is bypassed, deliberately and on the record.** An untrained model
will not beat frame persistence, and refusing to analyse it would defeat the
purpose: the point is precisely what the criterion reports on a model that has
learned nothing. The output records `gate: bypassed (untrained control)` rather
than a verdict.

**Prediction, on the `s` view, joint 1, eight seeds:**

- **Geometric:** translation on >= 6/8 seeds **and** median winning residual
  below the A3 floor of 0.60. Then the `s` view's verdict is substantially a
  property of a bounded non-negative code, P1's 64 of 64 is largely an
  artefact, and A4's confirmation stands as a statement about the criterion
  being under-specified but **not** as one about dynamics. This is the outcome
  that damages the headline, and it is the one being tested for.
- **Learned:** the untrained `s` view either fails the floor, or shows no
  majority at >= 6/8. Then geometry alone does not produce the trained verdict
  and the dynamical reading survives this test. It survives but is not proven.
- Anything else: report the numbers, claim neither.

**Also recorded, not predicted:** what the `i` and `r` views do at random
initialisation, and the untrained firing rates and graded fractions, so a
degenerate population is visible as such rather than being read as a result.

**What it cannot do.** Passing this control does not establish a dynamical
account. It removes one alternative explanation. The decisive test remains an
environment whose correct family is rotation, which requires a `v6` cell to
clear the gate, and none has.
