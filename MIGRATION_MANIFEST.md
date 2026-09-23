# Migration manifest

`refractory` was created on 31 August 2026 from `used_coordinates`. **No file in
the source repository was moved, modified or deleted.** Verify with a diff
against `../used_coordinates/src`; the table below records the checksum of every
vendored file at the moment it was copied.

## Why vendor rather than import

The claim this project can make depends entirely on the estimator being the
same one. "A second architecture clears the criterion" means nothing if the
criterion has drifted, and a criterion reached over `PYTHONPATH` into a sibling
directory drifts silently the next time that sibling is edited. Copying pins
it, and the checksums make a later divergence visible instead of invisible.

The cost is that a genuine improvement to the parent's estimator has to be
pulled across deliberately. That is the intended trade: pulling it across is a
decision with a date on it, and inheriting it is not.

## What is vendored, verbatim

Everything below is byte-identical to `../used_coordinates/src` as of
31 August 2026. Regenerate this table with:

```bash
python3 -c "import hashlib,pathlib;[print(f.name, hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(pathlib.Path('src/vendor').glob('*.py'))]"
```

| file | lines | sha256 (first 16) | identical to parent |
|---|---:|---|---|
| `analyze_arm_zeroshot.py` | 192 | `aab3d3b0f0ba35f8` | yes |
| `analyze_visual_zeroshot.py` | 138 | `43b2c21c8a829cf7` | yes |
| `arm_env.py` | 100 | `b87eeff977506449` | yes |
| `arm_env_v2.py` | 76 | `1cd797bcaee5848a` | yes |
| `arm_env_v4.py` | 85 | `1024ac197c937a32` | yes |
| `arm_env_v5.py` | 83 | `3d27960abfd5e669` | yes |
| `arm_env_v7.py` | 53 | `ca66ec327680e9ce` | yes |
| `arm_family.py` | 143 | `91de7af35f5bb9ed` | yes |
| `arm_fast.py` | 221 | `5c4f0cb9c40c771e` | yes |
| `arm_models.py` | 81 | `db5dc77e455fbda6` | yes |
| `arm_torus.py` | 342 | `6b61f758b8b2d2f4` | yes |
| `blackout_validity.py` | 170 | `6c2b94cc0327467f` | yes |
| `detach.py` | 37 | `77f5b3c909c45cc3` | yes |
| `envs.py` | 180 | `91289b97052e8f9a` | yes |
| `interventions.py` | 185 | `c8294f9998ee54c8` | yes |
| `models.py` | 93 | `1d465e58b595a9ee` | yes |
| `rssm.py` | 152 | `135f3a7cd8585333` | yes |
| `train.py` | 139 | `b8bdcb995d8dc77f` | yes |
| `train_arm_visual.py` | 141 | `8ba745d52cc8ea1d` | yes |
| `train_rssm_dmc.py` | 149 | `8f77b8e076bd61b5` | yes |
| `train_visual.py` | 86 | `d15ddccc642aab1f` | yes |
| `visual_env.py` | 67 | `639749f5499e0e00` | yes |
| `visual_models.py` | 43 | `44cd09c99abd01a5` | yes |
| `zeroshot.py` | 263 | `c633f6a42701a867` | yes |
Twenty-four files, 3,219 lines, all identical to the parent.

`rssm.py` and `train_rssm_dmc.py` are vendored although this project trains no
RSSM: `blackout_validity.py` imports them, and the gate has to be the parent's
gate for its calibration against the parent's RSSM checkpoints to mean anything.

## What is new, and what it depends on

| file | lines | depends on the parent for |
|---|---:|---|
| `src/spiking.py` | 278 | the encoder/decoder shapes in `visual_models.py`, `visual_env.IMAGE_H/W` |
| `src/spike_fast.py` | 211 | subclasses `arm_fast.CachedRollout`; `_advance` is the only override |
| `src/spike_torus.py` | 318 | the grids, deflation, search, nulls, basis scoring and uniformity check from `arm_torus.py` and `interventions.py` |
| `src/spike_family.py` | 146 | `arm_family.TRANSLATION_GRID` and `arm_family._refine` |
| `src/gate_blackout.py` | 148 | `blackout_validity.check` and `_last_visible`, called rather than reimplemented |
| `src/train_spiking.py` | 122 | `train_arm_visual.make_batch`, `prediction_loss`, and the loss weighting per variant |
| `src/surrogate_diag.py` | 122 | `train_arm_visual.make_batch` |
| `src/make_numbers.py` | 181 | nothing; reads this project's run JSON |

The estimator's numerical path is the parent's. `spike_fast.conjugacy_losses`
and `translation_losses` reproduce the parent's structure with the transform
restricted to a named block of the state, and
`tests/test_spike_fast.py::test_view_all_matches_the_parents_whole_state_rotation`
asserts that the `view="all"` case agrees with `arm_fast.rotate_states` to
1e-12 on a model whose state is a single block. Without that test the
comparison between this project's numbers and the parent's would be between two
different estimators.

## Checkpoints read from the parent, never written to

`src/gate_blackout.py --calibrate` reads, read-only:

    ../used_coordinates/runs/arm_v6/arm_gru_dark_s*.pt      8 checkpoints
    ../used_coordinates/runs/arm_v7/arm_gru_dark_s*.pt      8
    ../used_coordinates/runs/rssm_v6/arm_rssm_dark_s*.pt    8
    ../used_coordinates/runs/rssm_v7/arm_rssm_dark_s*.pt    8

The evaluation window is drawn from this project's own seed and sequence
length, so per-seed ratios differ from the parent's stored report by a few per
cent while every verdict matches: `arm_gru_dark_s0` reads 0.5905 here against
0.5873 there, on 1,322 scored blackout steps against the parent's 756. The
calibration checks verdicts, which are what the gate is for; it does not claim
bit-identical reproduction of a run made with a different evaluation window.

## One vendored file has a path assumption that does not survive the move

`detach.py` ends with

    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

which resolves to the project root when the file sits at `src/detach.py`, as it
does in the parent. Here it sits at `src/vendor/detach.py`, so the chdir lands
in `src/` instead. A relative command or log path passed to it is then resolved
from the wrong directory and the exec fails with `FileNotFoundError`.

It is not patched, because patching it would forfeit the byte-identical
guarantee for the sake of one caller. **Pass absolute paths:**

```bash
R=$(pwd); python3 src/vendor/detach.py "$R/logs/pilot_v6.log" "$R/run_pilot.sh" v6
```

Caught on 31 August 2026 on the first launch, which failed loudly in the log
rather than silently. Then confirm ppid 1 **and** a `??` TTY before walking
away; re-parenting to init alone is not enough.
