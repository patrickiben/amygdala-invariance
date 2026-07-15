# amygdala-invariance — a region-agnostic encoder-invariance harness

A runnable scaffold for a family of pre-registered studies: is a *supervised* encoder's
contribution to a brain region's BOLD prediction *identifiable and encoder-invariant*, or is
it **degenerate** with what any competent — or even low-level / random-weight — visual encoder
already supplies?

The same decision engine runs three affective-motivational poles **plus a sensory-relay positive
control**. Each pole **fails or passes the "is there a stimulus-computable target?" precondition for
a *different* reason** — that contrast is the scientific point — and the positive control proves the
harness is alive so the nulls are credible:

| Region | Role | Target encoder | Subregions | Feasibility & why |
|---|---|---|---|---|
| `amygdala` | aversive / salience-appraisal | EmoNet (emotion-supervised) | LB/CM/SF/AStr (Amunts) | **AMBER** — no unified model, but affect is partly stimulus-evoked → target defensible; gated on reliability |
| `nucleus_accumbens` | appetitive / reward-value | ImageReward (value-supervised) | NAc L/R (CIT168) | **RED** on ds002837 (1.5T film) — multiband signal loss & reward under-sampling → `INAPPLICABLE`-likely |
| `nucleus_accumbens_3t` | appetitive / reward-value | ImageReward + food-appeal | NAc L/R (CIT168) | **AMBER — LIVE** on [`ds007267`](https://openneuro.org/datasets/ds007267) (3T food valuation, per-image ratings); gated on a NAc-tSNR audit + a visual-category control |
| `orbitofrontal_vmpfc` | value-integration / common-currency | population appeal proxy | OFC/vmPFC (Harvard-Oxford / HCP-MMP) | **RED** — most mature theory, but subjective value **isn't a stimulus property**; orbital dropout + DMN/narrative confound → `INAPPLICABLE`-or-confounded |
| `lgn` | **positive control** (first-order visual relay) | DINOv2 (any visual encoder) | LGN L/R (Morel/THOMAS) | **GREEN** — carries the retinal image, so a visual encoder *must* predict it; validates the harness detects signal + passes the gate |

> **Inverted semantics for the positive control.** For the value poles, an `H0` / break-case
> verdict means *"encoder artifact"* (bad). For `lgn` it is the **expected, correct** outcome:
> low-level features *should* dominate a low-level relay. The pass-condition for a positive
> control is therefore **gate-passes + strong encoder R²** (`checks.positive_control_ok`), not
> an `H1` verdict. If `lgn` returns `INAPPLICABLE` or near-zero R², the *pipeline* is broken —
> which is exactly what makes the NAc/OFC nulls trustworthy rather than dead-pipeline artifacts.

This generalizes the **representation-invariance test** from *A False Floor* (originally a
check on an information *limit*) to a check on a neural-*coding* attribution. A genuine
coding attribution should survive across paradigmatically different frozen encoders; a fit
that is degenerate across encoders is unidentified.

> **Status: scaffold.** The scientific core (banded ridge, variance partitioning, group-mean
> split-half noise ceiling, reliability gate, and the frozen decision rule + break-case) is
> implemented and unit-tested, and the whole pipeline **runs end-to-end on synthetic data for
> both regions** so you can watch the identical decision logic return the right verdict per
> ground-truth scenario. The real-data adapters (BOLD, film frames, atlas, encoders) are
> **honest stubs** that raise `NotImplementedError` with exact wiring instructions. Nothing
> here has touched real fMRI yet, and no scientific claim is made.
>
> **The NAc region carries a RED feasibility verdict on purpose.** A passive romance film has
> no reward/expectation schedule, and 1.5 T multiband-4 specifically guts NAc signal
> (Srirangarajan 2021: NAcc reward *d* 2.39→0.64 single- vs multi-band). So `INAPPLICABLE` (the
> reliability gate fails) or `H0` (value ≈ low-level) are the first-class, publishable expected
> outcomes — not a positive to be mined. See
> [`../nucleus-accumbens-gap_feasibility.md`](../nucleus-accumbens-gap_feasibility.md).

This is an exploratory research artifact. It is **not** part of the Second Brain and makes no
manuscript-grade claim. The design is fixed by
[`amygdala-rank1_feasibility-and-protocol.md`](../amygdala-rank1_feasibility-and-protocol.md)
and frozen in [`preregistration.yaml`](preregistration.yaml).

## Why a synthetic path exists

The pre-registered decision rule is the falsifiable heart of the study. Before spending a
week wrangling 20 GB of 1.5 T fMRI and a copyrighted film, you want proof the rule actually
distinguishes the three ground-truth worlds it claims to:

| Synthetic scenario | Ground truth | Verdict the rule must return |
|---|---|---|
| `H1` | EmoNet-specific signal that other **high-level** encoders also recover, low-level/random do not | **H1** (identifiable emotion-supervised code) |
| `H0` | signal fully carried by **low-level** structure that every encoder captures equally | **H0** (encoder-non-specific / degenerate) |
| `inapplicable` | no reliable across-subject signal (low tSNR / ISC ≈ 0) | **INAPPLICABLE** (dataset measurement floor, not a verdict on the claim) |

`tests/test_smoke_end_to_end.py` asserts exactly this.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make test        # unit tests + end-to-end synthetic smoke (both regions)
make smoke       # run the 3 synthetic scenarios for EVERY region, print verdicts
python -m amyg_inv.cli smoke nucleus_accumbens   # one region only
make run         # real-data path — raises NotImplementedError until adapters are wired (see data/README.md)
```

## Layout

```
preregistration.yaml        FROZEN, region-agnostic: shared decision/reliability/fitting + a `regions:` map
                            (amygdala, nucleus_accumbens — each with its own subregions/target/encoders)
amyg_inv/
  config.py                 loads + hashes the pre-registration (provenance gate)
  encoders/                 base ABC; lowlevel + random implemented; emonet/dinov2/clip = stubs
  data/                     synthetic (implemented); bold/atlas/stimulus = stubs w/ instructions
  modeling/                 banded_ridge, variance_partition, noise_ceiling, hemodynamics
  reliability/gate.py       tSNR + venous-invariance + ISC reliability gate → per-subregion pass/abstain
  decision/rule.py          the frozen H1 / H0 / ABSTAIN / INAPPLICABLE rule + break-case (intervals, not points)
  pipeline.py               wires it all: features → fit → partition → ceiling → gate → decision
  report.py                 structured ledger (JSON + Markdown); every number is an interval
  cli.py                    `python -m amyg_inv.cli smoke|run`
tests/                      pytest: ridge, partition, ceiling, gate, decision rule, end-to-end smoke
data/README.md              exact download + wiring instructions for the real adapters
```

## The one real-world blocker (per the protocol)

Sourcing the exact *500 Days of Summer* video and validating **frame → TR alignment** against
the NNDb timing is the practical make-or-break step. `amyg_inv/data/stimulus.py` documents the
alignment contract and exposes an alignment-error metric that the reliability audit consumes.
