# Corrected real-data analysis deposit

The current evidence is under `scripts/reanalysis/results/`. The original July 8 logs in `results/`
are historical and superseded where a corrected log exists. Use [REPRODUCE.md](REPRODUCE.md) for
setup, input paths, run order and explicit remaining reconstruction gaps.

| Current archived result | Evidence |
|---|---|
| Movie EmoNet-unique over ImageNet L +0.0022 / R +0.0017 | `scripts/reanalysis/results/pls20_result.txt` |
| Movie EmoNet−ResNet L +0.0009 (p=.002) / R +0.0006 (p=.003); fusiform +0.0105 | `scripts/reanalysis/results/second_affect_swap_result.txt` |
| Two-affect-encoder convergence L −0.28 / R −0.00 (CIs include 0); fusiform +0.79 | `scripts/reanalysis/results/noise_ceiling_result.txt` |
| Movie increment survives DINOv2 L +0.0021 / R +0.0015 | `scripts/reanalysis/results/movie_strongbaseline_result.txt` |
| Single-neuron affect−object pooled −0.086, left −0.109, MFC +0.100 | `scripts/reanalysis/results/amyg_neuron_rsa_result.txt` |
| Non-superiority margin +0.100; unit-bootstrap upper −0.0065, session upper +0.0063 | `scripts/reanalysis/results/single_neuron_tost_result.txt` |

The movie result is a small but significant single-encoder increment. The construct-invariance
argument rests on its failure to generalize across independent affect encoders; the data do not
establish reliably negative amygdala encoder agreement. See the manuscript and changelog for scope.

No film, brain data, derived feature arrays or weights are redistributed. The film must be supplied
lawfully; public brain data and weights remain subject to their original terms. The reconstructed
movie extractor is not yet validated against a full-data rerun. Static log checks verify the archived
numbers exist, not that this reconstruction regenerates them.
