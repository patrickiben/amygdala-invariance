# Runbook

For the deposited four-arm manuscript analysis, follow **[real_data/REPRODUCE.md](real_data/REPRODUCE.md)**.
Its current targets are the corrected logs under `real_data/scripts/reanalysis/results/`; v1.0 movie
comparisons and convergence values were superseded by v1.1.

For quick local verification without input data:

```bash
python -m pip install -r requirements.txt
make test
make smoke
python validation/numbers_gate.py
python validation/robustness_gate.py
```

`amyg_inv` also contains a region-agnostic synthetic scaffold and exploratory NAc/tSNR utilities.
`python -m amyg_inv.cli run` is an explicit unavailable/stub entrypoint; it is not the path that
produced the manuscript results. See `SCAFFOLD.md` and `data/README.md` for that separate exploratory
work. Synthetic H1/H0/INAPPLICABLE examples are software fixtures, not additional brain-data findings.
