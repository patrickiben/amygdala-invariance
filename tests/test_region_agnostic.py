"""The same decision engine must work for BOTH regions defined in the pre-registration.

Proves the harness is region-agnostic: amygdala (4 subregions, EmoNet target) and
nucleus_accumbens (2 subregions, ImageReward target) each run the three synthetic
ground-truth worlds through the identical pipeline and return the correct verdict.
"""
import numpy as np
import pytest

from amyg_inv.config import list_regions, load_prereg
from amyg_inv.data import synthetic
from amyg_inv.pipeline import Overrides, run

REGIONS = list_regions()
SCENARIOS = [("H1", "H1"), ("H0", "H0"), ("inapplicable", "INAPPLICABLE")]


def test_all_three_poles_present():
    for pole in ("amygdala", "nucleus_accumbens", "orbitofrontal_vmpfc"):
        assert pole in REGIONS


def test_ofc_config_shape():
    ofc = load_prereg("orbitofrontal_vmpfc")
    assert ofc.target == "appeal_value"
    assert ofc.d("decision", "subregion_majority") == 2   # 2 of 3
    assert ofc.d("nuisance_required") == ["dmn_narrative"]  # mandatory DMN/narrative regression
    assert ofc.d("feasibility") == "RED"


@pytest.mark.parametrize("region", REGIONS)
@pytest.mark.parametrize("scenario,expected", SCENARIOS)
def test_region_scenario_verdict(region, scenario, expected):
    cfg = load_prereg(region)
    rng = np.random.default_rng(cfg.seed)
    feat, bold, tsnr = synthetic.make(scenario, cfg, rng, sizes=synthetic.FAST)
    res = run(feat, bold, tsnr, cfg, rng, overrides=Overrides(n_boot=60, n_folds=5))
    assert res.verdict.branch == expected, f"{region}/{scenario}: {res.verdict.reason}"


def test_nac_config_shape():
    nac = load_prereg("nucleus_accumbens")
    assert nac.target == "imagereward"
    assert nac.subregions == ["NAc_L", "NAc_R"]
    assert nac.d("decision", "subregion_majority") == 2   # 2 of 2 hemispheres
    # backbone-matched control is present as a named (generic) encoder
    assert "clip_headoff" in nac.all_encoders


def test_nac_3t_variant_points_at_ds007267():
    nac3t = load_prereg("nucleus_accumbens_3t")
    assert nac3t.d("dataset", "id") == "openneuro/ds007267"
    assert nac3t.d("dataset", "value_anchor") == "rating_value"   # subject's own per-trial value
    assert nac3t.d("mask", "qc_gate") == "nac_tsnr_audit"         # mandatory pre-flight (fail -> INAPPLICABLE)
    assert nac3t.d("feasibility") == "AMBER"                       # a live test, not a foregone positive
