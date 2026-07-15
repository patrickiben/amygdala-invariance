"""Load and hash the frozen, region-agnostic pre-registration.

The SHA-256 of preregistration.yaml (the WHOLE file, all regions) is recorded on load and
stamped into every results ledger, so any post-hoc edit to a registered parameter is
detectable. `load_prereg(region)` merges the shared decision/reliability/ceiling/fitting
blocks with the named region block (region keys win; decision_overrides fold into decision).
"""
from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PREREG = os.path.join(_REPO_ROOT, "preregistration.yaml")
DEFAULT_REGION = "amygdala"


@dataclass(frozen=True)
class Prereg:
    raw: Dict[str, Any]      # merged (shared + region) config
    sha256: str              # hash of the whole file (covers all regions)
    path: str
    region: str

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def subregions(self) -> List[str]:
        return list(self.raw["subregions"])

    @property
    def all_encoders(self) -> List[str]:
        return list(self.raw["all_encoders"])

    @property
    def target(self) -> str:
        return self.raw["target_encoder"]

    @property
    def lowlevel(self) -> str:
        return self.raw["lowlevel_encoder"]

    @property
    def highlevel(self) -> List[str]:
        return list(self.raw["highlevel_encoders"])

    @property
    def adversaries(self) -> List[str]:
        return list(self.raw["adversary_encoders"])

    def d(self, *keys: str) -> Any:
        node: Any = self.raw
        for k in keys:
            node = node[k]
        return node


def list_regions(path: str = DEFAULT_PREREG) -> List[str]:
    with open(path, "rb") as fh:
        raw = yaml.safe_load(fh.read())
    return list(raw.get("regions", {}).keys())


def load_prereg(region: str = DEFAULT_REGION, path: str = DEFAULT_PREREG) -> Prereg:
    with open(path, "rb") as fh:
        blob = fh.read()
    sha = hashlib.sha256(blob).hexdigest()
    doc = yaml.safe_load(blob)
    if "regions" not in doc or region not in doc["regions"]:
        avail = list(doc.get("regions", {}).keys())
        raise ValueError(f"region {region!r} not found; available: {avail}")

    merged = _merge(doc, region)
    _validate(merged)
    return Prereg(raw=merged, sha256=sha, path=path, region=region)


def _merge(doc: Dict[str, Any], region: str) -> Dict[str, Any]:
    shared = {k: v for k, v in doc.items() if k != "regions"}
    merged = copy.deepcopy(shared)
    block = copy.deepcopy(doc["regions"][region])
    overrides = block.pop("decision_overrides", {})
    merged.update(block)                                  # region keys win
    merged.setdefault("decision", {}).update(overrides)  # per-region decision tweaks
    merged["region"] = region
    return merged


def _validate(raw: Dict[str, Any]) -> None:
    required = [
        "seed", "target_encoder", "lowlevel_encoder", "highlevel_encoders",
        "adversary_encoders", "all_encoders", "subregions", "decision",
        "reliability", "ceiling", "fitting",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"merged config missing keys: {missing}")
    dec = raw["decision"]
    for k in ("delta", "break_case_k", "recover_fraction",
              "min_recovering_highlevel", "subregion_majority", "ci_level"):
        if k not in dec:
            raise ValueError(f"decision.{k} missing")
    if not (0 < dec["ci_level"] < 1):
        raise ValueError("decision.ci_level must be in (0, 1)")
    if dec["subregion_majority"] > len(raw["subregions"]):
        raise ValueError("decision.subregion_majority exceeds the number of subregions")
    named = set(raw["highlevel_encoders"]) | set(raw["adversary_encoders"]) \
        | {raw["target_encoder"], raw["lowlevel_encoder"]}
    unknown = named - set(raw["all_encoders"])
    if unknown:
        raise ValueError(f"encoders named but absent from all_encoders: {unknown}")
