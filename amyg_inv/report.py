"""Structured results ledger: JSON + Markdown. Every number is an interval; the frozen
pre-registration hash is stamped in so the parameters behind a verdict are auditable.
"""
from __future__ import annotations

import json
from typing import Dict

from .pipeline import RunResult


def to_dict(res: RunResult, cfg, label: str = "") -> Dict:
    return {
        "label": label,
        "prereg_sha256": res.prereg_sha256,
        "verdict": {
            "branch": res.verdict.branch,
            "reason": res.verdict.reason,
            "n_passing": res.verdict.n_passing,
            "per_subregion": res.verdict.per_subregion,
        },
        "subregions": {
            s.name: {
                "reliability_pass": s.reliability_pass,
                "gate": res.gates[s.name].reason,
                "ceiling_ci": res.diagnostics[s.name]["ceiling"],
                "encoder_r2_normalized": {
                    e: [iv.lo, iv.central, iv.hi] for e, iv in s.encoder_r2.items()
                },
                "emonet_unique_vs_lowlevel": [
                    s.emonet_unique_vs_lowlevel.lo,
                    s.emonet_unique_vs_lowlevel.central,
                    s.emonet_unique_vs_lowlevel.hi,
                ],
            }
            for s in res.subregions
        },
    }


def to_markdown(res: RunResult, cfg, label: str = "") -> str:
    v = res.verdict
    target = cfg.target
    low = cfg.lowlevel
    adv = next((a for a in cfg.adversaries if a != low), None)  # e.g. 'random'
    lines = [
        f"# Verdict: **{v.branch}**" + (f"  ({label})" if label else ""),
        "",
        f"> {v.reason}",
        "",
        f"- region: **{cfg.region}**  ·  target encoder: **{target}**",
        f"- reliability-passing subregions: **{v.n_passing} / {len(res.subregions)}**",
        f"- pre-registration SHA-256: `{res.prereg_sha256[:16]}…`",
        "",
        f"| subregion | gate | ceiling R² (lo·mid·hi) | {target} | {low} | {adv or '—'} | {target}-unique |",
        "|---|---|---|---|---|---|---|",
    ]

    def cell(s, name):
        if name is None or name not in s.encoder_r2:
            return "—"
        iv = s.encoder_r2[name]
        return f"{iv.lo:.2f}·{iv.central:.2f}·{iv.hi:.2f}"

    for s in res.subregions:
        c = res.diagnostics[s.name]["ceiling"]
        ue = s.emonet_unique_vs_lowlevel
        lines.append(
            f"| {s.name} | {'pass' if s.reliability_pass else 'ABSTAIN'} "
            f"| {c[0]:.2f}·{c[1]:.2f}·{c[2]:.2f} "
            f"| {cell(s, target)} | {cell(s, low)} | {cell(s, adv)} "
            f"| {ue.lo:.2f}·{ue.central:.2f}·{ue.hi:.2f} |"
        )
    lines += ["", "_All R² are ceiling-normalized; intervals are subject-bootstrap "
              f"{int(cfg.d('decision','ci_level')*100)}% CIs. Numbers are not claims until the "
              "real-data adapters are wired and the reliability gate is passed on ds002837._"]
    return "\n".join(lines)


def write(res: RunResult, cfg, out_prefix: str, label: str = "") -> None:
    import os
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    with open(out_prefix + ".json", "w") as fh:
        json.dump(to_dict(res, cfg, label), fh, indent=2)
    with open(out_prefix + ".md", "w") as fh:
        fh.write(to_markdown(res, cfg, label))
