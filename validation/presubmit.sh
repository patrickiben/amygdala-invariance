#!/usr/bin/env bash
# One-command pre-submission gate for the amygdala false-floor manuscript.
#   bash validation/presubmit.sh
# Bundles: path guard, numbers-gate, robustness re-derivation, citation-integrity, AI-artifact scrub.
# Modeled on ~/Documents/directed-asymmetry-contagion/tools/presubmit.sh. Exit 0 = all clear.
set -uo pipefail
V="$(cd "$(dirname "$0")" && pwd)"; RD="$V/../real_data"; RE="$(cd "$V/../.." && pwd)"; PY="${PYTHON:-python3}"; fail=0
run(){ echo; echo ">> $1"; printf '%.0s-' {1..70}; echo; shift; "$@"; rc=$?; [ $rc -ne 0 ] && { fail=1; echo "   [FAILED rc=$rc]"; }; return 0; }
echo "=== PRE-SUBMISSION GATE — amygdala false-floor manuscript ==="
run "G1 · absolute-path guard"        bash "$V/check_paths.sh" "$RD"
run "G2 · numbers-gate (numbers->logs)" "$PY" "$V/numbers_gate.py"
run "G4 · robustness re-derivation"   "$PY" "$V/robustness_gate.py"
run "G3 · citation-integrity gate"    "$PY" "$V/check_citations.py" "$V/refs.tex" --mailto patrickiben@gmail.com
run "G7 · AI-artifact / hidden-text"  "$PY" "$V/scrub_ai_artifacts.py" "$RE/MANUSCRIPT_amygdala-false-floor_v2.md"
echo; echo "=============================================================================="
[ $fail -eq 0 ] && echo " RESULT: ALL CLEAR — deterministic pre-submission checks pass" || echo " RESULT: ATTENTION NEEDED — see above"
echo "=============================================================================="
exit $fail
