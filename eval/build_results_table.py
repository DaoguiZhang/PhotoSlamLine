#!/usr/bin/env python3
"""Build the results table from the validated evaluator's per-run ATE plus the
accepted-frame counts from final_state.csv.

Input: /workspace/eval_runs/cabinet2_ate.txt produced by
  for d in */*/*/rep*; do ... eval_tum_se3_v2.py ... done
No number is typed by hand; this script only aggregates and formats.
"""
import statistics as st
from collections import defaultdict

rows = []
for line in open("/workspace/eval_runs/cabinet2_ate.txt"):
    line = line.strip()
    if not line:
        continue
    d, ate, ok = line.split()
    scene, mod, arm, rep = d.split("/")
    okn, tot = ok.split("=")[1].split("/")
    rows.append({
        "scene": scene, "mod": mod, "arm": arm, "rep": rep,
        "ate": float(ate.split("=")[1]),
        "ok": int(okn), "tot": int(tot), "cov": int(okn) / int(tot),
    })

cells = defaultdict(list)
for r in rows:
    cells[(r["scene"], r["mod"], r["arm"])].append(r)

out = []
out.append("| scene | modality | arm | runs | accepted coverage "
           "median [min,max] | ATE RMSE (m) median [min,max] | per-run ATE |")
out.append("|---|---|---|---|---|---|---|")
for key in sorted(cells):
    rs = cells[key]
    cov = [r["cov"] for r in rs]
    ate = [r["ate"] for r in rs]
    out.append("| %s | %s | %s | %d | %.3f [%.3f,%.3f] | %.4f [%.4f,%.4f] | %s |" % (
        key[0], key[1], key[2], len(rs),
        st.median(cov), min(cov), max(cov),
        st.median(ate), min(ate), max(ate),
        " ".join("%.4f" % a for a in sorted(ate))))
open("/workspace/deliv/cabinet-eval/eval/RESULTS_TABLE.md", "w").write(
    "\n".join(out) + "\n")
print("\n".join(out))
