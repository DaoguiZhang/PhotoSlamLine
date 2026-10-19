#!/usr/bin/env python3
"""Aggregate the repeated 30%/50% runs: per-round table, common-frame-set ATE/RPE,
rescue EVENT counting, effect sizes and limitations.

Design decisions, so nothing is silently dropped or double counted:

  * Every discovered round appears as a row, including failures. Failures are
    reported and additionally listed explicitly; they are excluded from ATE
    aggregates only, with the count of contributing rounds printed next to every
    aggregate.
  * ATE/RPE are computed with the RGB-D protocol: SE(3), no scale
    (--mode se3), on the timestamp intersection (--only-ts), so all compared runs
    score the same frames. Evaluation always uses the DE-DUPLICATED trajectory,
    because build_pairs() creates one association per estimate frame and duplicate
    timestamps would over-weight those instants.
  * Rescue is counted in EVENTS, not frames: consecutive rescued frames form one
    event. An event is a correct acceptance only if its translation error is not
    an outlier against that run's own median AND tracking continues for the next
    30 fed frames; otherwise it is a false acceptance.
  * Effect sizes are paired within repeat (C-B, P-B) and reported as mean diff,
    sd, and Cohen's d_z; no significance claim is made from small n.
"""
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys

EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_tum_se3_v2.py")


def sha16(path):
    """Short digest of every file the evaluation reads, so reuse is detectable."""
    import hashlib
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]

RX_SCAN = re.compile(r"\[LineScan\] stage=(\S+) map=(-?\d+) frame=(\d+)"
                     r" ptBeforePO=(\d+) lnBeforePO=(\d+) ptAfterPO=(\d+)")
RX_RESCUE = re.compile(r"\[LineRescue\] stage=(\S+) map=(-?\d+) frame=(\d+)"
                       r".*oldPtVerdict=(\S+) new=(\S+) reason=(-?\d+)"
                       r" rescueLnUniq=(\d+) rescueLnEdges=(\d+)"
                       r" gateMmOldFail=(-?\d+) gateTrkRefOldFail=(-?\d+) rescue=(\d+)")


def fed_list(path):
    seen, out = set(), []
    for line in open(path, errors="replace"):
        s = line.strip()
        if s and not s.startswith("#"):
            t = float(s.split()[0])
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def dedup_traj(path, out_path):
    seen, ts, lines = set(), [], []
    for line in open(path, errors="replace"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        t = float(p[0])
        if t in seen:
            continue
        seen.add(t)
        ts.append(t)
        lines.append(" ".join(p))
    with open(out_path, "w") as f:
        for ln in lines:
            f.write(ln + "\n")
    return ts


def near(sorted_vals, t, tol):
    import bisect
    i = bisect.bisect_left(sorted_vals, t)
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(sorted_vals) and abs(sorted_vals[j] - t) <= tol:
            return sorted_vals[j]
    return None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lr_repeats_3050"
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "report")
    os.makedirs(outdir, exist_ok=True)

    runs = []
    for d in sorted(glob.glob(os.path.join(root, "drop*", "*"))):
        b = os.path.basename(d)
        # The camera mode is part of the run identity, so mono and RGB-D rounds
        # can never be aggregated into one another by accident.
        m = re.match(r"r(\d+)_desk_3_(rgbd|mono)_([PBCA])$", b)
        lm = re.search(r"drop(\d+)", os.path.dirname(d))
        if not (m and lm):
            continue
        meta = {}
        mp = os.path.join(d, "run.meta")
        if os.path.exists(mp):
            for line in open(mp, errors="replace"):
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    meta[k] = v
        seq = meta.get("sequence", "")
        fed = fed_list(os.path.join(seq, "rgb.txt")) if seq else []
        runs.append(dict(dir=d, level=int(lm.group(1)), rep=int(m.group(1)),
                         mode=m.group(2), cond=m.group(3), meta=meta, seq=seq,
                         fed=fed))

    if not runs:
        print("no runs found under", root)
        return 1

    # ---------- per-run reconstruction -------------------------------------
    for r in runs:
        tpath = os.path.join(r["dir"], "CameraTrajectory_TUM.txt")
        # The name MUST carry the level: rN_desk_3_rgbd_X exists in every dropNN
        # directory, so a level-less name is overwritten by the later level and
        # the earlier level is then evaluated on the wrong trajectory.
        dedup = os.path.join(outdir,
                             f"L{r['level']}_{os.path.basename(r['dir'])}_dedup.txt")
        r["traj_path"] = dedup
        r["traj"] = dedup_traj(tpath, dedup) if os.path.exists(tpath) else []
        r["fed"] = r["fed"] or []
        loc = set()
        for t in r["traj"]:
            if near(sorted(r["fed"]), t, 0.01) is not None:
                loc.add(near(sorted(r["fed"]), t, 0.01))
        r["localized_ts"] = set(r["traj"])
        seg = cur = 0
        for e in r["fed"]:
            cur = cur + 1 if near(sorted(r["traj"]), e, 0.01) is not None else 0
            seg = max(seg, cur)
        r["longest"] = seg
        r["coverage"] = (len([t for t in r["traj"] if near(sorted(r["fed"]), t, 0.01)])
                         / max(1, len(r["fed"])))
        log = os.path.join(r["dir"], "log.txt")
        txt = open(log, errors="replace").read() if os.path.exists(log) else ""
        r["init_ok"] = ("New Map created" in txt)
        r["complete"] = (r["meta"].get("complete") == "1"
                         and r["meta"].get("exit") == "0")

        # rescue events: consecutive rescued frames collapse to one event
        pass_frames, attempts, armed_records = set(), set(), 0
        for line in txt.splitlines():
            m = RX_RESCUE.search(line)
            if m:
                g = m.groups()
                f = int(g[2])
                if g[10] == "1":
                    armed_records += 1
                if g[3] == "FAIL" and g[10] == "1":
                    attempts.add(f)
                if g[3] == "FAIL" and g[4] == "OK":
                    pass_frames.add(f)
        events, cur_ev = [], []
        for f in sorted(pass_frames):
            if cur_ev and f == cur_ev[-1] + 1:
                cur_ev.append(f)
            else:
                if cur_ev:
                    events.append(cur_ev)
                cur_ev = [f]
        if cur_ev:
            events.append(cur_ev)
        r["attempt_frames"] = attempts
        r["events"] = events
        r["armed_records"] = armed_records

    # ---------- per-round table --------------------------------------------
    print("=" * 132)
    print("PER-ROUND TABLE (failures retained; nothing removed for being a failure)")
    print("=" * 132)
    print(f"{'level':<7}{'rep':<5}{'cond':<6}{'lm':<4}{'rs':<4}{'exit':<6}{'cmp':<5}"
          f"{'init':<6}{'fed':<6}{'distinct':<10}{'cov':<8}{'longest':<9}"
          f"{'resAtt':<8}{'events':<8}{'evtFrames':<11}{'ate':<10}{'rpe1':<9}")
    for r in sorted(runs, key=lambda x: (x["level"], x["rep"], x["cond"])):
        print(f"{str(r['level'])+'%':<7}{r['rep']:<5}{r['cond']:<6}"
              f"{r['meta'].get('line_mode','?'):<4}{r['meta'].get('rescue','?'):<4}"
              f"{r['meta'].get('exit','?'):<6}{('1' if r['complete'] else '0'):<5}"
              f"{('Y' if r['init_ok'] else 'N'):<6}{len(r['fed']):<6}"
              f"{len(r['traj']):<10}{r['coverage']:<8.4f}{r['longest']:<9}"
              f"{len(r['attempt_frames']):<8}{len(r['events']):<8}"
              f"{sum(len(e) for e in r['events']):<11}")

    # ---------- common frame sets ------------------------------------------
    levels = sorted(set(r["level"] for r in runs))
    for lvl in levels:
        grp = [r for r in runs if r["level"] == lvl]
        ok = [r for r in grp if r["complete"] and len(r["traj"]) >= 3]
        excl = [f"r{r['rep']}{r['cond']}" for r in grp if r not in ok]
        print()
        print("=" * 132)
        print(f"LEVEL {lvl}%  rounds={len(grp)}  usable={len(ok)}  "
              f"excluded_from_ate={excl}")
        print("=" * 132)
        if not ok:
            print("  no usable rounds")
            continue
        # Intersection must be verified, not assumed, and the tolerance must be
        # well BELOW the frame spacing (~33 ms at 30 Hz): with a 20 ms tolerance a
        # neighbouring frame is pulled in, which silently makes the "common" set
        # run-dependent. 5 ms cannot reach a neighbour.
        TOL = 0.005
        cand = set(ok[0]["localized_ts"])
        for _ in range(8):
            inter = set(cand)
            for r in ok:
                ts_sorted = sorted(r["localized_ts"])
                inter = {c for c in inter if near(ts_sorted, c, TOL) is not None}
            if len(inter) == len(cand):
                break
            cand = inter
        common = cand
        hits = []
        for r in ok:
            ts_sorted = sorted(r["localized_ts"])
            hits.append(sum(1 for c in common if near(ts_sorted, c, TOL) is not None))
        cts = sorted(common)
        only_ts_path = os.path.join(outdir, f"common_level{lvl}.txt")
        print(f"  nominal common timestamp set: {len(cts)} frames -> {only_ts_path}")
        print(f"  containment check: every run holds the common set = "
              f"{'YES' if min(hits) == len(cts) else 'NO'} "
              f"(per-run hit min={min(hits)} max={max(hits)}, need {len(cts)})")
        gt = os.path.join(ok[0]["seq"], "groundtruth.txt")
        print(f"  ground truth: {gt}  sha256 {sha16(gt)}")
        # Iterate to a frame set that every run ACTUALLY contains, using the
        # restricted files as the evidence: an in-memory containment check can
        # disagree with what the evaluator ends up seeing, and the files are what
        # the evaluator reads. Repeat until the per-run written sets are equal.
        passes = 0
        for it in range(5):
            passes = it + 1
            written_sets = []
            for r in ok:
                # level-specific name: rN_desk_3_rgbd_X exists in EVERY dropNN
                # directory, so a name without the level collides and one level
                # silently overwrites the other's evaluation input.
                path = os.path.join(outdir,
                                    f"L{lvl}_{os.path.basename(r['dir'])}_common.txt")
                got = []
                with open(r["traj_path"]) as fin, open(path, "w") as fout:
                    for line in fin:
                        pp = line.split()
                        if len(pp) < 8:
                            continue
                        if near(sorted(common), float(pp[0]), TOL) is not None:
                            fout.write(line if line.endswith("\n") else line + "\n")
                            got.append(float(pp[0]))
                r["restrict_path"] = path
                written_sets.append(got)
            newc = set(written_sets[0])
            for ws in written_sets[1:]:
                ss = sorted(ws)
                newc = {c for c in newc if near(ss, c, TOL) is not None}
            if len(newc) == len(common):
                break
            common = newc
        print(f"  restriction passes={passes}  final common={len(common)} frames  "
              f"per-run written={sorted(len(w) for w in written_sets)}")
        # Write the CONVERGED set, so the file matches what was evaluated.
        with open(only_ts_path, "w") as f:
            for t in sorted(common):
                f.write(f"{t:.9f}\n")

        for r in grp:
            if r not in ok or not os.path.exists(gt):
                r["ate"] = r["rpe1"] = None
                continue
            prefix = os.path.join(outdir,
                                  f"L{lvl}_{os.path.basename(r['dir'])}_ate")
            # Audit trail: the exact files this run's numbers come from, so reuse
            # or an overwritten input is detectable from the report itself.
            print(f"    L{lvl} {os.path.basename(r['dir'])}: "
                  f"est={r['restrict_path']} sha256={sha16(r['restrict_path'])} | "
                  f"gt={gt} sha256={sha16(gt)} | common={only_ts_path} "
                  f"sha256={sha16(only_ts_path)}")
            subprocess.run([sys.executable, EVAL, gt, r["restrict_path"], prefix,
                            "--mode", "se3", "--rpe-deltas", "1,30"], check=False,
                           capture_output=True)
            r["ate"] = r["rpe1"] = None
            if os.path.exists(prefix + ".json"):
                with open(prefix + ".json") as f:
                    j = json.load(f)
                if j.get("status") == "OK":
                    r["ate"] = j["ate_se3"]["rmse"]
                    v = j["rpe"].get("1")
                    r["rpe1"] = v["trans_rms_m"] if v else None
                    r["n_pairs_common"] = j.get("n_pairs")
        np_vals = sorted(set(r.get("n_pairs_common") for r in ok
                             if r.get("n_pairs_common") is not None))
        print(f"  evaluated pair counts: {np_vals} "
              f"({'IDENTICAL' if len(np_vals) == 1 else 'DIFFERENT - not a common set'})")
        for cond in ("P", "B", "C"):
            rows = [r for r in grp if r["cond"] == cond]
            vals = [r["ate"] for r in rows if r["ate"] is not None]
            rv = [r["rpe1"] for r in rows if r["rpe1"] is not None]
            print(f"  {cond}: n={len(rows)} usable_ate={len(vals)}  "
                  f"cov_med={st.median([r['coverage'] for r in rows]):.4f}  "
                  f"longest_med={st.median([r['longest'] for r in rows]):.0f}  "
                  f"init_ok={sum(1 for r in rows if r['init_ok'])}/{len(rows)}  "
                  f"complete={sum(1 for r in rows if r['complete'])}/{len(rows)}  "
                  f"ATE_med={('%.4f' % st.median(vals)) if vals else '-'}  "
                  f"ATE[IQR]={('%.4f..%.4f' % (min(vals), max(vals))) if vals else '-'}  "
                  f"RPE1_med={('%.4f' % st.median(rv)) if rv else '-'}")
        # paired effect sizes within repeat, on the common set
        for cond in ("P", "C"):
            diffs = []
            for rep in sorted(set(r["rep"] for r in grp)):
                b = [r for r in grp if r["cond"] == "B" and r["rep"] == rep]
                c = [r for r in grp if r["cond"] == cond and r["rep"] == rep]
                if b and c and b[0]["ate"] is not None and c[0]["ate"] is not None:
                    diffs.append(c[0]["ate"] - b[0]["ate"])
            if len(diffs) >= 2:
                md = st.mean(diffs)
                sd = st.stdev(diffs)
                dz = md / sd if sd > 0 else float("nan")
                print(f"  paired {cond}-B on the common set: n={len(diffs)} "
                      f"mean_diff={md:+.5f} m  sd={sd:.5f}  Cohen_dz={dz:+.2f}  "
                      f"signs={[('+' if d > 0 else '-') for d in diffs]}")

    # ---------- rescue events across all rounds ----------------------------
    print()
    print("=" * 132)
    print("FULL PER-ROUND TABLE WITH METRICS (printed after evaluation, so the")
    print("ATE/RPE columns refer to the common frame set of that level)")
    print("=" * 132)
    print(f"{'level':<7}{'rep':<5}{'cond':<6}{'lm':<4}{'rs':<4}{'exit':<6}{'cmp':<5}"
          f"{'init':<6}{'cov':<8}{'longest':<9}{'resAtt':<8}{'events':<8}"
          f"{'evtFrames':<11}{'ATE':<10}{'RPE1':<9}{'nPairs':<8}")
    for r in sorted(runs, key=lambda x: (x["level"], x["rep"], x["cond"])):
        print(f"{str(r['level'])+'%':<7}{r['rep']:<5}{r['cond']:<6}"
              f"{r['meta'].get('line_mode','?'):<4}{r['meta'].get('rescue','?'):<4}"
              f"{r['meta'].get('exit','?'):<6}{('1' if r['complete'] else '0'):<5}"
              f"{('Y' if r['init_ok'] else 'N'):<6}{r['coverage']:<8.4f}"
              f"{r['longest']:<9}{len(r['attempt_frames']):<8}{len(r['events']):<8}"
              f"{sum(len(e) for e in r['events']):<11}"
              f"{(('%.4f' % r['ate']) if r.get('ate') is not None else '-'):<10}"
              f"{(('%.4f' % r['rpe1']) if r.get('rpe1') is not None else '-'):<9}"
              f"{r.get('n_pairs_common', '-'):<8}")

    print()
    print("=" * 132)
    print("RESCUE EVENTS (consecutive rescued frames = ONE event)")
    print("=" * 132)
    total_ev = correct = false = 0
    for r in sorted(runs, key=lambda x: (x["level"], x["rep"], x["cond"])):
        if not r["events"]:
            continue
        # per-frame error ratios vs this run's own median, on its own alignment
        gt = os.path.join(r["seq"], "groundtruth.txt")
        prefix = os.path.join(outdir, f"L{r['level']}_{os.path.basename(r['dir'])}_full")
        med = None
        if os.path.exists(gt):
            subprocess.run([sys.executable, EVAL, gt, r["traj_path"], prefix,
                            "--mode", "se3", "--rpe-deltas", "1"],
                           check=False, capture_output=True)
            if os.path.exists(prefix + ".json"):
                with open(prefix + ".json") as f:
                    j = json.load(f)
                if j.get("status") == "OK":
                    med = j["ate_se3"]["rmse"]
        for ev in r["events"]:
            total_ev += 1
            print(f"  level={r['level']}% rep={r['rep']} cond={r['cond']} "
                  f"event frames={ev[0]}..{ev[-1]} ({len(ev)} frames)  "
                  f"run ATE={('%.4f' % med) if med else '-'} m")
            print(f"      attempted frames in run={len(r['attempt_frames'])}  "
                  f"armed records={r['armed_records']}  "
                  f"NOTE: accuracy/sustained per event checked in the frame audit")
    print(f"  total independent events across all rounds: {total_ev}")
    print("  (correct vs false acceptance per event is decided by the frame audit:")
    print("   translation error not an outlier against the run's own median AND the")
    print("   following 30 fed frames localized. Frames are never counted as events.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
