#!/usr/bin/env python3
"""Is a map merge ONE transform per event, per keyframe?

Earlier I "convicted" a merge of placing a map's keyframes inconsistently by
comparing two keyframes' spatial distance (0.874 m). That is not a proof: two
keyframes at different places are expected to end up at different places. The
right question is whether every moved keyframe satisfies

    t'_k  ==  R_G * t_k + T_G          for ONE (R_G, T_G) per merge event

The audit's MERGE rows give, per moved keyframe, the pose translation before and
after and the applied rotation MAGNITUDE, which is identical for all keyframes of
an event - so |w| is known and only the rotation AXIS (2 dof) plus T_G (3) have to
be fitted. Method: coarse grid over the axis, closed-form optimal T_G for each
trial axis (T_G = mean(t'_k - R t_k)), then local refinement. No single keyframe is
trusted, which is exactly the criticism of the earlier argument.

Also reported: keyframes moved TWICE inside one event. The merge applies poses in
two loops (the local window, then the whole current map), so a keyframe in both
gets two transforms - a single-G fit must flag those, and they are not evidence of
an inconsistency.

Usage: merge_transform_audit.py --audit DIR/audit.txt [--max-events 4] [--top 6]
"""

import argparse
import math
import re
import sys


ROW = re.compile(
    r"^(\d+) ([\d.]+) MERGE from_map=(-?\d+) to_map=(-?\d+) scale=([\d.eE+-]+) "
    r"kf=(\d+) dT=([\d.eE+-]+) dR_deg=([\d.eE+-]+) before\(([-\d.eE,]+)\) after\(([-\d.eE,]+)\)")


def parse_events(path, seq_gap=400):
    events = []
    with open(path, errors="ignore") as f:
        for line in f:
            if " MERGE " not in line:
                continue
            m = ROW.match(line)
            if not m:
                continue
            seq, fm, tm = int(m.group(1)), int(m.group(3)), int(m.group(4))
            sc, kf = float(m.group(5)), int(m.group(6))
            dR = float(m.group(8))
            before = [float(x) for x in m.group(9).split(",")]
            after = [float(x) for x in m.group(10).split(",")]
            if events and events[-1]["fm"] == fm and events[-1]["tm"] == tm \
                    and abs(events[-1]["scale"] - sc) < 1e-9 and seq - events[-1]["last"] <= seq_gap:
                events[-1]["rows"].append((seq, kf, before, after, dR))
                events[-1]["last"] = seq
            else:
                events.append({"fm": fm, "tm": tm, "scale": sc, "first": seq, "last": seq,
                               "rows": [(seq, kf, before, after, dR)]})
    return events


def rodrigues(ax, th):
    n = math.sqrt(sum(x * x for x in ax))
    if n < 1e-12 or abs(th) < 1e-12:
        return [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    k = [x / n for x in ax]
    c, s = math.cos(th), math.sin(th)
    K = [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]
    return [[(1 - c) * k[i] * k[j] + c * (1 if i == j else 0) + s * K[i][j] for j in range(3)]
            for i in range(3)]


def mv(R, t):
    return [sum(R[i][j] * t[j] for j in range(3)) for i in range(3)]


def rss_for_axis(rows, ax, th):
    R = rodrigues(ax, th)
    pred = [mv(R, b) for _, _, b, _, _ in rows]
    n = len(rows)
    T = [sum(a[i] - pred[j][i] for j, (_, _, _, a, _) in enumerate(rows)) / n for i in range(3)]
    rss = 0.0
    for j, (_, _, _, a, _) in enumerate(rows):
        for i in range(3):
            d = a[i] - (pred[j][i] + T[i])
            rss += d * d
    return rss, R, T


def fit(rows):
    """Coarse axis grid (|w| = the event's dR) then local refinement."""
    th = math.radians(rows[0][4])
    best = None
    for i in range(25):
        theta = math.pi * i / 24.0
        for j in range(2 if i in (0, 24) else 48):
            phi = 2 * math.pi * j / 48.0
            ax = (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
            rss, R, T = rss_for_axis(rows, ax, th)
            if best is None or rss < best[0]:
                best = (rss, ax, th, R, T)
    # local refinement on the axis and on |w|
    step = 0.25
    rss, ax, th, R, T = best
    for _ in range(40):
        improved = False
        for dth in (step, -step, 0.0):
            for dphi in (0.0, step, -step):
                if dth == 0 and dphi == 0:
                    continue
                ax2 = list(ax)
                # perturb by rotating the axis slightly
                ax2[0] += dphi * ax2[1] - dth * ax2[2]
                ax2[1] += -dphi * ax2[0] + dth * 0.0
                ax2[2] += dth * ax2[0]
                rss2, R2, T2 = rss_for_axis(rows, ax2, th + dth)
                if rss2 < rss:
                    rss, ax, th, R, T = rss2, ax2, th + dth, R2, T2
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    res = []
    for seq, kf, b, a, dR in rows:
        p = mv(R, b)
        res.append((math.sqrt(sum((a[i] - (p[i] + T[i])) ** 2 for i in range(3))), kf, seq))
    return R, T, th, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--max-events", type=int, default=4)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    events = parse_events(a.audit)
    print("=== %s: %d merge event(s) ===" % (a.audit, len(events)))
    for ev in events:
        kfs = [r[1] for r in ev["rows"]]
        dup = sorted({k for k in kfs if kfs.count(k) > 1})
        print("  seq %d..%d map %d->%d scale=%.6f rows=%d unique_kfs=%d dR=%.4f deg%s"
              % (ev["first"], ev["last"], ev["fm"], ev["tm"], ev["scale"], len(kfs),
                 len(set(kfs)), ev["rows"][0][4],
                 ("  KFs moved twice: %s" % dup) if dup else ""))
    for ev in events[:a.max_events]:
        # fit on the FIRST move of each keyframe (a second move is a different loop)
        seen = set()
        first = []
        for r in ev["rows"]:
            if r[1] in seen:
                continue
            seen.add(r[1])
            first.append(r)
        if len(first) < 4:
            print("\n  map %d->%d: too few keyframes (%d) to fit G" % (ev["fm"], ev["tm"], len(first)))
            continue
        R, T, th, res = fit(first)
        rs = sorted(x[0] for x in res)
        med = rs[len(rs) // 2]
        p95 = rs[min(len(rs) - 1, int(0.95 * (len(rs) - 1)))]
        print("\n  map %d->%d scale=%.6f : fitted G rotation=%.4f deg |T|=%.6f m over %d KFs"
              % (ev["fm"], ev["tm"], ev["scale"], math.degrees(th), math.sqrt(sum(x * x for x in T)),
                 len(first)))
        print("    per-KF residual |t' - (R t + T)|: median=%.6f p95=%.6f max=%.6f (m)"
              % (med, p95, rs[-1]))
        print("    worst:")
        for r, kf, seq in sorted(res, reverse=True)[:a.top]:
            print("      kf=%-5d residual=%.6f m  (MERGE row seq %d)" % (kf, r, seq))
        # reference: the model that trusts the identity-pose keyframe alone
        for r in first:
            if max(abs(x) for x in r[2]) < 1e-6 and sum(abs(x) for x in r[2]) < 1e-6:
                T1 = [r[3][i] for i in range(3)]
                rs1 = []
                for _, kf, b, aa, _ in first:
                    p = mv(rodrigues([0, 0, 1], 0.0), b)   # R = I placeholder (origin KF has no rotation info)
                    rs1.append((math.sqrt(sum((aa[i] - (b[i] + T1[i])) ** 2 for i in range(3))), kf))
                rs1.sort(reverse=True)
                print("    [reference] pure-translation model fitted from the identity-pose kf=%d: "
                      "median residual=%.6f max=%.6f"
                      % (r[1], sorted(x[0] for x in rs1)[len(rs1) // 2], rs1[0][0]))
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
