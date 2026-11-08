#!/usr/bin/env python3
"""Attribute an exported trajectory error to the culling chain.

The exporter writes a frame pose as

    Tcw_export = Tcr * TrwKF * Two

where Tcr is the pose stored for the frame relative to its reference keyframe and
TrwKF is the pose of the surviving keyframe reached by walking the mTcp links up
from that reference keyframe. When the reference keyframe was CULLED, TrwKF is a
RECONSTRUCTION of its pose: it is a frozen relative transform (mTcp), written once
by KeyFrame::SetBadFlag, multiplied by the CURRENT pose of the surviving ancestor.

The online pose trace records, for the same frame, refKFpose_now: the pose the
reference keyframe actually had when the frame was tracked. Therefore

    E = | t(TrwKF * Two) - t(refKFpose_now * Two) |

is exactly how far the reconstruction displaces the very keyframe the estimator
used (the common Tcr factor and its rotation cannot change a translation norm).

  * hops = 0  -> TrwKF IS refKFpose_now, so E must be ~0.
  * hops > 0  -> E is the chain's inconsistency. A large E on hops > 0 rows means
                 the frozen mTcp no longer reconstructs the culled keyframe's
                 pose, which is what makes the exported file jump whenever the
                 reference keyframe switches back to a live one.

The script also reports, when the diagnostic chain column is present, the
per-hop difference between the stored mTcp and the relative transform implied by
the two keyframes' current poses (that is the staleness itself, hop by hop).

Usage:
  chain_error_check.py --diag DIR/export_diag.txt --trace DIR/online_trace.txt
                       [--tum DIR/CameraTrajectory_TUM.txt] [--top 12]
"""

import argparse
import math
import sys


def quat_to_R(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 0:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def apply(R, t):
    return [R[0][0] * t[0] + R[0][1] * t[1] + R[0][2] * t[2],
            R[1][0] * t[0] + R[1][1] * t[1] + R[1][2] * t[2],
            R[2][0] * t[0] + R[2][1] * t[1] + R[2][2] * t[2]]


def norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def pct(xs, p):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = min(len(ys) - 1, max(0, int(round(p / 100.0 * (len(ys) - 1)))))
    return ys[k]


def read_diag(path):
    """Returns (Two, rows) with rows = dicts."""
    two = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("# Two"):
                v = line.split()
                two = ([float(x) for x in v[2:5]], [float(x) for x in v[5:9]])
                continue
            if line.startswith("#"):
                continue
            if not line.strip():
                continue
            head, _, chain = line.partition(" [")
            v = head.split()
            if len(v) < 15:
                continue
            # Older artifacts stop after liveKFpose (15 fields); the OWN pose was
            # added later (22 fields).
            own = ([float(x) for x in v[15:18]], [float(x) for x in v[18:22]]) \
                if len(v) >= 22 else None
            rows.append({
                "ts": float(v[0]),
                "refkf": int(v[1]),
                "refmap": int(v[2]),
                "livekf": int(v[3]),
                "hops": int(v[4]),
                "x": float(v[5]), "y": float(v[6]), "z": float(v[7]),
                "livepose": ([float(x) for x in v[8:11]], [float(x) for x in v[11:15]]),
                "ownpose": own,
                "chain": chain.rstrip("]\n ") if chain else "",
            })
    return two, rows


def read_trace(path):
    """Returns {round(ts,6): dict}."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            v = line.split()
            if len(v) != 29:
                continue
            ts = float(v[0])
            out[round(ts, 6)] = {
                "ts": ts,
                "frame": int(v[1]),
                "state": int(v[2]),
                "is_set": int(v[3]),
                "refkf": int(v[5]),
                "refmap": int(v[6]),
                "parent": int(v[7]),
                "Tcw": ([float(x) for x in v[8:11]], [float(x) for x in v[11:15]]),
                "Tcr": ([float(x) for x in v[15:18]], [float(x) for x in v[18:22]]),
                "refpose_now": ([float(x) for x in v[22:25]], [float(x) for x in v[25:29]]),
            }
    return out


def parse_chain(chain):
    """[(child, parent, dt_stored, dt_implied, dA)]"""
    hops = []
    for tok in chain.split():
        try:
            ids, rest = tok.split(">", 1) if tok.count(">") == 1 else (None, None)
            parts = tok.split(">")
            if len(parts) != 3:
                continue
            child, parent = int(parts[0]), int(parts[1])
            stored, _, tail = parts[2].partition("|")
            implied, _, ang = tail.partition(";dA=")
            st = [float(x) for x in stored.split(",")]
            im = [float(x) for x in implied.split(",")]
            hops.append((child, parent, st, im, float(ang)))
        except Exception:
            continue
    return hops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--tum")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    two, rows = read_diag(a.diag)
    trace = read_trace(a.trace)
    if two is None:
        print("no '# Two' header in the diag: cannot compare (older artifact)")
        return 1
    Rtwo = quat_to_R(two[1])
    t_two = two[0]

    groups = {0: [], 1: []}
    worst = []
    hop_stats = []
    matched = 0
    for r in rows:
        tr = trace.get(round(r["ts"], 6))
        if tr is None:
            continue
        matched += 1
        RA, tA = quat_to_R(r["livepose"][1]), r["livepose"][0]
        RB, tB = quat_to_R(tr["refpose_now"][1]), tr["refpose_now"][0]
        # E = | t(A*Two) - t(B*Two) |
        pa = [tA[i] + apply(RA, t_two)[i] for i in range(3)]
        pb = [tB[i] + apply(RB, t_two)[i] for i in range(3)]
        E = dist(pa, pb)
        key = 0 if r["hops"] == 0 else 1
        groups[key].append(E)
        worst.append((E, r, tr))
        for h in parse_chain(r["chain"]):
            hop_stats.append((h[4], dist(h[2], h[3]), h[0], h[1], r["ts"], r["hops"]))

    print("=== chain-vs-used-reference-pose error  E = |t(TrwKF*Two) - t(refKFpose_now*Two)| ===")
    print("rows: diag=%d matched-to-trace=%d" % (len(rows), matched))
    for key, name in ((0, "hops=0 (direct, must be ~0)"), (1, "hops>0 (reconstructed)")):
        xs = groups[key]
        if not xs:
            print("  %-34s : no rows" % name)
            continue
        print("  %-34s : n=%4d  max=%.4f m  p95=%.4f  median=%.4f"
              % (name, len(xs), max(xs), pct(xs, 95), pct(xs, 50)))

    print()
    print("=== worst rows ===")
    worst.sort(key=lambda w: -w[0])
    for E, r, tr in worst[:a.top]:
        print("  E=%8.4f m  ts=%.6f frame=%5d refKF=%4d liveKF=%4d hops=%d refmap=%d"
              % (E, r["ts"], tr["frame"], r["refkf"], r["livekf"], r["hops"], r["refmap"]))

    if hop_stats:
        dAs = [h[0] for h in hop_stats]
        dTs = [h[1] for h in hop_stats]
        print()
        print("=== per-hop stored mTcp vs implied (current poses), %d hops ===" % len(hop_stats))
        print("  angle  dA_deg : max=%.6f  p95=%.6f  median=%.6f"
              % (max(dAs), pct(dAs, 95), pct(dAs, 50)))
        print("  trans  |dT| m : max=%.6f  p95=%.6f  median=%.6f"
              % (max(dTs), pct(dTs, 95), pct(dTs, 50)))
        hop_stats.sort(key=lambda h: -h[1])
        print("  worst hops:")
        for dA, dT, child, parent, ts, hops in hop_stats[:a.top]:
            print("    |dT|=%8.5f m  dA=%9.5f deg  %d->%d  ts=%.6f (row hops=%d)"
                  % (dT, dA, child, parent, ts, hops))

    if a.tum:
        seq = []
        for line in open(a.tum):
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            seq.append((float(v[0]), [float(v[1]), float(v[2]), float(v[3])]))
        byts = {round(r["ts"], 6): r for r in rows}
        print()
        print("=== exported adjacent-frame steps > 0.30 m (input 30 Hz, ~0.02 m expected) ===")
        n = 0
        for (t0, p0), (t1, p1) in zip(seq, seq[1:]):
            d = dist(p0, p1)
            if d <= 0.30:
                continue
            n += 1
            r0, r1 = byts.get(round(t0, 6)), byts.get(round(t1, 6))
            h0 = r0["hops"] if r0 else -1
            h1 = r1["hops"] if r1 else -1
            k0 = r0["livekf"] if r0 else -1
            k1 = r1["livekf"] if r1 else -1
            print("  %.6f -> %.6f  step=%.4f m  hops %d->%d  liveKF %d->%d" % (t0, t1, d, h0, h1, k0, k1))
        if n == 0:
            print("  none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
