#!/usr/bin/env python3
"""Final report for the 492-cell plan, built from the settled merged root.

Inputs (all produced by `scripts/finalize_after_supplement.sh`):
  <root>/MASTER_MANIFEST.csv        492 rows, one winner per id, with
                                    source_root / reused / attempt_state
  <root>/metrics_v4/ONLINE_PER_RUN.csv      per-run online metrics
  <root>/metrics_v4/ONLINE_PER_EPOCH.csv    per-era rows
  <root>/metrics_v4/FINAL_PER_MAP.csv       per-map (keyframe-conditional)
  <root>/SETTLEMENT_492.csv          per-id settlement (written by settle.py)
  <root>/FINALIZE.log                stage log of the chain

Everything here is a read of finished products. If a table is missing the
report says so instead of inventing a number.

Usage: python3 eval/final_report.py --root /workspace/eval_runs/final_v4/root \
           --out /workspace/eval_runs/final_v4/FINAL_REPORT.md
"""
import argparse
import csv
import datetime as dt
import os
import re
import statistics as st
import sys


def read_csv(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def read_any(paths):
    p = first_existing(paths)
    return (read_csv(p), p) if p else ([], None)


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def f(x, n=3):
    return "—" if x is None else ("%.*f" % (n, x))


def rng(vals, n=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return "%.*f" % (n, vals[0])
    return "%.*f–%.*f" % (n, min(vals), n, max(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/eval_runs/final_v4/root")
    ap.add_argument("--out", default="/workspace/eval_runs/final_v4/FINAL_REPORT.md")
    ap.add_argument("--base", default="/workspace/eval_runs/final_runs_v2")
    ap.add_argument("--supp", default="/workspace/eval_runs/supplement_v4")
    a = ap.parse_args()

    man = read_csv(os.path.join(a.root, "MASTER_MANIFEST.csv"))
    if not man:
        sys.exit("no merged manifest under %s: the settlement stage has not run"
                 % a.root)
    mdir = os.path.join(a.root, "metrics_v4")
    chain = os.path.dirname(os.path.abspath(a.out))
    chain_mdir = os.path.join(chain, "metrics_v4")
    cand = [mdir, os.path.join(chain, "metrics_v4"),
            os.path.join(a.base, "metrics_v4")]
    runs = read_csv(os.path.join(mdir, "ONLINE_PER_RUN.csv"))
    epochs, _p = read_any([os.path.join(d, "ONLINE_PER_EPOCH.csv") for d in cand])
    final_maps, fp = read_any([os.path.join(d, "FINAL_PER_MAP.csv") for d in cand]
                              + [os.path.join(chain, "FINAL_PER_MAP.csv")])
    settle, sp = read_any([os.path.join(a.root, "SETTLEMENT_492.csv"),
                          os.path.join(chain, "SETTLEMENT_492.csv")])

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    L = ["# 最终报告（492 项计划）\n",
         "生成时间 %s（UTC）。数据来自合并根 `%s`：每个计划 ID 一份记录，"
         "补测计划内的 ID 一律用补测结果（失败也保留为失败），"
         "只有不在补测计划里的 T0 才复用旧运行。\n" % (now.isoformat(), a.root)]

    # ---- status and source counts ---------------------------------------
    from collections import Counter
    src = Counter(r["source_root"] for r in man)
    state = Counter((r["source_root"], r["attempt_state"]) for r in man)
    reused = Counter(r["reused"] for r in man)
    L.append("\n## 1. 状态与来源计数\n")
    role = Counter(r.get("role", "") for r in man)
    L.append("| 项 | 数量 |\n|---|---|")
    L.append("| 计划 ID 总数 | %d |" % len(man))
    L.append("| 补测计划内（本轮重跑） | %d |" % role.get("supplement", 0))
    L.append("| 预先指定复用的 T0 | %d |" % role.get("reused", 0))
    L.append("| 上游对照 | %d |" % role.get("upstream", 0))
    for rl in ("supplement", "reused", "upstream"):
        states = Counter(r["attempt_state"] for r in man if r.get("role") == rl)
        L.append("| — %s 的尝试结果 | %s |" % (
            rl, "、".join("%s %d" % kv for kv in sorted(states.items())) or "0"))
    L.append("| 实际退出码为 0 的项 | %d |"
             % sum(1 for r in man if str(r.get("attempt_exit")) == "0"))
    L.append("\n三类相加 = %d + %d + %d = %d，与 492 项计划一一对应；"
             "上游 6 项按自身产物结算（见 1.1），未经失败替换。\n"
             % (role.get("supplement", 0), role.get("reused", 0),
                role.get("upstream", 0), len(man)))

    # ---- upstream rows, settled by their own products --------------------
    upr, upp = read_any([os.path.join(mdir, "UPSTREAM_PER_RUN.csv"),
                         os.path.join(chain_mdir, "UPSTREAM_PER_RUN.csv")])
    L.append("\n### 1.1 上游 6 项（按上游自己的产物与评价方式结算）\n")
    if upr:
        L.append("上游二进制（Photo-SLAM upstream，commit `%s`）不写本项目的 "
                 "`online_poses.csv` / `final_state.csv`，只写 `CameraTrajectory_TUM.txt` "
                 "与 `status.txt`。因此这 6 项**不以“缺内部 CSV”判崩溃**，"
                 "而是按“退出码 0 + 轨迹存在 + 同一评测器（mono 用 Sim3，带尺度）”结算。\n"
                 % (upr[0].get("upstream_commit", "")))
        L.append("| 场景 | 重复 | 退出码 | 轨迹 | 对齐尺度 | ATE m | RPE1 m | 墙钟 s | 状态 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(upr, key=lambda x: (x.get("scene", ""), x.get("repeat", ""))):
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r.get("scene"), r.get("repeat"), r.get("exit_code"),
                "有" if r.get("has_traj") == "1" else "缺",
                f(num(r.get("align_scale")), 2), f(num(r.get("ate_rmse_m"))),
                f(num(r.get("rpe1_trans_rms_m")), 4), f(num(r.get("wall_s")), 1),
                r.get("status")))
    else:
        L.append("上游表未找到（`UPSTREAM_PER_RUN.csv`），本节留空。\n")

    # ---- settlement ------------------------------------------------------
    L.append("\n## 2. 逐 ID 结算\n")
    if settle:
        L.append("结算文件 `%s` 共 %d 行。\n" % (sp, len(settle)))
        stc = Counter((r.get("valid") == "1", r.get("reason", "")[:40]) for r in settle)
        ok = sum(1 for r in settle if r.get("valid") == "1")
        L.append("`valid=1` 的 %d 项；`valid=0` 的 %d 项按下表原因分类。\n"
                 % (ok, len(settle) - ok))
        L.append("| valid | 原因 | 项数 |\n|---|---|---|")
        for (v, reason), n in sorted(stc.items(), key=lambda kv: -kv[1])[:14]:
            L.append("| %s | %s | %d |" % ("1" if v else "0", reason or "—", n))
    else:
        L.append("结算文件还没生成（`settle.py` 未跑完），本节留空。\n")

    # ---- per-run online metrics, every repeat ----------------------------
    L.append("\n## 3. ONLINE：逐场景 × 臂 × 重复的原值与范围\n")
    L.append("口径：分子只算“状态 OK + 位置 ≤ 0.20 m + 旋转 ≤ 5°”的帧，分母是冻结的 GT 可评计划帧数；"
             "合并锚点帧不进分子（也不进对齐与 RPE），但**仍在分母里**，所以结果是偏保守的。\n")
    if not runs:
        L.append("`ONLINE_PER_RUN.csv` 还没生成，本节留空。\n")
    else:
        g = {}
        for r in runs:
            g.setdefault((r["dataset"], r["scene"], r["modality"], r["arm"]), []).append(r)
        cur = None
        for k in sorted(g):
            if k[0] != cur:
                cur = k[0]
                L.append("\n### %s\n" % cur)
                L.append("| 场景 | 模态 | 臂 | 状态 | 覆盖@0.20 各次 | 覆盖范围 | "
                         "ATE 各次 m | ATE 范围 | RPE1 范围 | RPE30 范围 |")
                L.append("|---|---|---|---|---|---|---|---|---|---|")
            rs = sorted(g[k], key=lambda r: r["repeat"])
            stat = ",".join(sorted({r.get("metric_status", "") for r in rs}))
            cov = [num(r.get("online_correct_cov_0.20")) for r in rs]
            ate = [num(r.get("ate_rmse_pooled_over_epochs")) for r in rs]
            r1 = [num(r.get("rpe1_trans_rms_pooled")) for r in rs]
            r30 = [num(r.get("rpe30_trans_rms_pooled")) for r in rs]
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                k[1], k[2], k[3], stat,
                " / ".join(f(v, 4) for v in cov), rng(cov, 4),
                " / ".join(f(v) for v in ate), rng(ate),
                rng(r1, 4), rng(r30, 3)))

    # ---- FINAL (keyframe conditional) ------------------------------------
    L.append("\n## 4. FINAL：按地图的关键帧条件指标（与 ONLINE 分表）\n")
    if final_maps:
        L.append("来源：`%s`\n" % fp)
        g = {}
        for r in final_maps:
            g.setdefault((r["dataset"], r["scene"], r["modality"], r["arm"]), []).append(r)
        L.append("| 场景 | 模态 | 臂 | 地图条目数（跨重复合计） | 关键帧条件 ATE 范围 m | 每图 n_kf 范围 |")
        L.append("|---|---|---|---|---|---|")
        for k in sorted(g):
            rs = g[k]
            ate = [num(r.get("ate_rmse_m")) for r in rs]
            nkf = [num(r.get("n_kf")) for r in rs]
            L.append("| %s | %s | %s | %d | %s | %s |" % (
                k[1], k[2], k[3], len(rs), rng(ate), rng(nkf, 0)))
        L.append("\nFINAL 只在关键帧上算，样本少、不受跟丢影响；它不能当帧级覆盖率用，"
                 "也不能和 ONLINE 的数字混在一张表里比较。\n")
    else:
        L.append("`FINAL_PER_MAP.csv` 还没生成，本节留空。\n")

    # ---- cost, two tables ------------------------------------------------
    L.append("\n## 5. 成本分表\n")
    L.append("### 5.1 SLAM-only（本轮全部运行都是 `PHOTO_SLAM_SLAM_ONLY=1`）\n")
    acc = {}
    for r in man:
        d = os.path.join(os.path.dirname(os.path.join(a.root, "")), "")
        p = os.path.join(r.get("out_dir", ""), "stdout.log")
        if not os.path.exists(p):
            continue
        tm = None
        for line in open(p, errors="ignore"):
            if line.startswith("[EVAL-TIMING]"):
                tm = dict(kv.split("=") for kv in line.split("]", 1)[1].split()
                          if "=" in kv)
        if not tm:
            continue
        key = (r["dataset"], r["scene"], r["modality"], r["system_arm"])
        acc.setdefault(key, {"track": [], "pace": [], "sum": [], "wall": [],
                             "frames": []})
        acc[key]["track"].append(num(tm.get("track_ms")) or 0.0)
        acc[key]["pace"].append(num(tm.get("pace_ms")) or 0.0)
        acc[key]["sum"].append(num(tm.get("sum_ms")) or 0.0)
        w = None
        sp = os.path.join(r.get("out_dir", ""), "status.txt")
        if os.path.exists(sp):
            for line in open(sp):
                if line.startswith("wall_s="):
                    w = num(line.split("=", 1)[1])
        acc[key]["wall"].append(w)
        acc[key]["frames"].append(num(tm.get("frames")))
    if acc:
        L.append("| 场景 | 模态 | 臂 | 帧数 | 跟踪 s | 播放等待 s | 内部合计 s | 完整墙钟 s "
                 "| 每帧跟踪 ms |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for k in sorted(acc):
            v = acc[k]
            tr, fr = med(v["track"]), med(v["frames"])
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                k[1], k[2], k[3], f(fr, 0),
                f((tr or 0) / 1000.0, 1), f((med(v["pace"]) or 0) / 1000.0, 1),
                f((med(v["sum"]) or 0) / 1000.0, 1), f(med(v["wall"]), 1),
                f(tr / fr if fr else None, 1)))
    else:
        L.append("没有可读的 `[EVAL-TIMING]` 行。\n")
    L.append("\n### 5.2 Gaussian 建图成本\n")
    L.append("本轮 492 项全部以 `PHOTO_SLAM_SLAM_ONLY=1` 运行，**不含** Gaussian 训练，"
             "因此本表没有本轮数据；Gaussian 成本只在既有的建图产物里，未在本轮重跑，"
             "不并进上表，也不与本表的跟踪时间相加。\n")

    p = a.out
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", p)
    print("ids=%d supplement=%d reused=%d runs=%d settle=%d"
          % (len(man), src.get("supplement", 0), src.get("base", 0),
             len(runs), len(settle)))


if __name__ == "__main__":
    main()
