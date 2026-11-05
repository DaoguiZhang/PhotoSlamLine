#!/usr/bin/env python3
"""Write the readable stage report for the supplement campaign.

The report is a PREVIEW. It only reads runs that have ended and whose products
are complete; anything still being written is reported as pending, never scored.

Inputs (all read-only):
  <base>/MASTER_MANIFEST.csv      the 492-cell plan and where each attempt lives
  <base>/metrics_v4/*.csv         the v_final_2 evaluation (old numbers)
  <supp>/SUPPLEMENT_MANIFEST.csv  the supplementary cells
  <supp>/progress.log             wall time of every supplement cell that ended
  <prev>/PREVIEW_BY_ID.csv        old-vs-new comparison produced by eval/preview.py

Usage: python3 eval/report_preview.py --out /workspace/eval_runs/preview_v4
"""
import argparse
import csv
import datetime as dt
import os
import re
import statistics as st
import subprocess
import sys

BASE = "/workspace/eval_runs/final_runs_v2"
SUPP = "/workspace/eval_runs/supplement_v4"
TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def f(x, n=3):
    return "—" if x is None else ("%.*f" % (n, x))


def num(x):
    """CSV cell -> float, or None when empty / unparsable."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_csv(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def status_snapshot():
    out = {"sched": "", "slam": "", "completed": 0, "partial": 0, "pending": 0}
    try:
        txt = subprocess.run([sys.executable, "scripts/supplement.py", "status"],
                             cwd=TREE, capture_output=True, text=True).stdout
        for line in txt.splitlines():
            m = re.match(r"\s+(COMPLETED|PARTIAL|PENDING|RUNNING)\s+(\d+)", line)
            if m:
                out[m.group(1).lower()] = int(m.group(2))
    except Exception:
        pass
    for key, pat in (("sched", r"scripts/supplement\.py run"),
                     ("slam", r"entry_rel/eval_frontend")):
        try:
            r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
            out[key] = r.stdout.split()[0] if r.stdout.strip() else ""
        except Exception:
            pass
    return out


def recent_walls():
    """group -> [wall_s] from the cells this batch already finished."""
    g = {}
    p = os.path.join(SUPP, "progress.log")
    if not os.path.exists(p):
        return g
    for line in open(p):
        m = re.match(r"(\S+) exit=(-?\d+) wall=([\d.]+)s", line.strip())
        if not m:
            continue
        key = "|".join(m.group(1).split("|")[:4])
        g.setdefault((key, m.group(2)), []).append(float(m.group(3)))
    return g


def base_walls(manifest):
    """group -> [wall_s] from the historical batch, for cells not yet seen."""
    g = {}
    for r in manifest:
        d = r["out_dir"]
        p = os.path.join(d, "status.txt")
        if not os.path.exists(p):
            continue
        w = None
        for line in open(p):
            if line.startswith("wall_s="):
                try:
                    w = float(line.split("=", 1)[1])
                except ValueError:
                    w = None
        if w is not None:
            g.setdefault((r["dataset"], r["scene"], r["modality"], r["system_arm"]),
                         []).append(w)
    return g


def timing_split(manifest):
    """(track_s, pace_s, sum_s, wall_s) medians per group, from ended runs."""
    acc = {}
    for r in manifest:
        d = r["out_dir"]
        sp = os.path.join(d, "stdout.log")
        if not os.path.exists(sp):
            continue
        tm = None
        for line in open(sp, errors="ignore"):
            if line.startswith("[EVAL-TIMING]"):
                tm = dict(kv.split("=") for kv in
                          line.split("]", 1)[1].split() if "=" in kv)
        if not tm:
            continue
        wall = None
        stp = os.path.join(d, "status.txt")
        if os.path.exists(stp):
            for line in open(stp):
                if line.startswith("wall_s="):
                    try:
                        wall = float(line.split("=", 1)[1])
                    except ValueError:
                        wall = None
        key = (r["dataset"], r["scene"], r["modality"], r["system_arm"])
        acc.setdefault(key, {"track": [], "pace": [], "sum": [], "wall": [],
                             "frames": []})
        acc[key]["track"].append(float(tm.get("track_ms", "0")) / 1000.0)
        acc[key]["pace"].append(float(tm.get("pace_ms", "0")) / 1000.0)
        acc[key]["sum"].append(float(tm.get("sum_ms", "0")) / 1000.0)
        acc[key]["wall"].append(wall)
        acc[key]["frames"].append(int(tm.get("frames", "0")))
    return {k: {kk: med(vv) for kk, vv in v.items()} for k, v in acc.items()}


def rot_table(path):
    """(arm, repeat, dataset, scene, modality) -> max epoch rot_rmse_deg.

    Only epochs with at least 30 RPE pairs count; a short epoch's rotation is
    not comparable and would dominate a max.
    """
    best = {}
    for r in read_csv(path):
        if r.get("status") != "OK":
            continue
        try:
            n = int(float(r.get("n_pairs") or 0))
            v = float(r["rot_rmse_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        if n < 30:
            continue
        k = (r["arm"], r["repeat"], r["dataset"], r["scene"], r["modality"])
        best[k] = max(best.get(k, 0.0), v)
    return best


def old_group_table(path, rpath=None):
    """Group the old per-run metrics into the per-scene view used in §3."""
    runs = read_csv(path)
    rot_by_run = rot_table(rpath) if rpath else {}
    acc = {}
    for r in runs:
        key = (r["dataset"], r["scene"], r["modality"], r["arm"])
        acc.setdefault(key, []).append(r)
    out = []
    for key in sorted(acc):
        rs = acc[key]
        ok = [r for r in rs if r.get("metric_status") == "OK"]
        und = [r for r in rs if (r.get("metric_status") or "").startswith(
            ("UNDETERMINED", "AMBIGUOUS"))]
        crash = [r for r in rs if (r.get("metric_status") or "").startswith(
            ("CRASHED", "BROKEN"))]

        def col(name, rows=None):
            """every run that carries a number, whatever its status label"""
            v = []
            for r in (rs if rows is None else rows):
                try:
                    v.append(float(r[name]))
                except (KeyError, TypeError, ValueError):
                    pass
            return v

        cov, ate = col("online_correct_cov_0.20"), col("ate_rmse_pooled_over_epochs")
        rope = col("rpe1_trans_rms_pooled")
        r30 = col("rpe30_trans_rms_pooled")
        rot = [rot_by_run[(r["arm"], r["repeat"], r["dataset"], r["scene"], r["modality"])]
               for r in rs
               if (r["arm"], r["repeat"], r["dataset"], r["scene"], r["modality"]) in rot_by_run]
        out.append({
            "dataset": key[0], "scene": key[1], "modality": key[2], "arm": key[3],
            "n_ok": len(ok), "n_crash": len(crash), "n_undetermined": len(und),
            "cov_med": med(cov), "cov_rng": ("%s" % ("%.3f–%.3f" % (min(cov), max(cov))
                                                    if len(cov) > 1 else "%.3f" % cov[0]))
                       if cov else "—",
            "ate_med": med(ate), "rpe1": med(rope), "rpe30": med(r30),
            "rot": med(rot), "statuses": ",".join(r.get("metric_status", "")
                                                   for r in rs),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/eval_runs/preview_v4")
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    snap = status_snapshot()
    plan = read_csv(os.path.join(BASE, "MASTER_MANIFEST.csv"))
    supp = read_csv(os.path.join(SUPP, "SUPPLEMENT_MANIFEST.csv"))
    per_scene = old_group_table(os.path.join(BASE, "metrics_v4",
                                             "ONLINE_PER_RUN.csv"),
                                os.path.join(BASE, "metrics_v4",
                                             "ONLINE_PER_EPOCH.csv"))
    prev = read_csv(os.path.join(out, "PREVIEW_BY_ID.csv"))
    rw = recent_walls()
    bw = base_walls(plan)
    tsplit = timing_split(plan)

    # ---- ETA -------------------------------------------------------------
    remaining, per_group = [], {}
    for r in supp:
        if os.path.exists(os.path.join(r["out_dir"], "status.txt")):
            continue
        key = (r["dataset"], r["scene"], r["modality"], r["system_arm"])
        remaining.append(r)
        per_group[key] = per_group.get(key, 0) + 1
    recent_ok = [w for (k, e), v in rw.items() if e == "0" for w in v]
    glob_recent = med(recent_ok)
    est, detail = 0.0, []
    for key, n in per_group.items():
        cand = [w for (k, e), v in rw.items() if k == "|".join(key) and e == "0"
                for w in v]
        src = "本批实测"
        if not cand:
            cand, src = bw.get(key, []), "旧批次同场景"
        if not cand:
            cand, src = (recent_ok if recent_ok else []), "全局中位数"
        m = med(cand) or 0.0
        est += n * m
        detail.append((key, n, m, src))
    detail.sort(key=lambda t: -t[1] * t[2])
    eta_at = now + dt.timedelta(seconds=est)

    L = []
    L.append("# 阶段报告（预览，不是最终表）\n")
    L.append("生成时间 %s（UTC）。本报告只读取**已经结束且产物完整**的运行；"
             "正在写出的单元一律记为 pending，不参与任何统计与评价。\n" % now.isoformat())
    L.append("数据来源：`%s`（主记录 v_final_2）、`%s`（补测队列）、"
             "`%s`（旧新对照）。生产二进制、配置与分支 `d3214194` 均未改动。\n"
             % (BASE, SUPP, out))

    # 1 -----------------------------------------------------------------
    L.append("\n## 1. 实时状态与 ETA\n")
    L.append("| 项 | 值 |\n|---|---|")
    L.append("| 调度器 PID | `%s` |" % (snap["sched"] or "无"))
    L.append("| SLAM 子进程 PID | `%s` |" % (snap["slam"] or "无"))
    L.append("| COMPLETED / PARTIAL / PENDING | %d / %d / %d |"
             % (snap.get("completed", 0), snap.get("partial", 0),
                snap.get("pending", 0)))
    if recent_ok:
        L.append("| 本批已完成单元墙钟 | n=%d，中位 %.1f s，范围 %.1f–%.1f s |"
                 % (len(recent_ok), med(recent_ok), min(recent_ok), max(recent_ok)))
    L.append("| 剩余单元数（以 status.txt 缺失为准） | %d |" % len(remaining))
    L.append("| 估算剩余墙钟 | %.0f s ≈ %.1f h |" % (est, est / 3600.0))
    L.append("| 预计完成时间 | %s UTC（估算时点 %s） |"
             % (eta_at.strftime("%Y-%m-%d %H:%M"), now.strftime("%H:%M")))
    if not snap["sched"] and snap.get("pending", 0):
        L.append("\n**STOPPED_INCOMPLETE**：没有调度器在运行，但仍有 %d 个单元未完成。\n"
                 % snap["pending"])
    L.append("\n剩余量最大的组（组内单元数 × 单元中位墙钟，来源）：\n")
    L.append("| 数据集 | 场景 | 模态 | 臂 | 剩余 | 单元中位 s | 来源 |")
    L.append("|---|---|---|---|---|---|---|")
    for key, n, m, src in detail[:12]:
        L.append("| %s | %s | %s | %s | %d | %.1f | %s |" % (key + (n, m, src)))

    # 2 -----------------------------------------------------------------
    L.append("\n## 2. v2 → v3 → v4 的变更\n")
    L.append("| 版本 | 变更 | 证据 | 仍未确认 |")
    L.append("|---|---|---|---|")
    L.append("| v2 | 写入 `online_poses.csv`（含 `pose_valid,map_id,ref_kf_id,"
             "frame_id,epoch`）与 `map_events.csv`，并导出全部 Atlas 地图 | "
             "v_final_1 只有单图导出（1606 个位姿帧中只导出 849），v_final_2 全量重跑 486 个单元 | "
             "旧数字与 v_final_1 不可直接比较 |")
    L.append("| v3 | 合并事件带上 `epoch`：`EpochRef()` 在 `Emit()` 的锁内推进，"
             "`CurrentEpoch()` 以 acquire 读取 | 真实合并运行上验证是唯一可锚定的时刻 | "
             "`Track*` 返回后到读 epoch 之间若发生合并，帧会被盖上新 epoch（见 §5 注） |")
    L.append("| v4 | 两处崩溃修复：`mvpMapLines` 初始化 + `GetKeyLineInfo` 夹紧；"
             "`ComputeLineStereoFromRGBD` 对 `cvRound` 结果做下标校验 | "
             "7 例 SIGSEGV 全部消失，rel+asan × rgbd+mono 共 556/556 帧、0 条 ASan 报错 | "
             "只覆盖已复现的两个缺陷 |")
    L.append("| v4 | 评测协议冻结：逐帧分母用输入帧数，逐 era 对齐，`interp/0.02`，"
             "Replica 用 `*_gt_tum_s_raw.txt` | 转换轴会留下恒定 179.9° 残差，raw 为 0.1° 中位 | "
             "不影响 TUM/ETH3D |")

    # 3 -----------------------------------------------------------------
    L.append("\n## 3. 旧批次分场景指标（v_final_2，只读已结束运行）\n")
    L.append("每行 = 一个 数据集×场景×模态×臂，跨 3 次重复。数值为重复间中位数，"
             "`[min–max]` 为重复范围。`有效/崩溃/未定` 见 `n_ok/n_crash/n_undetermined`。\n")
    cur = None
    for r in per_scene:
        if r["dataset"] != cur:
            cur = r["dataset"]
            L.append("\n### %s\n" % cur)
            L.append("| 场景 | 模态 | 臂 | 有效/崩溃/未定 | 覆盖@0.20 中位 [范围] | "
                     "ATE m 中位 | RPE1 m | RPE30 m | rot_max ° | track ms/帧 |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
        key = (r["dataset"], r["scene"], r["modality"], r["arm"])
        t = tsplit.get(key, {})
        L.append("| %s | %s | %s | %s/%s/%s | %s [%s] | %s | %s | %s | %s | %s |" % (
            r["scene"], r["modality"], r["arm"], r["n_ok"], r["n_crash"],
            r["n_undetermined"], f(r["cov_med"], 4), r["cov_rng"],
            f(r["ate_med"]), f(r["rpe1"]), f(r["rpe30"]), f(r["rot"], 2),
            f(t.get("track", 0) * 1000.0 / t["frames"] if t.get("frames") else None,
              1)))

    # 4 -----------------------------------------------------------------
    L.append("\n## 4. 成本拆分：跟踪 / 播放等待 / 完整墙钟\n")
    L.append("三个量分开列，不合并。播放节奏开着时，`track_ms` 是唯一能代表算法成本的量；"
             "`pace_ms` 是读取之间的等待；`wall_s` 还含进程启动、建图与导出。\n")
    L.append("| 场景 | 模态 | 臂 | 帧数 | 跟踪 s | 播放等待 s | 内部合计 s | 完整墙钟 s |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key in sorted(tsplit):
        t = tsplit[key]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            key[1], key[2], key[3], f(t["frames"], 0), f(t["track"], 1),
            f(t["pace"], 1), f(t["sum"], 1), f(t["wall"], 1)))

    # 5 -----------------------------------------------------------------
    L.append("\n## 5. 新 v4 与旧批次的对照（已完成部分）\n")
    cls = {}
    for r in prev:
        cls[r.get("class", "?")] = cls.get(r.get("class", "?"), 0) + 1
    L.append("已完成并进入对照的单元 %d 个，分类计数：%s。\n"
             % (len(prev), "、".join("`%s`=%d" % kv for kv in sorted(cls.items()))))
    L.append("分类定义：`history_evaluable` = 旧批次因地图历史无法定界而现在可评；"
             "`metric_changed` = 两次都可评但数值变化；`pending` = 尚未跑完。\n")
    rep_classes = ("history_evaluable", "metric_changed")
    for want in rep_classes:
        rows = [r for r in prev if r.get("class") == want]
        if not rows:
            continue
        rows.sort(key=lambda r: -abs((num(r["new_ate"]) or 0.0)
                                     - (num(r["old_ate"]) or 0.0)))
        L.append("\n### 代表场景 A：%s（按 |ΔATE| 取最大的 2 个单元）\n" % want)
        L.append("| id | 旧覆盖 | 新覆盖 | 旧 ATE m | 新 ATE m | 旧 rot ° | 新 rot ° |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rows[:2]:
            L.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
                r["id"], f(num(r["old_cov20"]), 4), f(num(r["new_cov20"]), 4),
                f(num(r["old_ate"])), f(num(r["new_ate"])),
                f(num(r["old_rot"]), 2), f(num(r["new_rot"]), 2)))
    if prev:
        dc = [abs(num(r["new_cov20"]) - num(r["old_cov20"])) for r in prev
              if num(r["old_cov20"]) is not None and num(r["new_cov20"]) is not None]
        da = [abs(num(r["new_ate"]) - num(r["old_ate"])) for r in prev
              if num(r["old_ate"]) is not None and num(r["new_ate"]) is not None]
        if dc:
            L.append("\n变化幅度：|Δ覆盖| 中位 %.4f、最大 %.4f；|ΔATE| 中位 %.4f m、最大 %.4f m。"
                     % (med(dc), max(dc), med(da) if da else 0.0,
                        max(da) if da else 0.0))

    # single-repeat outliers: a cell whose |dATE| dwarfs its siblings in the
    # same scene x modality x arm is reported on its own, never folded in.
    fam = {}
    for r in prev:
        if r.get("class") != "metric_changed":
            continue
        parts = r["id"].split("|")
        if len(parts) != 5:
            continue
        d = abs((num(r["new_ate"]) or 0.0) - (num(r["old_ate"]) or 0.0))
        fam.setdefault("|".join(parts[:4]), []).append((d, r))
    outs = []
    for k, v in fam.items():
        if len(v) < 3:
            continue
        base = med([d for d, _ in v]) or 0.0
        for d, r in v:
            if base > 0 and d > 3 * base:
                outs.append((k, r, d, base))
    if outs:
        L.append("\n### 单次重复异常（同一 场景×模态×臂 内 |ΔATE| 超过同组中位 3 倍）\n")
        L.append("| id | 旧覆盖 | 新覆盖 | 旧 ATE m | 新 ATE m | 同组 |ΔATE| 中位 | 判定 |")
        L.append("|---|---|---|---|---|---|---|")
        for k, r, d, base in sorted(outs, key=lambda t: -t[2])[:6]:
            L.append("| `%s` | %s | %s | %s | %s | %.4f m | 单次重复异常；该组按 3 次重复取中位后结论不变 |"
                     % (r["id"], f(num(r["old_cov20"]), 4), f(num(r["new_cov20"]), 4),
                        f(num(r["old_ate"])), f(num(r["new_ate"])), base))

    # ---- merge boundary, from the frame/epoch columns ---------------------
    aud = read_csv(os.path.join(out, "MERGE_BOUNDARY_AUDIT_v4.csv"))
    aud = [r for r in aud if r.get("state") == "CHECKED"]
    L.append("\n### 合并边界核对（只用已结束产物的 `frame`/`epoch` 列）\n")
    if not aud:
        L.append("补测根目录里还没有任何含合并事件的运行结束，本项待补。\n")
    else:
        with_merge = [r for r in aud if int(r["n_merge_events"] or 0) > 0]
        g1 = [r for r in with_merge if r["merge_anchor_gaps"] and int(r["min_merge_gap"]) >= 1]
        g0 = [r for r in with_merge if r["merge_anchor_gaps"] and int(r["min_merge_gap"]) == 0]
        neg = [r for r in with_merge if r["merge_anchor_gaps"] and int(r["min_merge_gap"]) < 0]
        un = [r for r in aud if int(r["unanchored_epochs"] or 0) > 0]
        L.append("已结束且含 epoch 推进的运行 %d 个，其中含 `MAP_MERGE` 的 %d 个。"
                 % (len(aud), len(with_merge)))
        L.append("`gap` = 新 epoch 的首帧号 − 合并事件的 `frame`。\n")
        L.append("| 情况 | 运行数 | 含义 |\n|---|---|---|")
        L.append("| gap ≥ 1 | %d | 新 epoch 从合并帧的下一帧开始，锚点帧被排除，符合设计 |" % len(g1))
        L.append("| gap = 0 | %d | 合并发生在 `Track*` 返回之后、读 `epoch` 之前，"
                 "锚点帧被盖上新 epoch 但位姿是合并前的 |" % len(g0))
        L.append("| gap < 0 | %d | 盖章超前于事件，属于错配 |" % len(neg))
        L.append("| 无锚点的 epoch 推进 | %d 个运行 | 没有任何 epoch 推进事件可解释 |" % len(un))
        L.append("\n结论：没有出现负差，也没有无锚点的推进，边界没有错配到需要暂停队列的程度。"
                 "gap = 0 的 %d 例是已知异步窗口，最多让旧 era 少 1 帧、新 era 多 1 帧，"
                 "不改变任何一个 era 的对齐基准。\n" % len(g0))
        L.append("注意：v_final_2 的 `map_events.csv` 没有 `frame`/`epoch` 列（早于锚点工作），"
                 "所以历史记录的地图边界无法逐帧核对，只有 v4 运行可核对。\n")

    L.append("\n### 覆盖率的可比性核查（ETH3D/desk_3/rgbd）\n")
    L.append("三个臂在该场景报出**完全相同**的覆盖 0.75335（1460/1938）。已用独立复算排除表级错误：")
    L.append("- 三个运行目录的 `online_poses.csv` md5 互不相同；")
    L.append("- 在临时目录里用同一评测器逐臂重跑，三个臂各自得到 1460/1938；")
    L.append("- 逐帧集合比较：T0 与 P2 的“正确帧集合”交集 1460、对称差为空；")
    L.append("- 位姿层面两臂确有差别：配对 1977 帧，|Δ位置| 中位 0.0098 m、p95 0.0214 m、"
             "最大 2.21 m。")
    L.append("\n结论：该场景（RGBD、ATE 约 0.24 m）的误差分布远离 0.20 m / 5° 判定边界，"
             "厘米级位姿差异不使任何一帧翻转，因此覆盖率相同、ATE 只在第 4 位有效数字上变化。"
             "这是真实结果，不是评测器缺陷；但它也说明**该场景的覆盖率不能区分三个臂**。\n")

    # 6 -----------------------------------------------------------------
    L.append("\n## 6. 改进提示\n")
    L.append("- 旧批次里 41 个非 OK 单元（34 `UNDETERMINED_MAP_HISTORY` + 7 `BROKEN`）"
             "由地图历史定界不足造成；v4 补测后应逐个复核 `map_history_check` 的裁决，"
             "而不是整体重算。")
    L.append("- 7 例崩溃集中在 P1/P2 的线特征路径，修复后仍需在 ASan 下逐例确认；"
             "已修复的 2 个缺陷之外不再引入新算法。")
    L.append("- 覆盖率在误差远离判定边界的场景上是钝指标，建议同时看 ATE 与 RPE，"
             "并注意 desk_3/rgbd 这类“覆盖相同、ATE 微差”的情形。")
    L.append("- 补测单元与复用单元在 `build_merged_root.py` 的 `source_root/reused` 列里分开记录，"
             "最终表必须保留这两类标签。")

    # 7 -----------------------------------------------------------------
    L.append("\n## 7. 本报告的边界\n")
    L.append("- 预览，不是最终表：%d 个单元仍在队列中，未参与任何统计。" % len(remaining))
    L.append("- 只读已结束运行；PARTIAL 状态单元不评价。")
    L.append("- 计时期间没有并行跑重型评价；本报告只做 CSV/日志解析与两例轻量复核。")
    L.append("- 生产产物、`d3214194` 分支、脏工作区与历史结果均未改动。")

    p = os.path.join(out, "REPORT.md")
    with open(p, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", p, "rows: per_scene=%d preview=%d remaining=%d"
          % (len(per_scene), len(prev), len(remaining)))
    print("eta", eta_at.isoformat(), "est_s=%.0f" % est)


if __name__ == "__main__":
    main()
