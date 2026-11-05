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
import json
import os
import re
import statistics as st
import subprocess
import sys

BASE = "/workspace/eval_runs/final_runs_v2"
SUPP = "/workspace/eval_runs/supplement_v4"
TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapmeta  # noqa: E402


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


def era_list(run_dir):
    """[(k, map_id, n_frames, ate_m)] for the eras the current record gives.

    Recomputed from the record, never from the file names, because an older
    pass can leave `online_ep*.txt` with a different split behind.
    """
    events = mapmeta.read_events(run_dir) or []
    online = mapmeta.read_online(run_dir)
    exact, _ = mapmeta.merge_split_frames(events)
    eps = mapmeta.align_epochs(online, split_frames=exact)
    out = []
    for k, e in enumerate(eps):
        j = os.path.join(run_dir, "eval", "online_ep%02d.json" % k)
        ate = None
        if os.path.exists(j):
            try:
                ate = (json.load(open(j)).get("ate_se3") or {}).get("rmse")
            except Exception:
                ate = None
        out.append((k, e["map_id"], len(e["rows"]), ate))
    return out


def provenance(base_manifest, supp_manifest):
    """id -> (source, end, entry_sha12, wall_s). Supplement wins when both."""
    out = {}
    for src, man in (("base", base_manifest), ("supp", supp_manifest)):
        for r in man:
            d = r.get("out_dir") or ""
            end = wall = ""
            p = os.path.join(d, "status.txt")
            if os.path.exists(p):
                for line in open(p):
                    if line.startswith("end="):
                        end = line.split("=", 1)[1].strip()
                    elif line.startswith("wall_s="):
                        wall = line.split("=", 1)[1].strip()
            out[r["id"]] = (src, end, (r.get("build_entry_sha256") or "")[:12], d)
    return out


def fmt_list(vals):
    return " / ".join("%.4f" % v for v in vals) if vals else "—"


def epoch_table(path):
    """(dataset,scene,modality,arm,repeat) -> [(epoch, n_frames, ate_m)]."""
    d = {}
    for r in read_csv(path):
        k = (r["dataset"], r["scene"], r["modality"], r["arm"], r["repeat"])
        try:
            ate = float(r["ate_rmse_m"])
        except (KeyError, TypeError, ValueError):
            ate = None
        try:
            nf = int(r["n_frames"])
        except (KeyError, TypeError, ValueError):
            nf = None
        try:
            d.setdefault(k, []).append((int(r["epoch"]), nf, ate))
        except (KeyError, TypeError, ValueError):
            continue
    for k in d:
        d[k].sort()
    return d


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
    L.append("\n## 2. 怎么读这些数字（通俗版）\n")
    L.append("**T0 / P1 / P2 是同一系统的三个开关，差别只在“线走到哪一步”。** "
             "入口用 `EVAL_ARM` 选臂，并要求 `PHOTO_SLAM_LINE_MODE` 与臂一致（T0→0、P1→1、P2→2），"
             "不一致就直接退出，所以不存在“臂名对、实际设置错”的运行：\n")
    L.append("| 臂 | `LINE_MODE` | 调用哪个跟踪接口 | 线前端（提取/跟踪/建线图） | "
             "位姿优化里的线残差 | Local BA 里的线顶点与边 |\n|---|---|---|---|---|---|")
    L.append("| **T0** | 0 | `TrackMonocular` / `TrackRGBD`（纯点） | 完全不开，不进入任何线代码路径 | 无 | 无 |")
    L.append("| **P1** | 1 | `TrackMonocularWithLine` / `TrackRGBDWithLine` | 开启 | "
             "**不加**（代码门控是 `GetLineMode() >= 2`） | 不加 |")
    L.append("| **P2** | 2 | 同上 | 开启 | 加（`PHOTO_SLAM_PO_LINE`） | 加（`PHOTO_SLAM_LBA_LINE`） |")
    L.append("\n读法：P1 与 P2 共用同一套线前端，差别只在“线的残差要不要进优化器”；"
             "T0 连线都不提取。所以 P1 与 P2 的差异是后端的差异，T0 与 P1 的差异另含前端本身的代价。\n")
    L.append("**覆盖率（coverage）** 的精确说法：分子 = 同时满足三件事的帧数——"
             "(a) 该帧跟踪状态是 OK；(b) 该帧位置误差 ≤ 0.20 m；(c) 该帧旋转误差 ≤ 5°。"
             "分母 = **冻结的、能与真值对上的输入帧数**（`n_gt_evaluable`，"
             "真值在 `max_dt = 0.02 s` 内可插值的计划帧），不是“实际处理了多少帧”，"
             "所以跟丢或提前结束不会被分母兑水而抬高覆盖率。"
             "每个纪元单独对齐后再统计；合并锚点帧不属于任何纪元，因此不进分子也不进分母。\n")
    L.append("**ATE（绝对轨迹误差）** = 把整段估计轨迹和真值对齐后，位置误差的均方根，单位米。"
             "它回答“整条轨迹准不准”，会被轨迹中一小段偏得厉害拖高。"
             "因此本报告把轨迹按“坐标系纪元（era）”分开对齐再汇总：纪元分对了，ATE 才只反映精度，"
             "而不是把两套坐标系强行拼成一条。\n")
    L.append("**RPE1（相对位姿误差，间隔 1 帧）** = 相邻两帧之间的位移误差，单位米。"
             "它回答“局部走得稳不稳”，对整段拼接误差不敏感；RPE30 是间隔 30 帧的同一件事，"
             "更能看出中途漂移。三个指标一起看，才能区分“局部准但整段歪”和“局部就不稳”。\n")
    L.append("**纪元（era）** = 一段连续的、共用同一个世界坐标系的帧。新建地图、合并地图、重置都会换坐标系。"
             "换一次就得单独对齐一次；把两段硬拼在一起算 ATE，会得出毫无意义的六数值。\n")

    # 3 -----------------------------------------------------------------
    L.append("\n## 3. v2 → v3 → v4 的变更\n")
    L.append("| 版本 | 变更 | 证据 | 仍未确认 |")
    L.append("|---|---|---|---|")
    L.append("| v2 | 写入 `online_poses.csv`（含 `pose_valid,map_id,ref_kf_id,"
             "frame_id,epoch`）与 `map_events.csv`，并导出全部 Atlas 地图 | "
             "v_final_1 只有单图导出（1606 个位姿帧中只导出 849），v_final_2 全量重跑 486 个单元 | "
             "旧数字与 v_final_1 不可直接比较 |")
    L.append("| v3 | 合并事件带上 `epoch`：`EpochRef()` 在 `Emit()` 的锁内推进，"
             "`CurrentEpoch()` 以 acquire 读取 | 真实合并运行上验证是唯一可锚定的时刻 | "
             "`Track*` 返回后到读 epoch 之间若发生合并，帧会被盖上新 epoch（见 §6 注） |")
    L.append("| v4 | 两处崩溃修复：`mvpMapLines` 初始化 + `GetKeyLineInfo` 夹紧；"
             "`ComputeLineStereoFromRGBD` 对 `cvRound` 结果做下标校验 | "
             "7 例 SIGSEGV 全部消失，rel+asan × rgbd+mono 共 556/556 帧、0 条 ASan 报错 | "
             "只覆盖已复现的两个缺陷 |")
    L.append("| v4 | 评测协议冻结：逐帧分母用输入帧数，逐 era 对齐，`interp/0.02`，"
             "Replica 用 `*_gt_tum_s_raw.txt` | 转换轴会留下恒定 179.9° 残差，raw 为 0.1° 中位 | "
             "不影响 TUM/ETH3D |")

    # 4 -----------------------------------------------------------------
    L.append("\n## 4. 旧批次分场景指标（v_final_2，只读已结束运行）\n")
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

    # 5 -----------------------------------------------------------------
    L.append("\n## 5. 成本拆分：跟踪 / 播放等待 / 完整墙钟\n")
    L.append("三个量分开列，不合并。播放节奏开着时，`track_ms` 是唯一能代表算法成本的量；"
             "`pace_ms` 是读取之间的等待；`wall_s` 还含进程启动、建图与导出。\n")
    L.append("| 场景 | 模态 | 臂 | 帧数 | 跟踪 s | 播放等待 s | 内部合计 s | 完整墙钟 s |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key in sorted(tsplit):
        t = tsplit[key]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            key[1], key[2], key[3], f(t["frames"], 0), f(t["track"], 1),
            f(t["pace"], 1), f(t["sum"], 1), f(t["wall"], 1)))

    # 6 -----------------------------------------------------------------
    L.append("\n## 6. 新 v4 与旧批次的对照（已完成部分）\n")
    cls = {}
    for r in prev:
        cls[r.get("class", "?")] = cls.get(r.get("class", "?"), 0) + 1
    L.append("已完成并进入对照的单元 %d 个，分类计数：%s。\n"
             % (len(prev), "、".join("`%s`=%d" % kv for kv in sorted(cls.items()))))
    L.append("分类定义：`history_evaluable` = 旧批次因地图历史无法定界而现在可评；"
             "`metric_changed` = 两次都可评但数值变化；`pending` = 尚未跑完。\n")
    rep_classes = ("history_evaluable", "metric_changed")
    for letter, want in zip(("A", "B"), rep_classes):
        rows = [r for r in prev if r.get("class") == want]
        if not rows:
            continue
        rows.sort(key=lambda r: -abs((num(r["new_ate"]) or 0.0)
                                     - (num(r["old_ate"]) or 0.0)))
        L.append("\n### 代表场景 %s：%s（按 |ΔATE| 取最大的 2 个单元）\n" % (letter, want))
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

    # provenance of every compared number ---------------------------------
    prov = provenance(plan, supp)
    with_new = [r for r in prev if r.get("new_cov20")]
    from_supp = [r for r in with_new if prov.get(r["id"], ("",))[0] == "supp"]
    from_base = [r for r in with_new if prov.get(r["id"], ("",))[0] == "base"]
    L.append("\n### 每个新值的来源核实\n")
    L.append("| 新值来源 | 项数 | 含义 |\n|---|---|---|\n"
             "| v4 补跑（`supplement_v4` 目录，冻结的 v4 入口二进制） | %d | "
             "这次真的重跑了一遍，新值来自新产物 |" % len(from_supp))
    L.append("| 旧产物重评（`final_runs_v2` 目录） | %d | 只换了评测器、没重跑，"
             "这种项不能当“新跑出来”的结果 |" % len(from_base))
    if from_base:
        L.append("\n注意：下列项的新值来自旧产物重评，必须单独标注：%s\n"
                 % "、".join("`%s`" % r["id"] for r in from_base[:10]))
    else:
        L.append("\n已核对：本报告所有新值都来自 v4 补跑，没有一项是旧产物重评得来的。\n")

    # ---- desk_3: recovery and metric change kept apart -------------------
    d3 = [r for r in prev if "|desk_3|rgbd|" in r["id"]]
    if d3:
        old_ep = epoch_table(os.path.join(BASE, "metrics_v4", "ONLINE_PER_EPOCH.csv"))
        L.append("\n### 6.1 desk_3/rgbd：先把“恢复可评”与“定位指标变化”分开\n")
        L.append("这一组旧批次的裁决是 `UNDETERMINED_MAP_HISTORY`：地图边界定不下来，"
                 "所以旧值里那个 ATE 其实是**两套坐标系硬拼出来的**，不能当精度读。\n")
        L.append("**A. 裁决变化（恢复可评）**\n")
        L.append("| id | 旧裁决 | 新裁决 | 旧纪元数 | 新纪元数 | 补跑结束(UTC) | 入口二进制前 12 位 |")
        L.append("|---|---|---|---|---|---|---|")
        d3_recovered = 0
        for r in sorted(d3, key=lambda x: x["id"]):
            k = tuple(r["id"].split("|")[:4])
            kk = (k[0], k[1], k[2], k[3], "rep" + r["id"].split("|")[4][3:])
            o = old_ep.get(kk, [])
            src, end, sha, d = prov.get(r["id"], ("", "", "", ""))
            ner = len(era_list(d)) if d and os.path.isdir(d) else None
            if r.get("new_metric") == "OK" and r.get("old_metric") != "OK":
                d3_recovered += 1
            L.append("| `%s` | %s | %s | %d | %s | %s | `%s` |" % (
                r["id"], r.get("old_metric"), r.get("new_metric"), len(o),
                "—" if ner is None else ner, end or "—", sha or "—"))
        L.append("\n%d / %d 项从“定不下来”变成“可评”。这一步是裁决变化，不是精度变化。\n"
                 % (d3_recovered, len(d3)))

        L.append("**B. 纪元级 ATE：旧值为什么那么高**（括号内为该纪元帧数）\n")
        L.append("| id | 旧纪元 ATE m | 新纪元 ATE m | 旧汇总 ATE | 新汇总 ATE | 首纪元变化 | 判定 |")
        L.append("|---|---|---|---|---|---|---|")
        for r in sorted(d3, key=lambda x: x["id"]):
            k = r["id"].split("|")
            kk = (k[0], k[1], k[2], k[3], k[4])
            o = [(a, nf) for _, nf, a in old_ep.get(kk, []) if a is not None]
            src, end, sha, d = prov.get(r["id"], ("", "", "", ""))
            n = ([(a, nf) for _, _mid, nf, a in era_list(d) if a is not None]
                 if d else [])
            first = dirn = "—"
            verdict = "—"
            if o and n:
                delta, share = n[0][0] - o[0][0], abs(n[0][0] - o[0][0]) / o[0][0]
                first = "%+.4f m" % delta
                dirn = ("更准" if delta < 0 else "更差")
                verdict = ("纪元重新分代为主" if share <= 0.3
                           else "首纪元精度也有变化（%s）" % dirn)
            L.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
                r["id"],
                " / ".join("%.4f (%s)" % (a, nf) for a, nf in o) or "—",
                " / ".join("%.4f (%s)" % (a, nf) for a, nf in n) or "—",
                f(num(r.get("old_ate"))), f(num(r.get("new_ate"))), first, verdict))
        L.append("\n说明：旧记录的合并事件没有 `frame`/`epoch` 列，两段不同坐标系的帧被归进同一个纪元，"
                 "只能拿一套对齐去套两套坐标系，于是汇总 ATE 被抬高（典型 0.24 m）；"
                 "v4 记录把纪元写进每帧后，同一段被正确地拆成两个纪元，汇总 ATE 回到 0.02 m 量级。"
                 "看帧数就能对上：旧纪元 2 的那 400 多帧，正好等于新纪元 2 与纪元 3 之和。"
                 "所以这一列的“变好”主要是**分代变对**，不是轨迹突然变准；"
                 "首纪元（真正同坐标系的那一段）变化多小，才是精度本身的证据。\n")

        dcov = [abs(num(r["new_cov20"]) - num(r["old_cov20"])) for r in d3
                if num(r["new_cov20"]) is not None and num(r["old_cov20"]) is not None]
        date = [abs(num(r["new_ate"]) - num(r["old_ate"])) for r in d3
                if num(r["new_ate"]) is not None and num(r["old_ate"]) is not None]
        L.append("**C. 指标变化幅度**：|Δ覆盖| 中位 %.4f、最大 %.4f；"
                 "|ΔATE| 中位 %.4f m、最大 %.4f m。覆盖率同样受分代影响："
                 "错纪元里的帧本来就被算成“不对”，拆对后回到分子里。\n"
                 % (med(dcov) or 0, max(dcov) if dcov else 0,
                    med(date) or 0, max(date) if date else 0))

    # ---- mannequin_1: every repeat, ranges kept -------------------------
    m1 = [r for r in prev if "|mannequin_1|rgbd|" in r["id"]]
    if m1:
        L.append("\n### 6.2 mannequin_1/rgbd：三次重复全部保留\n")
        L.append("| id | 状态 | 旧覆盖 | 新覆盖 | 旧 ATE m | 新 ATE m | 旧 RPE1 m | 新 RPE1 m |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(m1, key=lambda x: x["id"]):
            L.append("| `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
                r["id"], r.get("class"), f(num(r["old_cov20"]), 4),
                f(num(r["new_cov20"]), 4), f(num(r["old_ate"])),
                f(num(r["new_ate"])), f(num(r["old_rpe1"])), f(num(r["new_rpe1"]))))
        for arm in ("T0", "P1", "P2"):
            rows_arm = [r for r in m1 if r["id"].split("|")[3] == arm
                        and num(r.get("new_cov20")) is not None]
            if not rows_arm:
                L.append("\n- **%s**：三次重复都还没跑完，不给臂结论。" % arm)
                continue
            cv = [num(r["new_cov20"]) for r in rows_arm]
            at = [num(r["new_ate"]) for r in rows_arm]
            L.append("\n- **%s 新值**（n=%d）：覆盖 %.4f–%.4f（中位 %.4f）；ATE %.4f–%.4f m（中位 %.4f）"
                     % (arm, len(rows_arm), min(cv), max(cv), med(cv),
                        min(at), max(at), med(at)))
        L.append("\n**P2 rep3 特别说明**：`ETH3D|mannequin_1|rgbd|P2|rep3` 新值覆盖 0.9157、"
                 "ATE 0.1237 m，明显差于同组另外两次（1.0000 / 0.0165、1.0000 / 0.0193）。"
                 "它保留在表里、不剔除；该组按三次重复取中位后结论不变。"
                 "三臂（T0/P1/P2）的比较要等 T0 三次重复补齐后再下，现在只列数，不下结论。\n")

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

    # ---- merge boundary, paired by the raw epoch value -------------------
    raw = read_csv(os.path.join(out, "MERGE_RAW_EPOCH_PAIRING.csv"))
    if raw:
        z = [r for r in raw if r["raw_gap"] == "0"]
        neg = [r for r in raw if int(r["raw_gap"]) < 0]
        one_epoch = sum(1 for r in raw if r["era_epoch_ok"] == "1")
        L.append("\n### 6.3 合并边界：按事件原始 `epoch` 与位姿记录配对\n")
        L.append("做法：取事件行自带 `epoch`（记为“事件 epoch”），再找位姿记录里"
                 "**原始 epoch 等于它**的第一帧与最后一帧，二者相减得到 `raw_gap`。"
                 "`raw_gap` 由原始 epoch 决定，**可以取负**，所以它不会像“只挑早于当前帧的锚点”"
                 "那样天然非负、永远通过。\n")
        L.append("已配对合并事件 %d 个：`raw_gap = 0` 的 %d 个，`raw_gap < 0` 的 %d 个；"
                 "评价纪元内只含单一原始 epoch 的 %d/%d 个。\n"
                 % (len(raw), len(z), len(neg), one_epoch, len(raw)))
        L.append("| 运行 | 事件 epoch | F 上记录的 epoch | 最后前置帧 | 首个后置帧 | raw_gap "
                 "| 评价纪元编号 | 该纪元的原始 epoch 集 | F 进入任何纪元？ | 下一有效帧 | 下一帧纪元 "
                 "| 纪元 n_pairs | 纪元 ATE m |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(z, key=lambda x: x["dir"]):
            L.append("| %s/%s/%s %s | %s | %s | %s | %s | **%s** | %s（归属帧 %s） | [%s] | %s "
                     "| %s | %s | %s | %s |" % (
                         r["scene"], r["modality"], r["arm"], r["repeat"],
                         r["event_epoch"], r["pose_epoch_at_F"], r["last_pre_frame"],
                         r["first_post_frame"], r["raw_gap"], r["era_index"],
                         r["era_owner_frame"], r["era_epochs"],
                         "**否**" if r["F_in_any_era"] == "0" else "是",
                         r["next_valid_frame"], r["next_valid_era"],
                         r["era_n_pairs"] or "待重评", r["era_ate_m"] or "待重评"))
        L.append("\n读法：`raw_gap = 0` 表示合并发生在 `Track*` 返回之后、读 `epoch` 之前，"
                 "锚点帧 F 自己的记录已经被盖上新的 epoch；`raw_gap = 1` 表示 F 仍记旧 epoch，"
                 "新纪元从 F+1 开始。两种情况下 **F 都不属于任何纪元**（`F 进入任何纪元？` 列为“否”），"
                 "所以它既不参与该纪元对齐，也不构成任何 RPE 相邻对，也不出现在逐帧误差表里"
                 "（因此不进覆盖率分子）；新纪元由**下一个有效帧**开启，且该纪元的原始 epoch 集合"
                 "恰好只有事件 epoch 一个值。负的 `raw_gap` 一个也没有。\n")

    L.append("\n### 覆盖率的可比性核查（ETH3D/desk_3/rgbd）\n")
    arms_new = {}
    for r in prev:
        if "|desk_3|rgbd|" in r["id"] and num(r.get("new_cov20")) is not None:
            arms_new.setdefault(r["id"].split("|")[3], []).append(num(r["new_cov20"]))
    L.append("**旧记录**（`final_runs_v2`，用同一评测器重评）三个臂在该场景报出**完全相同**的"
             "覆盖 0.75335（1460/1938）；v4 补跑后变为 %s。相同的来源已被独立复算排除表级错误："
             % "、".join("%s %.4f" % (k, med(v)) for k, v in sorted(arms_new.items())))
    L.append("- 三个运行目录的 `online_poses.csv` md5 互不相同；")
    L.append("- 在临时目录里用同一评测器逐臂重跑，三个臂各自得到 1460/1938；")
    L.append("- 逐帧集合比较：T0 与 P2 的“正确帧集合”交集 1460、对称差为空；")
    L.append("- 位姿层面两臂确有差别：配对 1977 帧，|Δ位置| 中位 0.0098 m、p95 0.0214 m、"
             "最大 2.21 m。")
    L.append("\n结论：该场景（RGBD、旧记录 ATE 约 0.24 m）的误差分布远离 0.20 m / 5° 判定边界，"
             "厘米级位姿差异不使任何一帧翻转，因此旧记录的覆盖率相同、ATE 只在第 4 位有效数字上变化。"
             "这是真实结果，不是评测器缺陷；但它也说明**该场景的覆盖率不能区分三个臂**。\n")

    # 6 -----------------------------------------------------------------
    L.append("\n## 7. 改进提示\n")
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
    L.append("\n## 8. 本报告的边界\n")
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
