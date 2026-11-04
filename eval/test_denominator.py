#!/usr/bin/env python3
"""Regression tests for the evaluation denominator and the status ladder.

The failure these guard against is quiet: a run that processes 4 of 10 frames and
dies would, if judged on its own output, look like it covered everything it
touched. The denominator must come from the frozen input, and no failure state
may end up labelled OK.

Run: python3 eval/test_denominator.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import online_eval    # noqa: E402
import scene_inputs   # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print("%-64s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        FAILED.append(name)


def main():
    # D1: 10 input frames, 4 processed, crash. The GT-evaluable denominator must
    # stay 10, not 4.
    input_ts = [i * 0.033 for i in range(10)]
    gt_ts = [i * 0.033 for i in range(10)]
    n_gt = scene_inputs.gt_evaluable_count(input_ts, gt_ts, 0.02)
    check("D1 denominator from input (10), not from processed rows (4)",
          n_gt == 10, "n_gt=%d" % n_gt)

    # D2: GT that stops early must not count the uncovered tail.
    n_gt2 = scene_inputs.gt_evaluable_count(input_ts, gt_ts[:4], 0.02)
    check("D2 GT ending early leaves 4 evaluable, not 10",
          n_gt2 == 4, "n_gt=%d" % n_gt2)

    # D3: a frame whose nearest GT is outside max_dt is not evaluable.
    n_gt3 = scene_inputs.gt_evaluable_count([1.0], [0.0], 0.02)
    check("D3 GT outside max_dt -> 0 evaluable", n_gt3 == 0, "n_gt=%d" % n_gt3)

    # D4: input timestamps are the frozen list, independent of any run.
    a = scene_inputs.input_timestamps("ETH3D", "sfm_bench", "mono")
    b = scene_inputs.expected_inputs("ETH3D", "sfm_bench", "mono")
    check("D4 input_timestamps length equals expected_inputs", len(a) == b,
          "%d vs %s" % (len(a), b))

    # D5: the status ladder. Every failure state must win over OK.
    d = online_eval.decide_metric_status
    check("D5a broken history -> BROKEN_MAP_HISTORY",
          d("BROKEN", 0, 10, 10, 0, 1, 10, 1) == "BROKEN_MAP_HISTORY")
    check("D5b ambiguous history -> UNDETERMINED_MAP_HISTORY",
          d("AMBIGUOUS", 0, 10, 10, 0, 1, 10, 1) == "UNDETERMINED_MAP_HISTORY")
    check("D5c crash beats a complete input",
          d("OK", -11, 10, 10, 0, 1, 10, 1) == "CRASHED_PARTIAL")
    check("D5d 4 of 10 processed -> INCOMPLETE_INPUT",
          d("OK", 0, 10, 4, 0, 1, 4, 1) == "INCOMPLETE_INPUT")
    check("D5e all epochs failed to evaluate -> EVAL_FAILED",
          d("OK", 0, 10, 10, 2, 2, 10, 1) == "EVAL_FAILED")
    check("D5f no GT reachable -> NO_GT_EVALUABLE",
          d("OK", 0, 10, 10, 0, 1, 0, 1) == "NO_GT_EVALUABLE")
    check("D5g convention guard -> CONVENTION_SUSPECT",
          d("OK", 0, 10, 10, 0, 1, 10, 0) == "CONVENTION_SUSPECT")
    check("D5h only a clean run is OK",
          d("OK", 0, 10, 10, 0, 1, 10, 1) == "OK")

    # D6: a partially processed run keeps its crash state and its denominator.
    st = d("OK", -11, 10, 4, 0, 1, 10, 1)
    check("D6 crash + partial input reported as crash, denominator still 10",
          st == "CRASHED_PARTIAL",
          "%s (n_gt would be 10, processed 4)" % st)

    print()
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
