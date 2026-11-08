#!/usr/bin/env python3
"""Create an L_str-OFF Gaussian config from an official one.

Only the three effective structural-loss weights are changed (to 0.0).
Everything else (line sampling, densification, keyframe policy, ...) is
byte-identical, so Full and OFF share one binary and one behaviour except for
the structural-loss term.
"""
import re
import sys

KEYS = (
    "weight_line_coherence",
    "weight_line_shape_ecc",
    "weight_line_shape_ori",
)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: make_off_cfg.py <official.yaml> <off.yaml>", file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src, encoding="utf-8").read()
    for key in KEYS:
        pat = re.compile(rf"^(\s*Optimization\.{key}:\s*)\S+", re.MULTILINE)
        if len(pat.findall(text)) != 1:
            print(f"ERROR: expected exactly one Optimization.{key} in {src}", file=sys.stderr)
            return 1
        text = pat.sub(r"\g<1>0.0", text)
    open(dst, "w", encoding="utf-8").write(text)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
