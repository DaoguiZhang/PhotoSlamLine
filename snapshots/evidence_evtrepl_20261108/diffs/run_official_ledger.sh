#!/usr/bin/env bash
# Complete official-test ledger.
#
# For each of the two official trees (O_raw = upstream + build compatibility,
# O_core = O_raw + the public engineering fixes) this runs every case in its own
# process, REPS times, and records:
#   * the library that was actually mapped into the process (/proc/self/maps)
#   * its sha256, so a result can never be attributed to the wrong build
#   * the exit code (139 = SIGSEGV, 134 = abort/uncaught exception, 1 = FAIL)
#   * the complete stdout+stderr, one file per (tree, rep, case)
#
# usage: run_official_ledger.sh [REPS]
set -u
REPS="${1:-3}"
cd /workspace/code/PSL-review || exit 1
OUT=ledger
mkdir -p "$OUT/raw"
: > "$OUT/ledger.csv"
echo "tree,rep,case,exit,result,lib_path,lib_sha256_16,notes" >> "$OUT/ledger.csv"

for T in O_raw O_core; do
  BIN=bin_$T
  [ -x "$BIN" ] || { echo "missing $BIN" >&2; continue; }
  for rep in $(seq 1 "$REPS"); do
    for c in 1 2 3 4 5 6 7 8; do
      log="$OUT/raw/${T}_rep${rep}_case${c}.log"
      env -u LD_LIBRARY_PATH ./"$BIN" --case="$c" > "$log" 2>&1
      rc=$?
      libpath=$(sed -n 's/^\[lib\] mapped=\([^ ]*\).*/\1/p' "$log" | head -1)
      sha=""
      [ -n "$libpath" ] && [ -f "$libpath" ] && sha=$(sha256sum "$libpath" | cut -c1-16)
      result=$(grep -E "^case $c: (OK|FAILED)" "$log" | tail -1 | sed 's/^case [0-9]*: //')
      [ -z "$result" ] && result="NO_RESULT"
      note=""
      [ "$rc" -eq 139 ] && note="SIGSEGV"
      [ "$rc" -eq 134 ] && note="abort(exception)"
      echo "$T,$rep,$c,$rc,$result,$libpath,$sha,$note" >> "$OUT/ledger.csv"
    done
  done
done
echo "ledger: $OUT/ledger.csv"
awk -F, 'NR>1{print $1, $4, $5}' "$OUT/ledger.csv" | sort | uniq -c
