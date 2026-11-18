#!/usr/bin/env bash
# Machine-readable version inventory for Photo-SLAM-L.
# Prints: branches with push status, worktrees, tags, on-disk experiment dirs.
# Read-only: never modifies the repository.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
CODE=$(dirname "$REPO")

echo "== repository =="
git -C "$REPO" remote -v | head -2
echo "head: $(git -C "$REPO" rev-parse --short HEAD) $(git -C "$REPO" log -1 --format='%s')"
echo

echo "== branches (UPLOADED = tip commit exists on some remote ref) =="
printf "%-40s %-12s %-10s %-8s %s\n" BRANCH HEAD UPLOADED UNPUSHED SUBJECT
for b in $(git -C "$REPO" for-each-ref --format='%(refname:short)' refs/heads/); do
    h=$(git -C "$REPO" rev-parse --short "$b")
    if [ -n "$(git -C "$REPO" branch -r --contains "$b" 2>/dev/null | grep -v HEAD)" ]; then
        on=yes
    else
        on=NO
    fi
    n=$(git -C "$REPO" rev-list --count "$b" --not --remotes)
    s=$(git -C "$REPO" log -1 --format='%s' "$b")
    printf "%-40s %-12s %-10s %-8s %s\n" "$b" "$h" "$on" "$n" "$s"
done
echo

echo "== worktrees =="
git -C "$REPO" worktree list
echo

echo "== tags not on GitHub =="
for t in $(git -C "$REPO" tag); do
    git -C "$REPO" ls-remote --tags origin "refs/tags/$t" 2>/dev/null | grep -q . || echo "  $t"
done
echo

echo "== on-disk experiment directories under $CODE =="
printf "%-44s %8s %5s %5s %5s\n" DIR MB GIT BUILD SRC
for d in "$CODE"/*/; do
    d=${d%/}
    sz=$(du -sm "$d" 2>/dev/null | cut -f1)
    g=no; [ -e "$d/.git" ] && g=yes
    b=no; { [ -f "$d/build/CMakeCache.txt" ] || [ -f "$d/CMakeCache.txt" ]; } && b=yes
    s=no; { [ -d "$d/src" ] && [ -d "$d/ORB-SLAM3" ]; } && s=yes
    case "$(basename "$d")" in
        PSL*|Photo-SLAM*|Photoslam_results|backup|report) ;;
        *) continue ;;
    esac
    printf "%-44s %8s %5s %5s %5s\n" "$(basename "$d")" "$sz" "$g" "$b" "$s"
done
