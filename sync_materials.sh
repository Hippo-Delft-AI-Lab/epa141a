#!/bin/bash
# Sync assignment and lab instructions from staff repo to student repo.
# Run this after updating assignments_ema/ or labs/ in the staff repo.
#
# Usage: ./sync_materials.sh

STAFF_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDENT_DIR="$(dirname "$STAFF_DIR")/epa141a-student"

if [ ! -d "$STUDENT_DIR/.git" ]; then
    echo "Student repo not found at $STUDENT_DIR"
    echo "Clone it first: gh repo clone Hippo-Delft-AI-Lab/epa141a $STUDENT_DIR"
    exit 1
fi

rsync -av --delete "$STAFF_DIR/assignments_ema/" "$STUDENT_DIR/assignments_ema/"
rsync -av --delete "$STAFF_DIR/labs/" "$STUDENT_DIR/labs/"

cd "$STUDENT_DIR"
git add -A

if git diff --cached --quiet; then
    echo "No changes to sync."
else
    git commit -m "Sync assignment and lab instructions from staff repo"
    git push
    echo "Done — student repo updated."
fi
