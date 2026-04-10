#!/bin/bash
# Release the model answer for a specific assignment to the student repo.
# Run this after the assignment deadline.
#
# Usage: ./release_answer.sh 01

if [ -z "$1" ]; then
    echo "Usage: $0 <assignment_number> (e.g. 01)"
    exit 1
fi

ASSIGNMENT=$1
STAFF_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDENT_DIR="$(dirname "$STAFF_DIR")/epa141a-student"

if [ ! -d "$STUDENT_DIR/.git" ]; then
    echo "Student repo not found at $STUDENT_DIR"
    exit 1
fi

FILES=$(find "$STAFF_DIR/model_answers_ema" -name "assignment_${ASSIGNMENT}_*")

if [ -z "$FILES" ]; then
    echo "No model answers found for assignment $ASSIGNMENT in model_answers_ema/"
    exit 1
fi

mkdir -p "$STUDENT_DIR/model_answers_ema"

for f in $FILES; do
    cp "$f" "$STUDENT_DIR/model_answers_ema/"
    echo "Copied: $(basename $f)"
done

cd "$STUDENT_DIR"
git add -A
git commit -m "Release model answer for assignment $ASSIGNMENT"
git push
echo "Done — assignment $ASSIGNMENT model answer released to students."
