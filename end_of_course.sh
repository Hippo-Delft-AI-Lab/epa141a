#!/bin/bash
# Remove all model answers from the student repo at the end of the course.
# Run this after the final assignment deadline has passed.
#
# Usage: ./end_of_course.sh

STAFF_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDENT_DIR="$(dirname "$STAFF_DIR")/epa141a-student"

if [ ! -d "$STUDENT_DIR/.git" ]; then
    echo "Student repo not found at $STUDENT_DIR"
    exit 1
fi

if [ ! -d "$STUDENT_DIR/model_answers_ema" ]; then
    echo "No model_answers_ema folder in student repo — nothing to remove."
    exit 0
fi

cd "$STUDENT_DIR"
rm -rf model_answers_ema
git add -A
git commit -m "Remove model answers at end of course"
git push
echo "Done — all model answers removed from student repo."
