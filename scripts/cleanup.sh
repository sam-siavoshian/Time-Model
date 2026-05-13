#!/usr/bin/env bash
# Clean up cruft before commits or preflight.
# Removes:
#  - __pycache__/ directories
#  - .pyc files
#  - .DS_Store
#  - logs older than 7 days
#  - intermediate checkpoints not in checkpoints/keep/
#  - tmp scratch in /tmp/p0_*.pt /tmp/p1_*.pt

set -eu

ROOT="/Users/samsiavoshian/Desktop/Coding Stuff/Time-Model"
cd "$ROOT"

echo "Cleaning Python cruft..."
find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -not -path './.venv/*' -delete 2>/dev/null || true
find . -type f -name '.DS_Store' -delete 2>/dev/null || true

echo "Cleaning stale logs (>7 days)..."
find logs -type f -mtime +7 -delete 2>/dev/null || true

echo "Cleaning intermediate checkpoints not in keep/..."
if [ -d checkpoints ]; then
    find checkpoints -type f -name '*_step*.pt' -not -path 'checkpoints/keep/*' -delete 2>/dev/null || true
fi

echo "Cleaning /tmp scratch..."
rm -f /tmp/p0_*.pt /tmp/p1_*.pt 2>/dev/null || true

echo "Cleaning empty dirs in checkpoints/ logs/ reports/..."
find checkpoints logs reports -type d -empty -delete 2>/dev/null || true

echo ""
echo "Disk usage after cleanup:"
du -sh data data/tokenized data_gen model scripts logs reports checkpoints 2>/dev/null || true

echo ""
echo "Cleanup done."
