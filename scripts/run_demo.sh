#!/usr/bin/env bash
set -e
python -m src.main --config configs/dev.yaml --source "${1:-0}"
