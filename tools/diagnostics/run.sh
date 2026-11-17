#!/bin/zsh
# Usage, from the repository root: zsh tools/diagnostics/run.sh <script.py> <transfer> [<transfer> ...]
# Runs seeds 1-3 one after another; the memory guard stops a run above 12 GB.
script=$1; shift
for sd in 1 2 3; do
  PYTHONPATH=src .venv/bin/python tools/diagnostics/$script $sd "$@" & P=$!
  zsh tools/diagnostics/memory_guard.sh $P 12 & wait $P
done
