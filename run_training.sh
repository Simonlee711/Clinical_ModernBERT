#!/bin/bash
# This script launches the pre-training pipeline.
# Ensure that the virtual environment is activated before running.

set -e
python -m src.main "$@"
