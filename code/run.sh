#!/bin/bash
apptainer exec --nv pytorch.sif ~/.local/bin/uv run "$@"
