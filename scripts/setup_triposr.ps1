# Clones TripoSR into vendor/ at the commit it is tested with, and installs the ai extra (torch from PyTorch's
# CUDA 12.6 build, see pyproject.toml). Run from the repo root.
# Local TripoSR needs transformers below 5; the trocr extra needs 5 or newer. With both, hf3d falls back to the
# Hugging Face Space, which works either way.
$ErrorActionPreference = "Stop"
$Commit = "107cefdc244c39106fa830359024f6a2f1c78871"
if (-not (Test-Path vendor/TripoSR)) { git clone https://github.com/VAST-AI-Research/TripoSR vendor/TripoSR }
git -C vendor/TripoSR checkout --quiet $Commit
uv sync --extra ai
uv run python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
