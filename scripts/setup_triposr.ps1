# Clones TripoSR into vendor/ at the commit it is tested with, and installs the ai extra (torch from PyTorch's
# CUDA 12.6 build, see pyproject.toml). Run from the repo root.
# Local TripoSR needs transformers below 5, and uv.lock installs transformers 5 (TrOCR needs it) whatever the
# extras: in the locked environment hf3d uses TripoSR's Hugging Face Space. The checkout is for a separate
# environment with transformers below 5.
$ErrorActionPreference = "Stop"
$Commit = "107cefdc244c39106fa830359024f6a2f1c78871"
if (-not (Test-Path vendor/TripoSR)) { git clone https://github.com/VAST-AI-Research/TripoSR vendor/TripoSR }
git -C vendor/TripoSR fetch --depth 1 origin $Commit  # an older shallow clone may not hold it
if ($LASTEXITCODE -ne 0) { throw "could not fetch TripoSR $Commit" }
git -C vendor/TripoSR checkout --quiet $Commit
if ($LASTEXITCODE -ne 0) { throw "could not check out TripoSR $Commit" }
uv sync --extra ai
uv run python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
