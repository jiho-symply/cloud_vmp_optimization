$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "run_12vm_branch_cut_benchmark.py"
$analysisDir = Join-Path $PSScriptRoot "results\analysis\chance_2sp_toy_12vm_branch_cut_round2_t10800"

New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null

& $python $script `
  --time-limit 10800 `
  --max-workers 4 `
  --threads 8 2>&1 | Tee-Object -FilePath (Join-Path $analysisDir "launcher_stdout.log")
