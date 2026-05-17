$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $PSScriptRoot "run_12vm_bc_cg_benchmark.py"
$analysisDir = Join-Path $PSScriptRoot "results\analysis\chance_2sp_toy_12vm_bc_cg_t10800"

New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null

$stdoutLog = Join-Path $analysisDir "launcher_stdout.log"
$stderrLog = Join-Path $analysisDir "launcher_stderr.log"

$args = @(
    $scriptPath,
    "--time-limit", "10800",
    "--seed-time-limit", "1200",
    "--max-workers", "4",
    "--threads", "8",
    "--strategy-threads", "4"
)

$process = Start-Process -FilePath $pythonExe `
    -ArgumentList $args `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Write-Output ("Started benchmark process: PID={0}" -f $process.Id)
Write-Output ("stdout: {0}" -f $stdoutLog)
Write-Output ("stderr: {0}" -f $stderrLog)
