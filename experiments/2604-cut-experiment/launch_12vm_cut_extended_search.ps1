param(
    [int]$ScreenTimeLimit = 1800,
    [int]$FinalTimeLimit = 7200,
    [int]$MaxWorkers = 4,
    [int]$Threads = 8,
    [int]$Finalists = 8
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $repoRoot
$analysisDir = Join-Path $PSScriptRoot "results\\analysis\\cut_12vm_od_sp_bj_cap8_avg20_extended_search"
$stdoutLog = Join-Path $analysisDir "launcher_stdout.log"
$stderrLog = Join-Path $analysisDir "launcher_stderr.log"
$pythonExe = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$scriptPath = Join-Path $PSScriptRoot "run_12vm_cut_extended_search.py"

New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null
if (Test-Path $stdoutLog) { Remove-Item $stdoutLog -Force }
if (Test-Path $stderrLog) { Remove-Item $stderrLog -Force }

$arguments = @(
    $scriptPath,
    "--screen-time-limit", $ScreenTimeLimit,
    "--final-time-limit", $FinalTimeLimit,
    "--max-workers", $MaxWorkers,
    "--threads", $Threads,
    "--finalists", $Finalists
)

$process = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Write-Output "pid=$($process.Id)"
Write-Output "stdout=$stdoutLog"
Write-Output "stderr=$stderrLog"
