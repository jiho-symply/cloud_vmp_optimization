param(
  [int]$OnDemand = 4,
  [int]$Spot = 4,
  [int]$Batch = 4,
  [int]$Threads = 32,
  [int]$MaxVcpu = 8,
  [double]$ServerCapacity = 8.0,
  [double]$NoRelHeurTime = 0.0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "run_bc00_open.py"
$totalVm = $OnDemand + $Spot + $Batch
$analysisName = "chance_2sp_toy_{0}vm_bc00_open_cap{1}_vcpu{2}" -f $totalVm, [int]$ServerCapacity, $MaxVcpu
if($NoRelHeurTime -gt 0){
  $analysisName += "_norel{0}" -f [int]$NoRelHeurTime
}
$analysisDir = Join-Path $PSScriptRoot ("results\analysis\{0}" -f $analysisName)

New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null

$stdout = Join-Path $analysisDir "launcher_stdout.log"
$stderr = Join-Path $analysisDir "launcher_stderr.log"

$process = Start-Process `
  -FilePath $python `
  -ArgumentList @(
    $script,
    "--threads", $Threads,
    "--on-demand", $OnDemand,
    "--spot", $Spot,
    "--batch", $Batch,
    "--max-vcpu", $MaxVcpu,
    "--server-capacity", $ServerCapacity,
    "--no-rel-heur-time", $NoRelHeurTime
  ) `
  -WorkingDirectory $repoRoot `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -PassThru

Set-Content -Path (Join-Path $analysisDir "launcher_pid.txt") -Value $process.Id
Write-Output $process.Id
