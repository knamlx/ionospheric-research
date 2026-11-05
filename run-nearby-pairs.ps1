param(
    [ValidateSet("run", "prepare", "status", "summary")]
    [string]$Action = "run",
    [ValidateRange(1, 32)]
    [int]$Workers = 12
)

$ErrorActionPreference = "Stop"
$pythonLauncher = Get-Command py -ErrorAction Stop
$entryPoint = Join-Path $PSScriptRoot "shared\code\run_nearby_pairs.py"
& $pythonLauncher.Source -3.13 $entryPoint $Action --workers $Workers
if ($LASTEXITCODE -ne 0) {
    throw "Nearby-pair action '$Action' failed with exit code $LASTEXITCODE."
}
