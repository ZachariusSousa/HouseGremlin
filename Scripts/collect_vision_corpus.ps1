param(
    [ValidateRange(1, 10000)]
    [int]$Count = 300,
    [string]$BrainUrl = "http://localhost:8080",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Output) {
    $Output = Join-Path $repoRoot "pc_brain\evals\vision_frames"
}
$resolvedOutput = [System.IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null

for ($index = 1; $index -le $Count; $index++) {
    $name = "frame-{0:D4}-{1}.jpg" -f $index, (Get-Date -Format "yyyyMMdd-HHmmssfff")
    $target = Join-Path $resolvedOutput $name
    Invoke-WebRequest -Uri "$($BrainUrl.TrimEnd('/'))/robot/camera/capture" -OutFile $target
    Write-Progress -Activity "Collecting Robit vision corpus" -Status "$index / $Count" -PercentComplete (($index / $Count) * 100)
    if ($index -lt $Count) { Start-Sleep -Seconds 1 }
}

Write-Host "[vision] Saved $Count opt-in frames to $resolvedOutput"
