[CmdletBinding()]
param(
    [string]$Python = (Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'),
    [string]$FfmpegDir,
    [string]$VlcDir,
    [string]$ModelsDir,
    [string]$OutputDir,
    [string]$WorkDir,
    [switch]$WithoutVlc,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$taskBuildScript = Join-Path $PSScriptRoot 'build_public_windows.py'
$taskBuildArguments = @($taskBuildScript)
foreach ($taskBuildOption in @(
    @('--ffmpeg-dir', $FfmpegDir),
    @('--vlc-dir', $VlcDir),
    @('--models-dir', $ModelsDir),
    @('--output-dir', $OutputDir),
    @('--work-dir', $WorkDir)
)) {
    if ($taskBuildOption[1]) {
        $taskBuildArguments += $taskBuildOption
    }
}
if ($WithoutVlc) { $taskBuildArguments += '--without-vlc' }
if ($CheckOnly) { $taskBuildArguments += '--check-only' }
& $Python @taskBuildArguments
exit $LASTEXITCODE
