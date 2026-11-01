param([switch]$UseSavedCapture, [switch]$UseSavedPredictions,
      [string]$CaptureDirectory='',
      [ValidateRange(0,2147483647)][int]$Seed=42,
      [ValidateRange(1,10000)][int]$Frames=2,
      [string]$PythonExecutable='.venv-perception/Scripts/python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
if (-not $CaptureDirectory) { $CaptureDirectory = "outputs/multiview/robot_relative_seed_$Seed" }
Push-Location -LiteralPath $projectRoot
try {
    $python = if ([IO.Path]::IsPathRooted($PythonExecutable)) { $PythonExecutable } else { Join-Path $projectRoot $PythonExecutable }
    if (-not (Test-Path -LiteralPath $python)) { throw 'Create the Python 3.11 perception environment first; see docs/perception.md.' }
    $env:PYTHONPATH = Join-Path $projectRoot 'src'
    function Assert-Success { if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE); remaining stages stopped." } }
    if (-not $UseSavedCapture -and -not $UseSavedPredictions) {
        & $python -X utf8 scripts/capture_multiview.py --seed $Seed --frames $Frames --output $CaptureDirectory
        Assert-Success
    }
    if (-not $UseSavedPredictions) {
        & $python -X utf8 scripts/package_rgb.py $CaptureDirectory
        Assert-Success
        $config = Get-Content -Raw -LiteralPath configs/perception_remote.json | ConvertFrom-Json
        if ($config.host -notmatch '^[a-zA-Z0-9.-]+$' -or $config.user -notmatch '^[a-zA-Z0-9_-]+$' -or
            [int]$config.port -lt 1 -or [int]$config.port -gt 65535 -or
            $config.wsl_key_path -notmatch '^/[a-zA-Z0-9_./-]+$') { throw 'Invalid remote configuration' }
        $target = "$($config.user)@$($config.host)"
        $linuxRoot = & wsl.exe -d $config.wsl_distribution -- wslpath -u ($projectRoot.Replace('\','/'))
        Assert-Success
        $linuxRoot = $linuxRoot.Trim()
        $capturePath = (Resolve-Path -LiteralPath $CaptureDirectory).Path
        $linuxCapture = & wsl.exe -d $config.wsl_distribution -- wslpath -u ($capturePath.Replace('\','/'))
        Assert-Success
        $linuxCapture = $linuxCapture.Trim()
        $remoteDirectory = '/workspace/hssg-perception/run_' + [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_fff')
        & wsl.exe -d $config.wsl_distribution -- ssh -o BatchMode=yes -o ConnectTimeout=15 -i $config.wsl_key_path -p $config.port $target "mkdir -p $remoteDirectory"
        Assert-Success
        & wsl.exe -d $config.wsl_distribution -- scp -q -i $config.wsl_key_path -P $config.port "$linuxRoot/scripts/infer_multiview_gpu.py" "$linuxCapture/rgb_inputs.zip" "${target}:${remoteDirectory}/"
        Assert-Success
        # Remote venv setup is documented in docs/perception.md. No private key is uploaded.
        & wsl.exe -d $config.wsl_distribution -- ssh -i $config.wsl_key_path -p $config.port $target "cd $remoteDirectory && python -m zipfile -e rgb_inputs.zip inputs && ../venv/bin/python -u infer_multiview_gpu.py --inputs inputs --output predictions && python -m zipfile -c predictions.zip predictions"
        Assert-Success
        & wsl.exe -d $config.wsl_distribution -- scp -q -i $config.wsl_key_path -P $config.port "${target}:${remoteDirectory}/predictions.zip" "$linuxCapture/predictions.zip"
        Assert-Success
        & $python -m zipfile -e (Join-Path $CaptureDirectory 'predictions.zip') $CaptureDirectory
        Assert-Success
    }
    & $python -X utf8 scripts/reconstruct_multiview.py --capture $CaptureDirectory --predictions (Join-Path $CaptureDirectory 'predictions') --output (Join-Path $CaptureDirectory 'reconstruction')
    Assert-Success
} finally {
    Pop-Location
}
