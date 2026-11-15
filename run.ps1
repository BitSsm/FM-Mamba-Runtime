param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs)
$ErrorActionPreference = 'Stop'
$taskPython = if ($env:FM_MAMBA_PYTHON) { $env:FM_MAMBA_PYTHON } else { Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Install .venv following README.md.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TORCHDYNAMO_DISABLE = '1'
$env:WANDB_MODE = 'disabled'
if (-not $PythonArgs) { $PythonArgs = @('experiments.py', 'list') }
Push-Location $PSScriptRoot
try { & $taskPython @PythonArgs; $taskResult = $LASTEXITCODE } finally { Pop-Location }
exit $taskResult
