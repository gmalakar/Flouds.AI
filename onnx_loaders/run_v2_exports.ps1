# Runs all v2 export commands. Continues on errors.
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\onnx_loaders\run_v2_exports.ps1
#   powershell -ExecutionPolicy Bypass -File .\onnx_loaders\run_v2_exports.ps1 -UseVenv -Force
# Customize ONNX_PATH via env if needed: $env:ONNX_PATH = "onnx"

param(
    [switch]$UseVenv,
    [switch]$Force
)

$ErrorActionPreference = 'Continue'

# Determine Python command (optionally use workspace .venv)
$repoRoot = Split-Path $PSScriptRoot -Parent
if ($UseVenv) {
    $pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (!(Test-Path $pythonExe)) {
        Write-Error "Python venv not found at $pythonExe. Create/activate .venv first."; exit 1
    }
    $pythonCmd = '"' + $pythonExe + '"'
} else {
    $pythonCmd = 'python'
}

$commands = @(
    "$pythonCmd onnx_loaders/export_model.py --model_for fe --task feature-extraction --model_name sentence-transformers/all-MiniLM-L6-v2 --optimize",
    "$pythonCmd onnx_loaders/export_model.py --model_for fe --task feature-extraction --model_name sentence-transformers/paraphrase-MiniLM-L6-v2 --optimize",
    "$pythonCmd onnx_loaders/export_model.py --model_for fe --task feature-extraction --model_name sentence-transformers/sentence-t5-base --finetune --use_t5_encoder --optimize",
    "$pythonCmd onnx_loaders/export_model.py --model_for fe --task feature-extraction --model_name PleIAs/Pleias-Pico --optimize",
    "$pythonCmd onnx_loaders/export_model.py --model_for fe --task feature-extraction --model_name sentence-transformers/all-mpnet-base-v2 --optimize --framework pt",
    "$pythonCmd onnx_loaders/export_model.py --model_for fe --task feature-extraction --model_name intfloat/e5-base-v2 --optimize",
    "$pythonCmd onnx_loaders/export_model.py --model_for fe --task feature-extraction --model_name hkunlp/instructor-xl --optimize",
    "$pythonCmd onnx_loaders/export_model.py --model_for s2s --task seq2seq-lm --model_name t5-small --optimize",
    "$pythonCmd onnx_loaders/export_model.py --model_for s2s --model_name facebook/bart-large-cnn --task text2text-generation-with-past --use_cache --pack_single_file --pack_single_threshold_mb 1024",
    "$pythonCmd onnx_loaders/export_model.py --model_for s2s --model_name Falconsai/text_summarization --optimize --task seq2seq-lm --model_folder falconsai_text_summarization",
    "$pythonCmd onnx_loaders/export_model.py --model_for s2s --model_name google/pegasus-cnn_dailymail --optimize --task text2text-generation-with-past --use_cache"
)

foreach ($cmd in $commands) {
    # If the script was invoked with -Force, append --force to commands that don't already include it
    $fullCmd = $cmd
    if ($Force -and ($cmd -notmatch "--force")) {
        $fullCmd = "$cmd --force"
    }

    Write-Host "`n=== Running: $fullCmd" -ForegroundColor Cyan
    try {
        Invoke-Expression $fullCmd
        Write-Host "Done: $fullCmd" -ForegroundColor Green
    }
    catch {
        Write-Warning "Failed: $fullCmd"
    }
}

Write-Host "`nAll v2 export commands completed." -ForegroundColor Yellow