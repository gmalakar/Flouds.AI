# Runs all v2 export commands. Continues on errors.
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\onnx_loaders\run_v2_exports.ps1
#   powershell -ExecutionPolicy Bypass -File .\onnx_loaders\run_v2_exports.ps1 -UseVenv -Force
#   powershell -ExecutionPolicy Bypass -File .\onnx_loaders\run_v2_exports.ps1 -Force -Optimize -Cleanup -SkipValidator -PruneCanonical
#
# Flags:
#   -UseVenv       : Use the repository .venv Python interpreter
#   -Force         : Append --force to each export command
#   -Optimize      : Append --optimize to each export command
#   -Cleanup       : Append --cleanup to each export command
#   -SkipValidator : Append --skip-validator to each export command
#   -PruneCanonical: Append --prune_canonical to each export command (optionally remove canonical ONNX files when merged artifacts exist)
#
# Customize ONNX_PATH via env if needed: $env:ONNX_PATH = "onnx"

param(
    [switch]$UseVenv,
    [switch]$Force,
    [switch]$SkipValidator,
    [switch]$Optimize,
    [switch]$Cleanup,
    [switch]$PruneCanonical,
    [switch]$NoLocalPrep
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

# ============================
# Recommended Models Only
# ============================

$commands = @(

    # -------------------------
    # Embedding Models (fe)
    # -------------------------

    #"$pythonCmd onnx_loaders/export_model.py --model-for fe --model-name nomic-ai/nomic-embed-text-v1.5 --task feature-extraction --library sentence_transformers --trust-remote-code"
    #"$pythonCmd onnx_loaders/export_model.py --model-for fe --model-name BAAI/bge-small-en-v1.5 --task feature-extraction --library sentence_transformers --normalize-embeddings",
    #"$pythonCmd onnx_loaders/export_model.py --model-for fe --model-name sentence-transformers/all-MiniLM-L6-v2 --task feature-extraction --library sentence_transformers --normalize-embeddings",
    #"$pythonCmd onnx_loaders/export_model.py --model-for fe --model-name intfloat/e5-small-v2 --task feature-extraction --library transformers --normalize-embeddings --force",
    #"$pythonCmd onnx_loaders/export_model.py --model-for fe --model-name BAAI/bge-m3 --task feature-extraction --library transformers --trust-remote-code",
    #"$pythonCmd onnx_loaders/export_model.py --model-for fe --model-name thenlper/gte-small --task feature-extraction --library transformers",

    "$pythonCmd onnx_loaders/export_model.py --model-for s2s --model-name t5-small --task seq2seq-lm --library transformers"
    #"$pythonCmd onnx_loaders/export_model.py --model-for s2s --model-name facebook/bart-large-cnn --task text2text-generation-with-past --pack-single-file --library transformers"

    #"$pythonCmd onnx_loaders/export_model.py --model-for llm --model-name microsoft/phi-2 --task text-generation-with-past --library transformers --trust-remote-code --merge",
    #"$pythonCmd onnx_loaders/export_model.py --model-for llm --model-name google/gemma-2b-it --task text-generation-with-past --library transformers --trust-remote-code --merge"   
)

foreach ($cmd in $commands) {
    # If -Force was passed, append --force unless already present
    $fullCmd = $cmd
    if ($Force -and ($cmd -notmatch "--force")) {
        $fullCmd = "$cmd --force"
    }

    # If -Optimize was passed, append --optimize unless already present
    if ($Optimize -and ($fullCmd -notmatch "--optimize")) {
        $fullCmd = "$fullCmd --optimize"
    }

    # If -Cleanup was passed, append --cleanup unless already present
    if ($Cleanup -and ($fullCmd -notmatch "--cleanup")) {
        $fullCmd = "$fullCmd --cleanup"
    }

    # If -PruneCanonical was passed, append --prune_canonical unless already present
    if ($PruneCanonical -and ($fullCmd -notmatch "--prune-canonical")) {
        $fullCmd = "$fullCmd --prune-canonical"
    }

    # If -NoLocalPrep was passed, append --no-local-prep unless already present
    if ($NoLocalPrep -and ($fullCmd -notmatch "--no-local-prep")) {
        $fullCmd = "$fullCmd --no-local-prep"
    }

    # If -SkipValidator was passed, append --skip-validator unless already present
    if ($SkipValidator -and ($fullCmd -notmatch "--skip-validator")) {
        $fullCmd = "$fullCmd --skip-validator"
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