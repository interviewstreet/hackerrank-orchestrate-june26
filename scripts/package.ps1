<#
.SYNOPSIS
    Allowlist-based packaging script that creates code.zip for submission.

.DESCRIPTION
    Builds code.zip with a top-level code/ directory containing only the
    explicitly listed source, tests, prompts, evaluation, requirements,
    and documentation files.

    Excluded by design: .env, caches, __pycache__, .pyc, logs, smoke outputs,
    output.csv, git metadata, and unrelated strategy artifacts.

    Run from challenge/ with the virtual environment active.

.EXAMPLE
    .\scripts\package.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path $PSScriptRoot -Parent   # challenge/
$staging = Join-Path $root "code_staging"
$codeDir = Join-Path $staging "code"        # top-level code/ inside zip
$dest = Join-Path $root "code.zip"

Write-Host "==> Packaging from: $root"
Write-Host "==> Staging at:     $staging"
Write-Host "==> Output:         $dest"

# Clean any previous staging
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Force $codeDir | Out-Null

$src = Join-Path $root "code"

# --- Allowlisted directories (copied in full, then pruned) ---
$allowedDirs = @(
    "agent",
    "evaluation",
    "tests"
)

foreach ($dir in $allowedDirs) {
    $srcDir = Join-Path $src $dir
    $dstDir = Join-Path $codeDir $dir
    if (Test-Path $srcDir) {
        Copy-Item $srcDir $dstDir -Recurse
    } else {
        Write-Warning "Directory not found, skipping: $srcDir"
    }
}

# --- Allowlisted individual files ---
$allowedFiles = @(
    "main.py",
    "requirements.txt",
    "README.md",
    "JUDGE_WALKTHROUGH.md",
    "__init__.py",
    "implementation_plan.md"
)

foreach ($f in $allowedFiles) {
    $srcFile = Join-Path $src $f
    if (Test-Path $srcFile) {
        Copy-Item $srcFile (Join-Path $codeDir $f)
    }
}

# --- Copy .env.example into code/ ---
$envExample = Join-Path $root ".env.example"
if (Test-Path $envExample) {
    Copy-Item $envExample (Join-Path $codeDir ".env.example")
}

# --- Copy evaluation/*.md explicitly (ensure both reports are included) ---
foreach ($mdFile in @("RESULTS.md", "evaluation_report.md")) {
    $mdPath = Join-Path $src "evaluation\$mdFile"
    if (Test-Path $mdPath) {
        Copy-Item $mdPath (Join-Path $codeDir "evaluation\$mdFile") -Force
    }
}

# --- Prune excluded artifacts from staging ---
$excludePatterns = @(
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    "*.egg-info",
    ".cache"
)

foreach ($pat in $excludePatterns) {
    Get-ChildItem $staging -Filter $pat -Recurse -Force |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# --- Verify required files are present ---
$required = @(
    "code\main.py",
    "code\requirements.txt",
    "code\README.md",
    "code\JUDGE_WALKTHROUGH.md",
    "code\.env.example",
    "code\agent\validator.py",
    "code\agent\pipeline.py",
    "code\agent\prompt.py",
    "code\evaluation\main.py",
    "code\evaluation\RESULTS.md",
    "code\evaluation\evaluation_report.md",
    "code\tests\test_validator.py"
)

$missing = @()
foreach ($r in $required) {
    if (-not (Test-Path (Join-Path $staging $r))) {
        $missing += $r
    }
}
if ($missing.Count -gt 0) {
    Write-Error "Required files missing from staging: $($missing -join ', ')"
    exit 1
}

# --- Confirm no secrets crept in ---
# .env (without .example suffix) must never be present; .env.example is allowed
$envHits = Get-ChildItem $staging -Filter ".env" -Recurse -Force
if ($envHits) {
    Write-Error "Secret file .env found in staging: $($envHits.FullName -join ', ')"
    Remove-Item $staging -Recurse -Force
    exit 1
}
$forbidden = @("*.key", "*.secret")
foreach ($pat in $forbidden) {
    $hits = Get-ChildItem $staging -Filter $pat -Recurse -Force
    if ($hits) {
        Write-Error "Secret-like file found in staging: $($hits.FullName -join ', ')"
        Remove-Item $staging -Recurse -Force
        exit 1
    }
}

# --- Compress ---
if (Test-Path $dest) { Remove-Item $dest -Force }
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $dest -CompressionLevel Optimal

# --- Report ---
Remove-Item $staging -Recurse -Force

$sizeMB = [math]::Round((Get-Item $dest).Length / 1MB, 3)
$sizeKB = [math]::Round((Get-Item $dest).Length / 1KB, 1)
$sha256 = (Get-FileHash $dest -Algorithm SHA256).Hash

Write-Host ""
Write-Host "==> code.zip created successfully"
Write-Host "    Size:   $sizeKB KB ($sizeMB MB)"
Write-Host "    SHA256: $sha256"
Write-Host ""
Write-Host "Submit three artifacts separately to HackerRank:"
Write-Host "  1. code.zip           (this archive)"
Write-Host "  2. output.csv         (inference results; gitignored; never bundled)"
Write-Host "  3. chat_transcript.txt (AI interaction transcript; outside code.zip)"
