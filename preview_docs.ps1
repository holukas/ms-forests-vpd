<#
.SYNOPSIS
    Preview the Quarto documentation locally.

.DESCRIPTION
    The docs are published from GitHub Actions, which needs Pages enabled on the
    repository. While the repository is private, this is the way to read them.

    Nothing in docs/ is executed at render time, so no data folder is needed.
    Quarto comes from the quarto-cli package in the dev group:

        uv sync --group dev

.PARAMETER Render
    Build once into docs/_site instead of starting the live preview.

.PARAMETER Port
    Port for the preview server. Quarto picks one if omitted.

.PARAMETER NoBrowser
    Do not open a browser window.

.EXAMPLE
    .\preview_docs.ps1
    Live preview, reloads on save, opens a browser.

.EXAMPLE
    .\preview_docs.ps1 -Render
    Build once into docs/_site.

.EXAMPLE
    .\preview_docs.ps1 -Port 5000 -NoBrowser
#>

[CmdletBinding()]
param(
    [switch]$Render,
    [int]$Port,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
$docsDir = Join-Path $repoRoot 'docs'

# Quarto from the project environment, else whatever is on PATH.
$venvQuarto = Join-Path $repoRoot '.venv\Scripts\quarto.exe'
if (Test-Path $venvQuarto) {
    $quarto = $venvQuarto
}
else {
    $quarto = (Get-Command quarto -ErrorAction SilentlyContinue).Source
}

if (-not $quarto) {
    Write-Error @'
Quarto not found. Install it with:

    uv sync --group dev

or from https://quarto.org/docs/get-started/
'@
    exit 1
}

$quartoArgs = @($(if ($Render) { 'render' } else { 'preview' }), $docsDir)

if (-not $Render) {
    if ($PSBoundParameters.ContainsKey('Port')) { $quartoArgs += @('--port', $Port) }
    if ($NoBrowser) { $quartoArgs += '--no-browser' }
}

Write-Host "Running: $quarto $($quartoArgs -join ' ')`n"
& $quarto @quartoArgs
exit $LASTEXITCODE
