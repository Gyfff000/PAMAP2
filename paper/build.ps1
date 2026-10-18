$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $texCommand = Get-Command xelatex -ErrorAction SilentlyContinue
    $texExe = if ($texCommand) { $texCommand.Source } elseif (Test-Path -LiteralPath 'C:\texlive\2026\bin\windows\xelatex.exe') { 'C:\texlive\2026\bin\windows\xelatex.exe' } else { throw '请安装带有 CTeX 的 TeX Live，或在 Codex LaTeX 编辑器中编译。' }
    1..2 | ForEach-Object {
        & $texExe -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
        if ($LASTEXITCODE -ne 0) { throw 'LaTeX 编译失败，请查看 main.log。' }
    }
} finally {
    Pop-Location
}
