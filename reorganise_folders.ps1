# reorganise_folders.ps1
# Organises dissartation_2 into clean subfolders.
# SAFE: only moves docs/outputs. All model scripts and CSVs stay in root
# so step4_model.py paths remain unbroken.
# Run from dissartation_2 directory:
#   cd "C:\Users\mar_m\Downloads\master\Term2\dissartation\files\dissartation_2"
#   .\reorganise_folders.ps1

$root = "C:\Users\mar_m\Downloads\master\Term2\dissartation\files\dissartation_2"

# ── Create folders ────────────────────────────────────────────────────────────
New-Item -ItemType Directory -Force "$root\docs"   | Out-Null
New-Item -ItemType Directory -Force "$root\output" | Out-Null

Write-Host "Folders created: docs\  output\" -ForegroundColor Cyan

# ── Move dissertation documents ───────────────────────────────────────────────
$docs = @(
    "DISSERTATION_BRIEF.md",
    "dissertation.tex",
    "dissertation_colab.ipynb"
)
foreach ($f in $docs) {
    if (Test-Path "$root\$f") {
        Move-Item "$root\$f" "$root\docs\$f" -Force
        Write-Host "  Moved -> docs\$f"
    }
}

# ── Move output files ─────────────────────────────────────────────────────────
$outputs = @(
    "supervisor_comparison_results.xlsx",
    "borough_wmape_map.png"
)
foreach ($f in $outputs) {
    if (Test-Path "$root\$f") {
        Move-Item "$root\$f" "$root\output\$f" -Force
        Write-Host "  Moved -> output\$f"
    }
}

# ── Delete junk result files (superseded by named experiment files) ────────────
$junk = @(
    "results_cv.csv",
    "results_summary.csv",
    "results_cv_multigraph_quick.csv",
    "results_summary_multigraph_quick.csv",
    "results_cv_multigraph.csv",
    "results_summary_multigraph.csv"
)
foreach ($f in $junk) {
    if (Test-Path "$root\$f") {
        Remove-Item "$root\$f" -Force
        Write-Host "  Deleted: $f" -ForegroundColor Yellow
    }
}

# ── Final state ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Done. Final layout:" -ForegroundColor Green
Write-Host ""
Write-Host "  data\              Raw input (BUSTO, AI23, Bus_Stops)"
Write-Host "  cache\             OSM download cache — DO NOT DELETE"
Write-Host "  docs\              Dissertation tex, notebook, brief"
Write-Host "  output\            Excel, maps, generated figures"
Write-Host "  step*.py           Pipeline scripts (stay in root)"
Write-Host "  stops_features*.csv  Feature matrices (stay in root)"
Write-Host "  results_*.csv      Experiment results (stay in root)"
Write-Host "  experiment_log.md  Full bug + results history"
Write-Host "  PROJECT_HANDOFF.md Paste into any new chat to restore context"
