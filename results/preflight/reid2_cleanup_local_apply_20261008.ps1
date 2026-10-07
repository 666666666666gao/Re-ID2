$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath('C:\Users\gb\projects\demo_dual_axis_20261002').TrimEnd('\')
$planPath = Join-Path $projectRoot 'results\preflight\reid2_cleanup_plan_20261008.json'
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if ($plan.status -ne 'VERIFIED_EXPLICIT_CLEANUP_READY_NOT_APPLIED') { throw 'Cleanup plan is not ready.' }
$proofPath = Join-Path $projectRoot 'results\preflight\reid2_cleanup_local_actual_20261008.json'
if (Test-Path -LiteralPath $proofPath) { throw 'Local cleanup already recorded.' }
$artifactRoots = @('D:\Program Files\UserCache\gb\ReID2-experiment-artifacts', 'E:\ReID2-experiment-artifacts')
$raw = @()
foreach ($root in $artifactRoots) {
    $raw += @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter '*.npz' | ForEach-Object { [PSCustomObject]@{ path=$_.FullName; bytes=$_.Length } })
}
$files = @($plan.local_packages) + @($plan.pathspecs)
foreach ($record in $files) {
    $target = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $record.path).Path)
    if (-not $target.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Target leaves project.' }
    if ($target -notmatch '\.(tar\.gz|pathspec)$') { throw 'Target is not a temporary transfer artifact.' }
    if ((Get-Item -LiteralPath $target).Length -ne $record.bytes) { throw 'Temporary file size changed.' }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) { throw 'Temporary file content changed.' }
}
foreach ($record in $plan.python_caches) {
    $target = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $record.path).Path)
    if (-not $target.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $target -Leaf) -ne '__pycache__') { throw 'Cache target leaves intended scope.' }
    if (@(Get-ChildItem -LiteralPath $target -Recurse -File | Where-Object { $_.Extension -ne '.pyc' }).Count) { throw 'Cache contains non-generated files.' }
}
$before = @{}
foreach ($drive in @('C','D','E')) { $before[$drive] = (Get-PSDrive -Name $drive).Free }
$removed = @()
foreach ($record in $files) {
    Remove-Item -LiteralPath $record.path
    $removed += [PSCustomObject]@{ path=$record.path; bytes=$record.bytes; sha256=$record.sha256; category='verified_transfer_temporary' }
}
foreach ($record in $plan.python_caches) {
    $target = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $record.path).Path)
    if (-not $target.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Recursive cache target leaves project.' }
    Remove-Item -LiteralPath $target -Recurse
    $removed += [PSCustomObject]@{ path=$record.path; bytes=$record.bytes; category='generated_python_cache' }
}
$empty = @()
foreach ($root in $artifactRoots) {
    $rootAbsolute = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $root).Path).TrimEnd('\')
    $dirs = Get-ChildItem -LiteralPath $root -Directory -Recurse | Sort-Object { $_.FullName.Length } -Descending
    foreach ($dir in $dirs) {
        $target = [IO.Path]::GetFullPath($dir.FullName)
        if (-not $target.StartsWith($rootAbsolute + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Empty directory leaves artifact root.' }
        if (@(Get-ChildItem -LiteralPath $target -Force).Count -eq 0) {
            Remove-Item -LiteralPath $target
            $empty += $target
        }
    }
}
foreach ($record in $raw) {
    if (-not (Test-Path -LiteralPath $record.path) -or (Get-Item -LiteralPath $record.path).Length -ne $record.bytes) { throw 'Protected local array changed.' }
}
$after = @{}
foreach ($drive in @('C','D','E')) { $after[$drive] = (Get-PSDrive -Name $drive).Free }
$result = [ordered]@{ status='ACTUAL_PROJECT_LOCAL_TEMPORARIES_CLEANED_RAW_PRESERVED'; finished=(Get-Date -Format o); removed=$removed; removed_items=$removed.Count; removed_bytes=($removed | Measure-Object -Property bytes -Sum).Sum; empty_directories_removed=$empty; protected_array_files=$raw.Count; protected_array_bytes=($raw | Measure-Object -Property bytes -Sum).Sum; before_free=$before; after_free=$after; new_neural_calls=0; new_optimizer_updates=0; goal='PAUSED_USER_REQUEST_OBJECTIVE_UNMET' }
$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $proofPath -Encoding utf8
[ordered]@{status=$result.status;removed_items=$result.removed_items;removed_bytes=$result.removed_bytes;empty_directories_removed=$empty.Count;protected_array_files=$raw.Count;after_free=$after} | ConvertTo-Json -Depth 4
