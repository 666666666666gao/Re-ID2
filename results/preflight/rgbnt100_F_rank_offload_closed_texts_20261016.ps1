$ErrorActionPreference = 'Stop'
$taskProject = [System.IO.Path]::GetFullPath('C:\Users\gb\projects\demo_dual_axis_20261002')
$taskSource = [System.IO.Path]::GetFullPath('C:\Users\gb\projects\demo_dual_axis_20261002\results\rgbnt100_expert_N5_AV_20261015')
$taskArchiveRoot = [System.IO.Path]::GetFullPath('D:\Program Files\UserCache\gb\ReID2-experiment-artifacts')
$taskDestination = [System.IO.Path]::GetFullPath('D:\Program Files\UserCache\gb\ReID2-experiment-artifacts\closed_text_results\rgbnt100_expert_N5_AV_20261015')
$taskProof = Join-Path $taskArchiveRoot 'ReID2_closed_N5_text_C_space_recovery_20261016.json'
if (-not $taskSource.StartsWith($taskProject + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Source outside project' }
if (-not $taskDestination.StartsWith($taskArchiveRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Destination outside artifact archive' }
if ((Get-Item -LiteralPath $taskSource).Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw 'Source is already a link' }
if (Test-Path -LiteralPath $taskDestination) { throw 'Destination already exists' }
if (Test-Path -LiteralPath $taskProof) { throw 'Recovery proof already exists' }
$taskCBefore = (Get-PSDrive -Name C).Free
$taskDBefore = (Get-PSDrive -Name D).Free
foreach ($taskSeed in 43..46) {
    foreach ($taskVariant in @('frequency_shared','axis_shared')) {
        $taskResultPath = Join-Path $taskSource "s$taskSeed\RGBNT100\training\RGBNT100_n5_AV_${taskVariant}_s$taskSeed\result.json"
        $taskResult = Get-Content -LiteralPath $taskResultPath -Raw | ConvertFrom-Json
        if ($taskResult.status -ne 'COMPLETE' -or $taskResult.epochs -ne 50) { throw 'Expected closed N5 run is incomplete' }
    }
}
$taskInputs = @(& rg --files -uu -- $taskSource)
if ($LASTEXITCODE -ne 0) { throw 'Cannot enumerate closed text records' }
$taskManifest = @($taskInputs | ForEach-Object {
    $taskItem = Get-Item -LiteralPath $_
    [pscustomobject]@{relative=$taskItem.FullName.Substring($taskSource.Length + 1); bytes=$taskItem.Length; sha256=(Get-FileHash -LiteralPath $taskItem.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$taskBytes = ($taskManifest | Measure-Object -Property bytes -Sum).Sum
if ($taskDBefore -le $taskBytes + 2090564504 + 500000000) { throw 'Insufficient D capacity for text offload and pending six F archives' }
$taskParent = Split-Path -Parent $taskDestination
New-Item -ItemType Directory -Path $taskParent -Force | Out-Null
Copy-Item -LiteralPath $taskSource -Destination $taskDestination -Recurse
$taskCopied = @(& rg --files -uu -- $taskDestination)
if ($LASTEXITCODE -ne 0 -or $taskCopied.Count -ne $taskManifest.Count) { throw 'Copied record count differs' }
foreach ($taskRow in $taskManifest) {
    $taskOld = Join-Path $taskSource $taskRow.relative
    $taskNew = Join-Path $taskDestination $taskRow.relative
    if ((Get-Item -LiteralPath $taskOld).Length -ne $taskRow.bytes -or (Get-FileHash -LiteralPath $taskOld -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskRow.sha256) { throw 'Source changed during copy' }
    if ((Get-Item -LiteralPath $taskNew).Length -ne $taskRow.bytes -or (Get-FileHash -LiteralPath $taskNew -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskRow.sha256) { throw 'Copied record verification failed' }
}
$taskResolvedSource = (Resolve-Path -LiteralPath $taskSource).ProviderPath
if ($taskResolvedSource -ne $taskSource -or -not $taskResolvedSource.StartsWith($taskProject + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Resolved delete target changed' }
Remove-Item -LiteralPath $taskResolvedSource -Recurse -Force
New-Item -ItemType Junction -Path $taskSource -Target $taskDestination | Out-Null
foreach ($taskRow in $taskManifest) {
    $taskRestored = Join-Path $taskSource $taskRow.relative
    if ((Get-Item -LiteralPath $taskRestored).Length -ne $taskRow.bytes -or (Get-FileHash -LiteralPath $taskRestored -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskRow.sha256) { throw 'Original path no longer resolves to exact records' }
}
$taskResult = [ordered]@{status='ACTUAL_C_FULL_RECOVERY_CLOSED_N5_TEXTS_MOVED_TO_D_ORIGINAL_PATH_JUNCTION'; at=(Get-Date).ToString('o'); source=$taskSource; destination=$taskDestination; files=$taskManifest.Count; bytes=$taskBytes; C_free_before=$taskCBefore; C_free_after=(Get-PSDrive -Name C).Free; D_free_before=$taskDBefore; D_free_after=(Get-PSDrive -Name D).Free; all_original_paths_size_sha_exact=$true; unique_records_deleted=0; new_NN=0; temperature_power_reads=0; manifest=$taskManifest}
$taskResult | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskProof -Encoding UTF8
$taskResult | Select-Object status,files,bytes,C_free_before,C_free_after,D_free_after,new_NN | ConvertTo-Json -Depth 3
