$ErrorActionPreference = 'Stop'
$project = 'C:\Users\gb\projects\demo_dual_axis_20261002'
$sourceBase = (Resolve-Path -LiteralPath 'D:\Program Files\UserCache\gb\ReID2-experiment-artifacts').ProviderPath
$targetBase = (Resolve-Path -LiteralPath 'E:\ReID2-experiment-artifacts\closed_20261006').ProviderPath
$proofPath = Join-Path $project 'results\preflight\own_D_closed_control_to_E_actual_20261006.json'
$progressPath = Join-Path $project 'results\preflight\own_D_closed_control_to_E_progress_20261006.json'
if (Test-Path -LiteralPath $proofPath) { throw 'This relocation already completed; do not replay' }
$batches = @(
    @{ name='full_official_control_m3b_20261004'; receipt='full_official_control_distances_local_archive_20261004.json'; status='ALL588_CONTROL_DISTANCE_FILES_LOCAL_SHA256_VERIFIED_SERVER_COPIES_CLEARED'; count=588 },
    @{ name='full_official_control_readout_20261004'; receipt='full_official_control_readout_distances_local_archive_20261004.json'; status='ALL98_READOUT_DISTANCE_FILES_LOCAL_SHA256_VERIFIED_SERVER_COPIES_CLEARED'; count=98 }
)
$control = Get-Content -LiteralPath (Join-Path $project 'results\full_official_control_m3b_20261004\analysis.json') -Raw | ConvertFrom-Json
$readout = Get-Content -LiteralPath (Join-Path $project 'results\full_official_control_readout_20261004\readout_analysis.json') -Raw | ConvertFrom-Json
if ($control.status -ne 'SIX_FULL_OFFICIAL_CONTROL_RUNS_AND1764_STATE_CASES_ANALYZED' -or $readout.status -ne 'BOTH_FROZEN_FULL_OFFICIAL_READOUTS_AND1176_CASES_ANALYZED') { throw 'Completed analysis evidence missing' }
$entries = @()
foreach ($batch in $batches) {
    $sourceRoot = [IO.Path]::GetFullPath((Join-Path $sourceBase ($batch.name+'\raw_distances')))
    $targetRoot = [IO.Path]::GetFullPath((Join-Path $targetBase ($batch.name+'\raw_distances')))
    if ((Resolve-Path -LiteralPath $sourceRoot).ProviderPath -ne $sourceRoot -or -not $sourceRoot.StartsWith($sourceBase+'\',[StringComparison]::OrdinalIgnoreCase) -or -not $targetRoot.StartsWith($targetBase+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Named archive paths differ from resolved own roots' }
    if (Test-Path -LiteralPath (Join-Path $targetBase $batch.name)) { throw 'Target exists; inspect the original relocation instead of restarting' }
    if (@(Get-ChildItem -LiteralPath $sourceRoot -Recurse -Attributes ReparsePoint).Count -ne 0) { throw 'Source archive has a redirected path' }
    $receipt = Get-Content -LiteralPath (Join-Path $project ('results\preflight\'+$batch.receipt)) -Raw | ConvertFrom-Json
    if ($receipt.status -ne $batch.status -or $receipt.distance_files -ne $batch.count -or $receipt.local_root -ne $sourceRoot) { throw 'Original archive receipt does not match the named batch' }
    $batchEntries = @()
    foreach ($group in $receipt.files.PSObject.Properties) {
        foreach ($file in $group.Value.PSObject.Properties) {
            $relative = $file.Name.Replace('/','\')
            $source = [IO.Path]::GetFullPath((Join-Path $sourceRoot $relative))
            $target = [IO.Path]::GetFullPath((Join-Path $targetRoot $relative))
            if (-not $source.StartsWith($sourceRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or -not $target.StartsWith($targetRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetExtension($source) -ne '.npz') { throw 'File escapes the explicitly selected raw archive' }
            if ((Resolve-Path -LiteralPath $source).ProviderPath -ne $source -or (Get-Item -LiteralPath $source).Length -ne $file.Value.bytes) { throw 'Source path or size differs from original verified archive' }
            $batchEntries += [PSCustomObject]@{ source=$source; retained=$target; bytes=[long]$file.Value.bytes; sha256=$file.Value.sha256 }
        }
    }
    if ($batchEntries.Count -ne $batch.count -or [long](($batchEntries | Measure-Object bytes -Sum).Sum) -ne $receipt.total_bytes) { throw 'Original archive count or byte total mismatch' }
    $entries += $batchEntries
}
$totalBytes = [long](($entries | Measure-Object bytes -Sum).Sum)
if ($entries.Count -ne 686 -or $totalBytes -ne 7408033915) { throw 'Expected closed686 raw archive inventory differs' }
$dBefore = ([IO.DriveInfo]::new('D:\')).AvailableFreeSpace
$eBefore = ([IO.DriveInfo]::new('E:\')).AvailableFreeSpace
if ($eBefore -le $totalBytes) { throw 'Measured E capacity cannot hold the selected raw archive' }
$started = Get-Date -Format o
$records = @()
$released = [long]0
foreach ($entry in $entries) {
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($entry.retained)) -Force | Out-Null
    if ((Resolve-Path -LiteralPath ([IO.Path]::GetDirectoryName($entry.retained))).ProviderPath -ne [IO.Path]::GetDirectoryName($entry.retained)) { throw 'Created target parent resolves outside the named target' }
    Copy-Item -LiteralPath $entry.source -Destination $entry.retained
    if ((Get-Item -LiteralPath $entry.retained).Length -ne $entry.bytes -or (Get-FileHash -LiteralPath $entry.retained -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw 'Copied file differs from original size/SHA receipt; source retained' }
    Remove-Item -LiteralPath $entry.source -Force
    $released += $entry.bytes
    $records += [PSCustomObject]@{ source=$entry.source; retained=$entry.retained; bytes=$entry.bytes; sha256=$entry.sha256; target_verified=$true; D_source_removed=$true }
    if ($records.Count % 100 -eq 0) {
        @{ status='ACTUAL_CLOSED_CONTROL_RAW_RELOCATION_RUNNING'; started_at=$started; files=$records.Count; bytes=$released; records=$records } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $progressPath -Encoding utf8
        Write-Output ('VERIFIED_MOVED '+$records.Count+' files/'+$released+' bytes')
    }
}
if ($records.Count -ne 686 -or $released -ne $totalBytes -or @($entries | Where-Object { Test-Path -LiteralPath $_.source }).Count -ne 0) { throw 'Selected raw files not fully relocated' }
$result = @{ status='ACTUAL_CLOSED_CONTROL686_RELOCATED_D_TO_E_SIZE_SHA_VERIFIED'; started_at=$started; verified_at=(Get-Date -Format o); files=$records.Count; bytes=$released; D_free_before=$dBefore; D_free_after=([IO.DriveInfo]::new('D:\')).AvailableFreeSpace; E_free_before=$eBefore; E_free_after=([IO.DriveInfo]::new('E:\')).AvailableFreeSpace; original_receipts_preserved=$true; lost_raw_files=0; weights_modified=0; NN_calls=0; active_receiver_or_source_modified=$false; records=$records }
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $proofPath -Encoding utf8
Write-Output ('ACTUAL_CLOSED_CONTROL_RELOCATION_COMPLETE files='+$result.files+' bytes='+$result.bytes+' D_free='+$result.D_free_after)
