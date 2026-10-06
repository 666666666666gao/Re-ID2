$ErrorActionPreference = 'Stop'
$archiveProject = 'C:\Users\gb\projects\demo_dual_axis_20261002'
$archiveSourceRoot = 'D:\Program Files\UserCache\gb\ReID2-experiment-artifacts\identity_coordinate_missing49_stream_20261005'
$archiveTargetRoot = 'E:\ReID2-experiment-artifacts\closed_20261006\identity_coordinate_missing49_stream_20261005'
$archiveProofPath = Join-Path $archiveProject 'results\preflight\own_D_closed_missing_to_E_actual_20261006.json'
$archiveProgressPath = Join-Path $archiveProject 'results\preflight\own_D_closed_missing_to_E_progress_20261006.json'
if (Test-Path -LiteralPath $archiveProofPath) { throw 'Relocation already complete; do not duplicate' }
if (Test-Path -LiteralPath $archiveTargetRoot) { throw 'Target already exists; inspect the original relocation rather than restarting' }
$archiveResolvedSource = (Resolve-Path -LiteralPath $archiveSourceRoot).ProviderPath
if ($archiveResolvedSource -ne [IO.Path]::GetFullPath($archiveSourceRoot)) { throw 'Source path differs from explicitly named archive' }
$archiveTargetRoot = [IO.Path]::GetFullPath($archiveTargetRoot)
if (-not $archiveTargetRoot.StartsWith('E:\ReID2-experiment-artifacts\closed_20261006\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Target escapes explicitly named cold archive' }
$archiveReadout = Get-Content -LiteralPath (Join-Path $archiveProject 'results\identity_coordinate_missing49_stream_full_completed_20261005\three_dataset_missing_analysis\result.json') -Raw | ConvertFrom-Json
if ($archiveReadout.status -ne 'ACTUAL_THREE_DATASET_FIXED_BEST_ALL49_FOURSTATE_CPU_READOUT' -or $archiveReadout.checked_condition_query_rows -ne 1385622 -or $archiveReadout.new_neural_calls -ne 0) { throw 'Closed all49 CPU evidence missing' }
$archiveNames = @('RGBNT100_identity_frequency_shared_narrow_s42','RGBNT100_identity_axis_shared_narrow_s42','MSVR310_identity_frequency_shared_narrow_s42','MSVR310_identity_axis_shared_narrow_s42')
$archiveExpected = @()
foreach ($archiveName in $archiveNames) {
    $archiveReceipt = Get-Content -LiteralPath (Join-Path $archiveProject ('results\identity_coordinate_missing49_stream_full_completed_20261005\full\'+$archiveName+'\stream_archive.json')) -Raw | ConvertFrom-Json
    if ($archiveReceipt.status -ne 'ALL_DECLARED_CONDITIONS_GT_AUDITED_LOCAL_VERIFIED_REMOTE_RAW_CLEARED' -or @($archiveReceipt.conditions.PSObject.Properties).Count -ne 49) { throw 'Original49 GT/local SHA ACK not complete' }
    foreach ($archiveCondition in $archiveReceipt.conditions.PSObject.Properties) {
        if (-not $archiveCondition.Value.local_verified_acknowledged -or -not $archiveCondition.Value.server_copy_cleared) { throw 'Condition not closed' }
        $archiveRelative = 'full\'+$archiveName+'\'+$archiveCondition.Name+'\raw.npz'
        $archiveExpected += [PSCustomObject]@{ relative=$archiveRelative; bytes=[long]$archiveCondition.Value.file.bytes; sha256=$archiveCondition.Value.file.sha256 }
    }
}
$archiveExpectedBytes = [long](($archiveExpected | Measure-Object -Property bytes -Sum).Sum)
$archiveActualFiles = @(Get-ChildItem -LiteralPath $archiveSourceRoot -File -Recurse)
if ($archiveExpected.Count -ne 196 -or $archiveActualFiles.Count -ne 196 -or $archiveExpectedBytes -ne 15465901224) { throw 'Expected196 archive inventory mismatch' }
foreach ($archiveEntry in $archiveExpected) {
    $archiveFile = [IO.Path]::GetFullPath((Join-Path $archiveSourceRoot $archiveEntry.relative))
    if (-not $archiveFile.StartsWith($archiveSourceRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Source file escapes named batch' }
    $archiveResolvedFile = (Resolve-Path -LiteralPath $archiveFile).ProviderPath
    if ($archiveResolvedFile -ne $archiveFile -or (Get-Item -LiteralPath $archiveFile).Length -ne $archiveEntry.bytes) { throw 'Source path/size differs from original ACK' }
}
$archiveDfreeBefore = ([IO.DriveInfo]::new('D:\')).AvailableFreeSpace
$archiveEfreeBefore = ([IO.DriveInfo]::new('E:\')).AvailableFreeSpace
if ($archiveEfreeBefore -le $archiveExpectedBytes) { throw 'E capacity smaller than measured closed batch' }
New-Item -ItemType Directory -Path $archiveTargetRoot | Out-Null
if ((Resolve-Path -LiteralPath $archiveTargetRoot).ProviderPath -ne $archiveTargetRoot) { throw 'Created target resolves outside named target' }
$archiveRecords = @()
$archiveReleased = [long]0
$archiveStarted = (Get-Date).ToString('o')
foreach ($archiveEntry in $archiveExpected) {
    $archiveFile = [IO.Path]::GetFullPath((Join-Path $archiveSourceRoot $archiveEntry.relative))
    $archiveTargetFile = [IO.Path]::GetFullPath((Join-Path $archiveTargetRoot $archiveEntry.relative))
    if (-not $archiveFile.StartsWith($archiveSourceRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or -not $archiveTargetFile.StartsWith($archiveTargetRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'File operation escapes verified named roots' }
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($archiveTargetFile)) -Force | Out-Null
    Copy-Item -LiteralPath $archiveFile -Destination $archiveTargetFile
    if ((Get-Item -LiteralPath $archiveTargetFile).Length -ne $archiveEntry.bytes) { throw 'Target copy size mismatch; source retained' }
    $archiveTargetHash = (Get-FileHash -LiteralPath $archiveTargetFile -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($archiveTargetHash -ne $archiveEntry.sha256) { throw 'Target copy differs from original source SHA ACK; source retained' }
    Remove-Item -LiteralPath $archiveFile -Force
    $archiveReleased += $archiveEntry.bytes
    $archiveRecords += [PSCustomObject]@{ source=$archiveFile; retained=$archiveTargetFile; bytes=$archiveEntry.bytes; sha256=$archiveTargetHash; target_verified=$true; D_source_removed=$true }
    if ($archiveRecords.Count % 25 -eq 0) {
        @{ status='ACTUAL_CLOSED_RAW_RELOCATION_RUNNING'; started_at=$archiveStarted; moved_files=$archiveRecords.Count; released_bytes=$archiveReleased; records=$archiveRecords } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $archiveProgressPath -Encoding utf8
        Write-Output ('CLOSED_RAW_VERIFIED_MOVED '+$archiveRecords.Count+' files/'+$archiveReleased+' bytes')
    }
}
if ($archiveRecords.Count -ne 196 -or $archiveReleased -ne $archiveExpectedBytes -or @(Get-ChildItem -LiteralPath $archiveSourceRoot -File -Recurse).Count -ne 0) { throw 'Final source inventory not empty/complete' }
if ((Resolve-Path -LiteralPath $archiveSourceRoot).ProviderPath -ne 'D:\Program Files\UserCache\gb\ReID2-experiment-artifacts\identity_coordinate_missing49_stream_20261005') { throw 'Recursive empty-directory cleanup target differs from named source' }
Remove-Item -LiteralPath $archiveSourceRoot -Recurse -Force
$archiveFinalFiles = @(Get-ChildItem -LiteralPath $archiveTargetRoot -File -Recurse)
if ($archiveFinalFiles.Count -ne 196 -or [long](($archiveFinalFiles | Measure-Object -Property Length -Sum).Sum) -ne $archiveExpectedBytes) { throw 'Retained target final inventory mismatch' }
$archiveResult = @{ status='ACTUAL_CLOSED_MISSING196_RELOCATED_D_TO_E_SIZE_SHA_VERIFIED'; started_at=$archiveStarted; verified_at=(Get-Date).ToString('o'); source_root=$archiveSourceRoot; retained_root=$archiveTargetRoot; files=196; bytes=$archiveExpectedBytes; D_free_before=$archiveDfreeBefore; D_free_after=([IO.DriveInfo]::new('D:\')).AvailableFreeSpace; E_free_before=$archiveEfreeBefore; E_free_after=([IO.DriveInfo]::new('E:\')).AvailableFreeSpace; original_GT_SHA_ACKs_preserved=$true; lost_raw_files=0; current_training_or_raw_receivers_modified=$false; NN_calls=0; records=$archiveRecords }
$archiveResult | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $archiveProofPath -Encoding utf8
Write-Output ('ACTUAL_CLOSED_RAW_RELOCATION_COMPLETE '+$archiveExpectedBytes+' bytes,196 files,D_free='+$archiveResult.D_free_after)
