$ErrorActionPreference = 'Stop'
$project = 'C:\Users\gb\projects\demo_dual_axis_20261002'
$sourceBase = (Resolve-Path -LiteralPath 'D:\Program Files\UserCache\gb\ReID2-experiment-artifacts').ProviderPath
$targetBase = (Resolve-Path -LiteralPath 'E:\ReID2-experiment-artifacts\closed_20261006').ProviderPath
$proofPath = Join-Path $project 'results\preflight\own_D_closed_M6_M8_to_E_20261006_actual.json'
$progressPath = Join-Path $project 'results\preflight\own_D_closed_M6_M8_to_E_20261006_progress.json'
if (Test-Path -LiteralPath $proofPath) { throw 'This relocation already completed; do not replay' }
$plan = Get-Content -LiteralPath (Join-Path $project 'results\preflight\own_D_closed_M6_M8_to_E_20261006_plan.json') -Raw | ConvertFrom-Json
$entries = @($plan.entries)
if ($entries.Count -ne 637 -or @($entries.source | Sort-Object -Unique).Count -ne 637) { throw 'Exact closed archive count differs' }
foreach ($archive in @('full_official_public_outlet_identity_m6_20261005','full_official_frozen_public_identity_m8_20261005')) {
    $sourceRoot = [IO.Path]::GetFullPath((Join-Path $sourceBase $archive))
    $targetRoot = [IO.Path]::GetFullPath((Join-Path $targetBase $archive))
    if ((Resolve-Path -LiteralPath $sourceRoot).ProviderPath -ne $sourceRoot -or -not $sourceRoot.StartsWith($sourceBase+'\',[StringComparison]::OrdinalIgnoreCase) -or -not $targetRoot.StartsWith($targetBase+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Named roots differ from resolved own paths' }
    if (Test-Path -LiteralPath $targetRoot) { throw 'Target exists; inspect original relocation' }
    if (@(Get-ChildItem -LiteralPath $sourceRoot -Recurse -Attributes ReparsePoint).Count -ne 0) { throw 'Archive has a redirected path' }
}
foreach ($entry in $entries) {
    $sourcePath = [IO.Path]::GetFullPath($entry.source)
    $targetPath = [IO.Path]::GetFullPath($entry.retained)
    if (-not $sourcePath.StartsWith($sourceBase+'\',[StringComparison]::OrdinalIgnoreCase) -or -not $targetPath.StartsWith($targetBase+'\',[StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetExtension($sourcePath) -ne '.npz') { throw 'File escapes named archive' }
    if ((Resolve-Path -LiteralPath $sourcePath).ProviderPath -ne $sourcePath -or (Get-Item -LiteralPath $sourcePath).Length -ne $entry.bytes) { throw 'Source path or size differs from receipt' }
}
$totalBytes = [long](($entries | Measure-Object bytes -Sum).Sum)
if ($totalBytes -ne $plan.bytes) { throw 'Byte total differs from original receipts' }
$dBefore = ([IO.DriveInfo]::new('D:\')).AvailableFreeSpace
$eBefore = ([IO.DriveInfo]::new('E:\')).AvailableFreeSpace
if ($eBefore -le $totalBytes) { throw 'E capacity insufficient for selected archive' }
$started = Get-Date -Format o
$records = @()
$released = [long]0
foreach ($entry in $entries) {
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($entry.retained)) -Force | Out-Null
    if ((Resolve-Path -LiteralPath ([IO.Path]::GetDirectoryName($entry.retained))).ProviderPath -ne [IO.Path]::GetDirectoryName($entry.retained)) { throw 'Target parent resolves outside named path' }
    Copy-Item -LiteralPath $entry.source -Destination $entry.retained
    if ((Get-Item -LiteralPath $entry.retained).Length -ne $entry.bytes -or (Get-FileHash -LiteralPath $entry.retained -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw 'Copied file differs from original size/SHA; source retained' }
    Remove-Item -LiteralPath $entry.source -Force
    $released += $entry.bytes
    $records += [PSCustomObject]@{ source=$entry.source; retained=$entry.retained; bytes=$entry.bytes; sha256=$entry.sha256; target_verified=$true; D_source_removed=$true }
    if ($records.Count % 100 -eq 0) {
        @{status='ACTUAL_CLOSED_M6_M8_RAW_RELOCATION_RUNNING';started_at=$started;files=$records.Count;bytes=$released;records=$records} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $progressPath -Encoding utf8
        Write-Output ('VERIFIED_MOVED '+$records.Count+' files/'+$released+' bytes')
    }
}
if ($records.Count -ne 637 -or $released -ne $totalBytes -or @($entries | Where-Object { Test-Path -LiteralPath $_.source }).Count -ne 0) { throw 'Selected files not fully relocated' }
$result = @{status='ACTUAL_CLOSED_M6_M8_637_RELOCATED_D_TO_E_SIZE_SHA_VERIFIED';started_at=$started;verified_at=(Get-Date -Format o);files=$records.Count;bytes=$released;D_free_before=$dBefore;D_free_after=([IO.DriveInfo]::new('D:\')).AvailableFreeSpace;E_free_before=$eBefore;E_free_after=([IO.DriveInfo]::new('E:\')).AvailableFreeSpace;original_receipts_preserved=$true;lost_raw_files=0;weights_modified=0;NN_calls=0;records=$records}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $proofPath -Encoding utf8
Write-Output ('ACTUAL_CLOSED_M6_M8_RELOCATION_COMPLETE files='+$result.files+' bytes='+$result.bytes+' D_free='+$result.D_free_after)
