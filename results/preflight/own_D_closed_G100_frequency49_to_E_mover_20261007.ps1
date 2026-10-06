$ErrorActionPreference = 'Stop'
$projectPath = 'C:\Users\gb\projects\demo_dual_axis_20261002'
$ownRawRoot = [IO.Path]::GetFullPath('D:\Program Files\UserCache\gb\ReID2-experiment-artifacts')
$archiveRoot = [IO.Path]::GetFullPath('E:\ReID2-experiment-artifacts\closed_20261007')
$planFile = Join-Path $projectPath 'results\preflight\own_D_closed_G100_frequency49_to_E_plan_20261007.json'
$actualFile = Join-Path $projectPath 'results\preflight\own_D_closed_G100_frequency49_to_E_actual_20261007.json'
if (Test-Path -LiteralPath $actualFile) { throw 'This relocation has already run' }
$plan = Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json
if ($plan.files -ne 49 -or $plan.status -ne 'CLOSED_G100_FREQUENCY49_RAW_RELOCATION_READY_NOT_EXECUTED') { throw 'Unexpected relocation plan' }
$sourceReceipt = Join-Path $projectPath $plan.source_receipt
if ((Get-FileHash -LiteralPath $sourceReceipt -Algorithm SHA256).Hash.ToLower() -ne $plan.source_receipt_sha256) { throw 'Historical receipt changed' }
$freeBefore = (Get-PSDrive -Name D).Free
$destFree = (Get-PSDrive -Name E).Free
if ($destFree -lt ($plan.bytes + 2GB)) { throw 'E archive space is insufficient' }
$moved = @()
foreach ($record in $plan.records) {
    $sourcePath = [IO.Path]::GetFullPath($record.source)
    $destinationPath = [IO.Path]::GetFullPath($record.destination)
    if (-not $sourcePath.StartsWith($ownRawRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Source outside own D archive' }
    if (-not $destinationPath.StartsWith($archiveRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Destination outside declared E archive' }
    if ([IO.Path]::GetFileName($sourcePath) -ne 'raw.npz' -or (Test-Path -LiteralPath $destinationPath)) { throw 'Unexpected raw target' }
    if ((Get-Item -LiteralPath $sourcePath).Length -ne $record.bytes -or (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLower() -ne $record.sha256) { throw 'Source differs from original receipt' }
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destinationPath)) -Force | Out-Null
    Copy-Item -LiteralPath $sourcePath -Destination $destinationPath
    if ((Get-Item -LiteralPath $destinationPath).Length -ne $record.bytes -or (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash.ToLower() -ne $record.sha256) { throw 'Archive differs; D retained' }
    Remove-Item -LiteralPath $sourcePath
    if (Test-Path -LiteralPath $sourcePath) { throw 'Duplicate source remains' }
    $moved += $record
    [pscustomobject]@{completed=$moved.Count;total=49;bytes=($moved | Measure-Object -Property bytes -Sum).Sum;last=$record.key} | ConvertTo-Json -Compress
}
if ($moved.Count -ne 49 -or ($moved | Measure-Object -Property bytes -Sum).Sum -ne $plan.bytes) { throw 'Incomplete relocation' }
[pscustomobject]@{status='ACTUAL_CLOSED_G100_FREQUENCY49_RAW_RELOCATED_D_TO_E_SIZE_SHA_VERIFIED';finished=(Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz');files=49;bytes=$plan.bytes;records=$moved;lost_raw_files=0;deleted_weights=0;new_neural_calls=0;D_free_before=$freeBefore;D_free_after=(Get-PSDrive -Name D).Free;E_free_after=(Get-PSDrive -Name E).Free;limits=$plan.limits} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $actualFile -Encoding utf8
