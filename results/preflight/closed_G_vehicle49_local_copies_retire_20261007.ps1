$ErrorActionPreference = 'Stop'
$project = 'C:\Users\gb\projects\demo_dual_axis_20261002'
$pf = Join-Path $project 'results\preflight'
$review = Get-Content -LiteralPath (Join-Path $pf 'closed_G_vehicle49_archive27_source_review_20261007.json') -Raw | ConvertFrom-Json
if ($review.status -ne 'PASS' -or @($review.blocking_findings).Count -ne 0) { throw 'Source review not PASS' }
$scriptRelative = 'results/preflight/closed_G_vehicle49_local_copies_retire_20261007.ps1'
$reviewedSHA = $review.sources_sha256.$scriptRelative
if ((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $reviewedSHA) { throw 'Local retirement source differs from review' }
$ackPath = Join-Path $pf 'closed_G_vehicle49_archive27_ack_actual_20261007.json'
$ack = Get-Content -LiteralPath $ackPath -Raw | ConvertFrom-Json
if ($ack.status -ne 'ACTUAL_ALL196_CLOSED_G_RAW_ARCHIVE27_SIZE_SHA_ACK_LOCAL_NOT_REMOVED' -or $ack.files -ne 196 -or $ack.bytes -ne 15477041156) { throw 'All196 remote size/SHA ACK missing' }
if ($ack.remote_ack.status -ne 'ALL196_CLOSED_G_VEHICLE49_SIZE_SHA_VERIFIED' -or $ack.remote_ack.files -ne 196 -or $ack.remote_ack.bytes -ne $ack.bytes) { throw 'Remote ACK mismatch' }
if ($ack.local_files_removed -ne 0 -or $ack.unique_raw_lost -ne 0 -or $ack.remote_host -ne '2027') { throw 'Unexpected archive state' }
$remoteRoot = '/data/gb/Re-ID/DeMo-DualAxis/archives/closed_G_vehicle49_20261007'
if ($ack.remote_root -ne $remoteRoot -or $ack.remote_ack.manifest -ne ($remoteRoot + '/archive_manifest.json')) { throw 'Wrong archive root' }
$roots = @(
    'D:\Program Files\UserCache\gb\ReID2-experiment-artifacts\r201g_other_two_missing49_stream_20261006',
    'E:\ReID2-experiment-artifacts\closed_20261007\r201g_other_two_missing49_stream_20261006'
)
$proofPath = Join-Path $pf 'closed_G_vehicle49_local_copies_retired_actual_20261007.json'
if (Test-Path -LiteralPath $proofPath) { throw 'Local retirement already recorded' }
$rows = @($ack.records)
if ($rows.Count -ne 196 -or @($rows.source | Sort-Object -Unique).Count -ne 196) { throw 'Wrong retirement inventory' }
foreach ($row in $rows) {
    $absolute = [System.IO.Path]::GetFullPath($row.source)
    $scoped = $false
    foreach ($root in $roots) {
        if ($absolute.StartsWith($root + '\', [System.StringComparison]::OrdinalIgnoreCase)) { $scoped = $true }
    }
    if (-not $scoped -or [System.IO.Path]::GetFileName($absolute) -ne 'raw.npz') { throw 'Target outside exact owned closed G roots' }
    if ($row.destination -ne ($remoteRoot + '/full/' + $row.key + '/raw.npz')) { throw 'Remote destination differs from checked inventory' }
    if ((Get-Item -LiteralPath $absolute).Length -ne $row.bytes) { throw 'Local source size changed' }
    if ((Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) { throw 'Local source SHA changed' }
}
$freeBefore = @{ D = (Get-PSDrive -Name D).Free; E = (Get-PSDrive -Name E).Free }
foreach ($row in $rows) { Remove-Item -LiteralPath ([System.IO.Path]::GetFullPath($row.source)) -Force }
foreach ($row in $rows) { if (Test-Path -LiteralPath $row.source) { throw 'Local source remains' } }
$value = @{
    status = 'ACTUAL_CLOSED_G_VEHICLE49_LOCAL_COPIES_RETIRED_ALL_RAW_PRESERVED_ON27'
    finished = (Get-Date).ToString('o')
    files = 196
    bytes = 15477041156
    copied_bytes_D = 8055249136
    copied_bytes_E = 7421792020
    ack_sha256 = (Get-FileHash -LiteralPath $ackPath -Algorithm SHA256).Hash.ToLowerInvariant()
    remote_root = $remoteRoot
    remote_manifest_sha256 = $ack.remote_ack.manifest_sha256
    records = $rows
    unique_raw_lost = 0
    new_neural_calls = 0
    new_optimizer_updates = 0
    free_before = $freeBefore
    free_after = @{ D = (Get-PSDrive -Name D).Free; E = (Get-PSDrive -Name E).Free }
    limits = 'Inactive closed G vehicle49 only. Exact raw remains on27; old local receipt paths are historical and require explicit restore from the archive. Current I/J/baseline/G201 raw, selected weights, metrics, curves, GT and all user other projects are unchanged. No recursive directory removal.'
}
[System.IO.File]::WriteAllText($proofPath, (($value | ConvertTo-Json -Depth 10) + "`n"), [System.Text.UTF8Encoding]::new($false))
Write-Output ('ACTUAL_CLOSED_G_LOCAL_COPIES_RETIRED_RAW_PRESERVED_ON27 ' + $value.bytes)
