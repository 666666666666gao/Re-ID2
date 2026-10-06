$ErrorActionPreference = 'Stop'
$base = 'C:/Users/gb/projects/demo_dual_axis_20261002'
$pf = Join-Path $base 'results/preflight'
$proof = Join-Path $pf 'own_D_G_other_native_duplicate_retirement_actual_20261006.json'
if (Test-Path -LiteralPath $proof) { throw 'This duplicate retirement already has a receipt' }
$native = Get-Content -LiteralPath (Join-Path $pf 'r201g_other_two_missing49_stream_native_actual_session_20261006.json') -Raw | ConvertFrom-Json -AsHashtable
$full = Get-Content -LiteralPath (Join-Path $pf 'r201g_other_two_missing49_stream_full_actual_session_20261006.json') -Raw | ConvertFrom-Json -AsHashtable
if ($native.exit_code -ne 0 -or $full.exit_code -ne 0 -or $native.raw.Count -ne 4 -or $full.raw.Count -ne 196 -or -not $full.restored_copies_cleared) { throw 'Original native/full closeout is not accepted' }
$archive = (Resolve-Path -LiteralPath 'D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201g_other_two_missing49_stream_20261006').Path
$nativeRoot = [System.IO.Path]::GetFullPath((Join-Path $archive 'native')) + [System.IO.Path]::DirectorySeparatorChar
$fullRoot = [System.IO.Path]::GetFullPath((Join-Path $archive 'full')) + [System.IO.Path]::DirectorySeparatorChar
$qualified = @()
foreach ($key in $native.raw.Keys) {
    $old = $native.raw[$key]
    $kept = $full.raw[$key]
    $source = (Resolve-Path -LiteralPath $old.local).Path
    $retained = (Resolve-Path -LiteralPath $kept.local).Path
    if (-not $source.StartsWith($nativeRoot,[System.StringComparison]::OrdinalIgnoreCase) -or -not $retained.StartsWith($fullRoot,[System.StringComparison]::OrdinalIgnoreCase)) { throw 'Raw path outside the exact own native/full scope' }
    if ($old.file.bytes -ne $kept.file.bytes -or $old.file.sha256 -ne $kept.file.sha256) { throw 'Native and full receipt arrays are not byte-identical' }
    if ((Get-Item -LiteralPath $source).Length -ne $old.file.bytes -or (Get-Item -LiteralPath $retained).Length -ne $kept.file.bytes) { throw 'Current raw size differs from the original receipt' }
    $shaOld = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLower()
    $shaKept = (Get-FileHash -LiteralPath $retained -Algorithm SHA256).Hash.ToLower()
    if ($shaOld -ne $old.file.sha256 -or $shaKept -ne $shaOld) { throw 'Current raw SHA differs or files are not exact duplicates' }
    $qualified += [ordered]@{condition=$key;removed_native=$source;retained_full=$retained;bytes=$old.file.bytes;sha256=$shaKept}
}
if ($qualified.Count -ne 4) { throw 'Expected exactly four closed duplicate native arrays' }
$before = (Get-PSDrive D).Free
foreach ($row in $qualified) {
    Remove-Item -LiteralPath $row.removed_native
    if (Test-Path -LiteralPath $row.removed_native) { throw 'Native duplicate still exists' }
    if ((Get-Item -LiteralPath $row.retained_full).Length -ne $row.bytes -or (Get-FileHash -LiteralPath $row.retained_full -Algorithm SHA256).Hash.ToLower() -ne $row.sha256) { throw 'Canonical full raw changed during duplicate removal' }
}
$record = [ordered]@{status='ACTUAL_FOUR_CLOSED_NATIVE_DUPLICATES_REMOVED_FORMAL196_RETAINED';verified_at=(Get-Date -Format o);files_removed=4;bytes_removed=($qualified | ForEach-Object { $_.bytes } | Measure-Object -Sum).Sum;D_free_before_bytes=$before;D_free_after_bytes=(Get-PSDrive D).Free;lost_unique_raw_files=0;weights_removed=0;NN_calls=0;records=$qualified;note='Only four exact native/full normal duplicates removed after original full196 exit0. Original native receipt remains historical; retained canonical full location is recorded here. All196 formal raw, native text, selected normal arrays and necessary best weights remain.'}
[System.IO.File]::WriteAllText($proof,($record | ConvertTo-Json -Depth 8)+[Environment]::NewLine,[System.Text.UTF8Encoding]::new($false))
[ordered]@{status=$record.status;files=$record.files_removed;bytes=$record.bytes_removed;D_free_bytes=$record.D_free_after_bytes} | ConvertTo-Json
