$ErrorActionPreference = 'Stop'
$taskProject = 'C:\Users\gb\projects\demo_dual_axis_20261002'
$taskArchive = (Resolve-Path -LiteralPath 'D:\Program Files\UserCache\gb\ReID2-experiment-artifacts').Path
$taskProof = Join-Path $taskProject 'results\preflight\own_D_three_closed_shared_weights_retired_actual_20261007.json'
if (Test-Path -LiteralPath $taskProof) { throw 'Retirement already recorded' }
$taskCandidates = @(
    @{relative='retired_weaker_shared_baselines_20261005\runs\full_official_baselines_20261004\training\RGBNT201_demo_shared_s42\best.pth';dataset='RGBNT201';tag='full_official_baselines_20261004';name='RGBNT201_demo_shared_s42';receipt='retired_weaker_shared_baselines_20261005.json';amp_skipped_steps=0},
    @{relative='retired_weaker_shared_baselines_20261005\runs\full_official_baselines_20261004\training\RGBNT100_demo_shared_s42\best.pth';dataset='RGBNT100';tag='full_official_baselines_20261004';name='RGBNT100_demo_shared_s42';receipt='retired_weaker_shared_baselines_20261005.json';amp_skipped_steps=1},
    @{relative='retired_rgbnt201_negative_weights_20261005\runs\rgbnt201_original_anchor_r201a_20261005\training\RGBNT201_original_shared_fixed_s42\best.pth';dataset='RGBNT201';tag='rgbnt201_original_anchor_r201a_20261005';name='RGBNT201_original_shared_fixed_s42';receipt='rgbnt201_retired_negative_weights_20261005.json';amp_skipped_steps=0}
)
$taskSources = @{}
foreach ($taskReviewName in @('r201i_expert_seeds_source_review_20261006.json','r201i_missing49_source_review_20261007.json')) {
    $taskReview = Get-Content -LiteralPath (Join-Path $taskProject "results\preflight\$taskReviewName") -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($taskReview.status -ne 'PASS') { throw 'Current queue source review is not PASS' }
    foreach ($taskHashMap in @($taskReview.sources_sha256,$taskReview.directly_reused_sources_sha256)) {
        foreach ($taskProperty in $taskHashMap.PSObject.Properties) {
            $taskHash = (Get-FileHash -LiteralPath (Join-Path $taskProject $taskProperty.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($taskHash -ne $taskProperty.Value) { throw "Queue source changed: $($taskProperty.Name)" }
            $taskSources[$taskProperty.Name] = $taskHash
        }
    }
}
$taskCurrentAnchors = @()
foreach ($taskDataset in @('RGBNT201','RGBNT100','MSVR310')) {
    foreach ($taskVariant in @('frequency_shared','axis_shared')) {
        $taskRunPath = Join-Path $taskProject "results\r201i_primary_margin_20261006\$taskDataset\training\$($taskDataset)_r201i_$($taskVariant)_s42\result.json"
        $taskCurrentRun = Get-Content -LiteralPath $taskRunPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($taskCurrentRun.status -ne 'COMPLETE' -or $taskCurrentRun.epochs -ne 50 -or -not $taskCurrentRun.anchor.original_run.EndsWith("/$($taskDataset)_demo_s42")) { throw 'Current I anchor differs from protected original DeMo' }
        $taskCurrentAnchors += $taskCurrentRun.anchor.original_run
    }
}
$taskVerified = @()
$taskMetaFields = @('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')
foreach ($taskCandidate in $taskCandidates) {
    $taskPath = (Resolve-Path -LiteralPath (Join-Path $taskArchive $taskCandidate.relative)).Path
    if (-not $taskPath.StartsWith($taskArchive + '\',[StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetExtension($taskPath) -ne '.pth') { throw 'Deletion target outside exact owned archive scope' }
    $taskReceipt = Get-Content -LiteralPath (Join-Path $taskProject "results\preflight\$($taskCandidate.receipt)") -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskRecorded = @($taskReceipt.archives.PSObject.Properties | Where-Object {$_.Value.local -eq $taskPath})
    if ($taskRecorded.Count -ne 1) { throw 'Target is not uniquely identified in prior verified retirement receipt' }
    $taskEntry = $taskRecorded[0].Value
    $taskFile = Get-Item -LiteralPath $taskPath
    $taskHash = (Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskFile.Length -ne $taskEntry.file.bytes -or $taskHash -ne $taskEntry.file.sha256) { throw 'Archived weight size/SHA differs' }
    $taskTraining = Join-Path $taskProject "results\$($taskCandidate.tag)\training\$($taskCandidate.name)"
    $taskFrozen = Join-Path $taskProject "results\$($taskCandidate.tag)\frozen49\$($taskCandidate.name)"
    $taskRun = Get-Content -LiteralPath (Join-Path $taskTraining 'result.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskEval = Get-Content -LiteralPath (Join-Path $taskFrozen 'result.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskExit = Get-Content -LiteralPath ($taskFrozen + '_exit.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($taskRun.status -ne 'COMPLETE' -or $taskRun.epochs -ne 50 -or $taskRun.amp_skipped_steps -ne $taskCandidate.amp_skipped_steps -or $taskRun.training_heldout_identities -ne 0 -or $taskEval.status -ne 'COMPLETE' -or @($taskEval.measurements.PSObject.Properties).Count -ne 49 -or $taskExit.exit_code -ne 0 -or @(Get-ChildItem -LiteralPath $taskFrozen -Filter 'q_*_g_*.csv' -File).Count -ne 49) { throw 'Training/full49 closure not proven' }
    $taskRefFolder = Join-Path $taskProject "results\full_official_baselines_20261004\training\$($taskCandidate.dataset)_demo_s42"
    $taskRefRun = Get-Content -LiteralPath (Join-Path $taskRefFolder 'result.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskRows = @(Import-Csv -LiteralPath (Join-Path $taskTraining 'best_per_query.csv') -Encoding UTF8)
    $taskRefRows = @(Import-Csv -LiteralPath (Join-Path $taskRefFolder 'best_per_query.csv') -Encoding UTF8)
    if ($taskRefRun.status -ne 'COMPLETE' -or $taskRefRun.epochs -ne 50 -or $taskRows.Count -ne $taskRun.query_records -or $taskRows.Count -ne $taskRefRows.Count) { throw 'Normal query scope differs' }
    for ($taskIndex=0; $taskIndex -lt $taskRows.Count; $taskIndex++) {
        foreach ($taskField in $taskMetaFields) {
            if ($taskRows[$taskIndex].$taskField -ne $taskRefRows[$taskIndex].$taskField) { throw "GT metadata differs at $taskIndex/$taskField" }
        }
    }
    $taskCandidateMAP = 100 * ($taskRows | Measure-Object -Property AP -Sum).Sum / $taskRows.Count
    $taskCandidateR1 = 100 * ($taskRows | Measure-Object -Property 'Rank-1' -Sum).Sum / $taskRows.Count
    $taskRefMAP = 100 * ($taskRefRows | Measure-Object -Property AP -Sum).Sum / $taskRefRows.Count
    $taskRefR1 = 100 * ($taskRefRows | Measure-Object -Property 'Rank-1' -Sum).Sum / $taskRefRows.Count
    if ([Math]::Abs($taskCandidateMAP-$taskRun.full_metrics.mAP) -gt 1e-8 -or [Math]::Abs($taskCandidateR1-$taskRun.full_metrics.'Rank-1') -gt 1e-8 -or [Math]::Abs($taskRefMAP-$taskRefRun.full_metrics.mAP) -gt 1e-8 -or [Math]::Abs($taskRefR1-$taskRefRun.full_metrics.'Rank-1') -gt 1e-8 -or $taskRefMAP -le $taskCandidateMAP -or $taskRefR1 -lt $taskCandidateR1) { throw 'Retained original does not dominate the candidate normal primary metrics' }
    $taskVerified += [ordered]@{path=$taskPath;bytes=$taskFile.Length;sha256=$taskHash;name=$taskCandidate.name;dataset=$taskCandidate.dataset;normal_mAP=$taskCandidateMAP;normal_Rank1=$taskCandidateR1;retained_original_mAP=$taskRefMAP;retained_original_Rank1=$taskRefR1;queries=$taskRows.Count;full49_exit=0;prior_receipt=$taskCandidate.receipt;amp_skipped_steps=$taskRun.amp_skipped_steps;actual_optimizer_steps=$taskRun.optimizer_steps}
}
$taskBefore = (Get-PSDrive -Name D).Free
foreach ($taskItem in $taskVerified) { Remove-Item -LiteralPath $taskItem.path -Force }
foreach ($taskItem in $taskVerified) { if (Test-Path -LiteralPath $taskItem.path) { throw 'Target still exists after retirement' } }
$taskAfter = (Get-PSDrive -Name D).Free
$taskResult = [ordered]@{status='ACTUAL_THREE_ALREADY_REMOTE_RETIRED_LOCAL_SHARED_WEIGHTS_REMOVED';verified_at=(Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz');removed_files=3;removed_bytes=($taskVerified | Measure-Object -Property bytes -Sum).Sum;D_free_before=$taskBefore;D_free_after=$taskAfter;files=$taskVerified;current_I_anchors=$taskCurrentAnchors;current_queue_source_hashes=$taskSources;new_neural_calls=0;new_optimizer_updates=0;remote_calls=0;raw_files_deleted=0;limits='Only these three old local checkpoint binaries are retired. Full50/full49 text, curves, GT and all unique distances remain. Exact old NN replay now requires retraining. Normal primary metric dominance does not claim all six/missing dominance or a fair algorithm-effect estimate.'}
[IO.File]::WriteAllText($taskProof,($taskResult | ConvertTo-Json -Depth 8) + [Environment]::NewLine,[Text.UTF8Encoding]::new($false))
$taskResult | Select-Object status,removed_files,removed_bytes,D_free_before,D_free_after,new_neural_calls,raw_files_deleted | ConvertTo-Json
