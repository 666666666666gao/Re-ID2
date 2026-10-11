$ErrorActionPreference = 'Stop'
$taskProject = 'C:/Users/gb/projects/demo_dual_axis_20261002'
$taskNormalPath = "$taskProject/results/preflight/r201p_auxiliary_balance_actual_session_20261010.json"
$taskNormal = Get-Content -LiteralPath $taskNormalPath -Raw | ConvertFrom-Json -DateKind String
if ($taskNormal.status -ne 'ACTUAL_P_DATASET_NATIVE6_PASSED_FULL50_RUNNING' -or $taskNormal.current_dataset -notin @('MSVR310','RGBNT100') -or $taskNormal.receiver_pid -ne 14232) { throw 'Original P resume2 normal owner is not in the declared running phase.' }
$taskOwnerId = [int]$taskNormal.receiver_pid
$taskInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $taskOwnerId"
if ($null -eq $taskInfo -or $taskInfo.CommandLine -notlike '*r201p_auxiliary_balance_resume2_deploy_20261011.py*') { throw 'Original P normal owner is not verified; do not register another pipeline.' }
$taskProcess = Get-Process -Id $taskOwnerId
$null = $taskProcess.Handle
$taskBirth = $taskProcess.StartTime.ToUniversalTime().Ticks
$taskQueuePath = "$taskProject/results/preflight/r201p_paper6_after_resume2_queue_actual_20261011.json"
if (Test-Path -LiteralPath $taskQueuePath) { throw 'P normal-to-paper-six queue already registered.' }
if (Test-Path -LiteralPath "$taskProject/results/preflight/r201p_paper_missing6_actual_session_20261010.json") { throw 'P paper-six has already started.' }
$taskReview = Get-Content -LiteralPath "$taskProject/results/preflight/r201p_post_resume2_normal_queue_source_review_20261011.json" -Raw | ConvertFrom-Json -DateKind String
if ($taskReview.status -ne 'PASS' -or $taskReview.blocking_findings.Count -ne 0) { throw 'P paper-six queue source review did not pass.' }
foreach ($taskMap in @($taskReview.sources_sha256,$taskReview.directly_reused_sources_sha256)) {
    foreach ($taskProperty in $taskMap.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath "$taskProject/$($taskProperty.Name)" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskProperty.Value) { throw 'Reviewed queue or reused source bytes changed.' }
    }
}
$taskQueue = [ordered]@{
    status = 'ACTUAL_LOCAL_P_PIPELINE_WAITING_ORIGINAL_RESUME2_NORMAL_OWNER_EXIT'
    registered_at = (Get-Date).ToString('o')
    queue_pid = $PID
    original_normal_owner_pid = $taskOwnerId
    original_normal_owner_birth_utc_ticks = $taskBirth
    wait_seconds_per_interval = 300
    normal_training_invocations = 0
    paper6_deploy_invocations = 0
    new_neural_calls = 0
    new_optimizer_updates = 0
    new_49_evaluation = $false
    checkpoint_writes = 0
    neural_server = '2026'
    physical_gpus = @(2,3)
    maximum_concurrent_neural_models = 2
    reviewed_sources_exact = $true
}
function Write-TaskQueue {
    [IO.File]::WriteAllText($taskQueuePath, ($taskQueue | ConvertTo-Json -Depth 8) + "`n", [Text.UTF8Encoding]::new($false))
}
Write-TaskQueue
Write-Output ($taskQueue | ConvertTo-Json -Depth 8 -Compress)
do { $taskExited = $taskProcess.WaitForExit(300000) } while (-not $taskExited)
$taskClosed = Get-Content -LiteralPath $taskNormalPath -Raw | ConvertFrom-Json -DateKind String
if ($taskProcess.ExitCode -ne 0 -or $taskClosed.status -ne 'ACTUAL_P_TWO_WEAK_NORMAL_DATASETS_FOUR_FULL50_GT_RAW_COMPLETE' -or $taskClosed.exit_code -ne 0 -or $taskClosed.native_updates -ne 12 -or $taskClosed.newly_executed_native_updates -ne 9 -or $taskClosed.reused_native_updates -ne 3 -or $taskClosed.successful_updates -ne 15126 -or $taskClosed.additional_epochs -ne 200) { throw 'Original P normal owner did not close all declared work successfully.' }
$taskQueue.status = 'ACTUAL_P_NORMAL_CLOSED_PAPER6_DEPLOY_INVOKED_ONCE'
$taskQueue.normal_closed_at = (Get-Date).ToString('o')
$taskQueue.paper6_deploy_invocations = 1
$taskQueue.Remove('new_neural_calls')
Write-TaskQueue
& 'C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe' -X utf8 -B -S "$taskProject/results/preflight/r201p_paper_missing6_deploy_20261010.py"
if ($LASTEXITCODE -ne 0) { throw 'P paper-six driver exited nonzero; preserve primary evidence and do not retry unchanged.' }
$taskPaper = Get-Content -LiteralPath "$taskProject/results/preflight/r201p_paper_missing6_actual_session_20261010.json" -Raw | ConvertFrom-Json -DateKind String
if ($taskPaper.status -ne 'ACTUAL_P_ALL_DECLARED_FIXED_BEST_PAPER6_GT_RAW_CPU_COMPLETE_K201_RETAINED' -or $taskPaper.new_full_conditions -ne 24 -or $taskPaper.new_full_state_cases -ne 96 -or $taskPaper.new_optimizer_updates -ne 0 -or $taskPaper.new_49_evaluation) { throw 'P paper-six work has not closed with its declared scope.' }
$taskQueue.status = 'ACTUAL_P_NORMAL_PAPER6_ORIGINAL_QUEUE_COMPLETE'
$taskQueue.completed_at = (Get-Date).ToString('o')
$taskQueue.exit_code = 0
Write-TaskQueue
Write-Output ($taskQueue | ConvertTo-Json -Depth 8)
