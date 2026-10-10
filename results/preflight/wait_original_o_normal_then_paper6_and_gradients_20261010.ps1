$ErrorActionPreference = 'Stop'
$taskProject = 'C:/Users/gb/projects/demo_dual_axis_20261002'
$taskNormalPath = "$taskProject/results/preflight/r201o_same_state_contribution_actual_session_20261010.json"
$taskNormal = Get-Content -LiteralPath $taskNormalPath -Raw | ConvertFrom-Json -DateKind String
if ($taskNormal.status -ne 'ACTUAL_O_DATASET_NATIVE6_PASSED_FULL50_RUNNING' -or $taskNormal.current_dataset -ne 'RGBNT100' -or $taskNormal.receiver_pid -ne 40912) { throw 'Original O normal owner is not in the declared running phase.' }
$taskOwnerId = [int]$taskNormal.receiver_pid
$taskInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $taskOwnerId"
if ($null -eq $taskInfo -or $taskInfo.CommandLine -notlike '*r201o_same_state_deploy_20261010.py*') { throw 'Original O normal owner is not verified; do not register another pipeline.' }
$taskProcess = Get-Process -Id $taskOwnerId
$taskBirth = $taskProcess.StartTime.ToUniversalTime().Ticks
$taskQueuePath = "$taskProject/results/preflight/r201o_paper6_then_gradients_queue_actual_20261010.json"
if (Test-Path -LiteralPath $taskQueuePath) { throw 'O normal-to-paper-six-and-gradient queue already registered.' }
foreach ($taskAbsent in @('r201o_paper_missing6_actual_session_20261010.json','r201o_selected_task_gradients_actual_20261010.json')) {
    if (Test-Path -LiteralPath "$taskProject/results/preflight/$taskAbsent") { throw 'A declared O evaluation or diagnosis has already started.' }
}
$taskQueueReview = Get-Content -LiteralPath "$taskProject/results/preflight/r201o_post_normal_queue_source_review_20261010.json" -Raw | ConvertFrom-Json -DateKind String
if ($taskQueueReview.status -ne 'PASS' -or $taskQueueReview.blocking_findings.Count -ne 0) { throw 'O pipeline queue source review did not pass.' }
foreach ($taskMap in @($taskQueueReview.sources_sha256,$taskQueueReview.directly_reused_sources_sha256)) {
    foreach ($taskProperty in $taskMap.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath "$taskProject/$($taskProperty.Name)" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskProperty.Value) { throw 'Reviewed queue or reused source bytes changed.' }
    }
}
$taskQueue = [ordered]@{
    status = 'ACTUAL_LOCAL_O_PIPELINE_WAITING_ORIGINAL_NORMAL_OWNER_EXIT'
    registered_at = (Get-Date).ToString('o')
    queue_pid = $PID
    original_normal_owner_pid = $taskOwnerId
    original_normal_owner_birth_utc_ticks = $taskBirth
    wait_seconds_per_interval = 300
    normal_training_invocations = 0
    paper6_deploy_invocations = 0
    selected_gradient_deploy_invocations = 0
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
if ($taskClosed.status -ne 'ACTUAL_O_TWO_WEAK_NORMAL_DATASETS_FOUR_FULL50_GT_RAW_COMPLETE' -or $taskClosed.exit_code -ne 0 -or $taskClosed.native_updates -ne 12 -or $taskClosed.successful_updates -ne 15126 -or $taskClosed.additional_epochs -ne 200) { throw 'Original O normal owner did not close all declared work successfully.' }
$taskQueue.status = 'ACTUAL_O_NORMAL_CLOSED_PAPER6_DEPLOY_INVOKED_ONCE'
$taskQueue.normal_closed_at = (Get-Date).ToString('o')
$taskQueue.paper6_deploy_invocations = 1
$taskQueue.Remove('new_neural_calls')
Write-TaskQueue
& 'C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe' -X utf8 -B -S "$taskProject/results/preflight/r201o_paper_missing6_deploy_20261010.py"
if ($LASTEXITCODE -ne 0) { throw 'O paper-six driver exited nonzero; preserve primary evidence and do not retry unchanged.' }
$taskPaper = Get-Content -LiteralPath "$taskProject/results/preflight/r201o_paper_missing6_actual_session_20261010.json" -Raw | ConvertFrom-Json -DateKind String
if ($taskPaper.status -ne 'ACTUAL_O_ALL_DECLARED_FIXED_BEST_PAPER6_GT_RAW_CPU_COMPLETE_K201_RETAINED' -or $taskPaper.new_full_conditions -ne 24 -or $taskPaper.new_full_state_cases -ne 96 -or $taskPaper.new_49_evaluation) { throw 'O paper-six work has not closed with its declared scope.' }
$taskQueue.status = 'ACTUAL_O_PAPER6_CLOSED_SELECTED_GRADIENT_DEPLOY_INVOKED_ONCE'
$taskQueue.paper6_closed_at = (Get-Date).ToString('o')
$taskQueue.selected_gradient_deploy_invocations = 1
Write-TaskQueue
& 'C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe' -X utf8 -B -S "$taskProject/results/preflight/r201o_selected_gradients_deploy_20261010.py"
if ($LASTEXITCODE -ne 0) { throw 'O selected-gradient driver exited nonzero; preserve primary evidence and do not retry unchanged.' }
$taskGradient = Get-Content -LiteralPath "$taskProject/results/preflight/r201o_selected_task_gradients_actual_20261010.json" -Raw | ConvertFrom-Json -DateKind String
if ($taskGradient.status -ne 'ACTUAL_FOUR_SELECTED_O_TRAIN_ONLY_TASK_GRADIENT_DIAGNOSTICS_COMPLETE' -or $taskGradient.optimizer_updates -ne 0 -or $taskGradient.checkpoint_writes -ne 0) { throw 'O selected-gradient diagnosis has not closed successfully.' }
$taskSummary = Get-Content -LiteralPath "$taskProject/results/preflight/r201o_closed_selected_gradient_summary_actual_20261010.json" -Raw | ConvertFrom-Json -DateKind String
if ($taskSummary.status -ne 'ACTUAL_16_SELECTED_O_TRAIN_ONLY_GRADIENT_ROLE_READOUT_CPU_COMPLETE') { throw 'O selected-gradient CPU reduction is incomplete.' }
$taskQueue.status = 'ACTUAL_O_NORMAL_PAPER6_SELECTED_GRADIENT_ORIGINAL_QUEUE_COMPLETE'
$taskQueue.completed_at = (Get-Date).ToString('o')
$taskQueue.exit_code = 0
Write-TaskQueue
Write-Output ($taskQueue | ConvertTo-Json -Depth 8)
