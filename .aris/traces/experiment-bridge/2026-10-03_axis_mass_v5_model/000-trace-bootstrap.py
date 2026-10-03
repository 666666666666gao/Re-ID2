import json
from pathlib import Path
from datetime import datetime, timezone

p = Path(__file__).resolve().parent
t = datetime.now(timezone.utc).isoformat()
meta = {'skill': 'experiment-bridge', 'run_id': p.name, 'started_at': t,
        'executor': 'codex', 'executor_model': 'gpt-6-astra', 'executor_family': 'openai',
        'review_independence': 'same-family', 'acceptance_status': 'provisional',
        'project_dir': str(p.parents[3]), 'reasoning_effort': 'max', 'fork_turns': 'none',
        'agent_id': '/root/review_axis_mass_v5_model',
        'config_provenance': 'Actual native review configuration supplied by delegating parent task.'}
request = {'call_number': 1, 'purpose': 'axis-mass-v5-model-source-review', 'timestamp': t,
           'tool': 'spawn_agent', 'model': 'gpt-6-astra', 'reasoning_effort': 'max', 'fork_turns': 'none',
           'files_referenced': ['mass_axis_collaboration.py', 'run_mass_experiment.py',
                                'axis_collaboration.py', 'scaled_axis_collaboration.py',
                                'run_experiment.py', 'results/preflight/axis_collaboration_v5_mass_plan.json',
                                'results/preflight/axis_collaboration_v4_route_semantics_diagnostic.json',
                                'results/preflight/axis_collaboration_v4_missing_development26_repaired_launch.json'],
           'prompt': (p / '000-task-request.txt').read_text(encoding='utf-8').rstrip('\n')}
for name, data in [('run.meta.json', meta), ('001-model-review.request.json', request)]:
    (p / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Trace request and metadata persisted.')
