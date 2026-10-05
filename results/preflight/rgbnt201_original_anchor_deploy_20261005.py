"""Deploy R201A once per phase, after source review and the actual native gate."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, OPTIONS, command, remote_python
from results.preflight.rgbnt201_original_anchor_stream_20261005 import archive_session


REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
PREFLIGHT = REMOTE + '/runs/rgbnt201_original_anchor_preflight_fp32_fusion_20261005'
TRAIN = REMOTE + '/runs/rgbnt201_original_anchor_r201a_20261005'
ANCHOR = REMOTE + '/runs/full_official_baselines_20261004/training/RGBNT201_demo_s42'
M8 = REMOTE + '/runs/full_official_frozen_public_identity_m8_20261005'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'train'), required=True)
    args = parser.parse_args()
    pf = PROJECT / 'results/preflight'
    plan = json.loads((pf / 'rgbnt201_priority_plan_20261005.json').read_text(encoding='utf-8'))
    review = json.loads((pf / 'rgbnt201_original_anchor_source_review_20261005.json').read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    sources = plan['R201A']['sources_sha256']
    assert review['sources_sha256'] == sources
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest for name, digest in sources.items())
    root = PREFLIGHT if args.mode == 'preflight' else TRAIN
    proof = pf / ('rgbnt201_original_anchor_' + args.mode + '_session_20261005.json')
    assert not proof.exists()
    if args.mode == 'train':
        actual = json.loads((pf / 'rgbnt201_original_anchor_actual_preflight_20261005.json').read_text(encoding='utf-8'))
        assert actual['result']['status'] == 'PASS_RGBNT201_ORIGINAL_ANCHOR_NATIVE2_SMOKE6'
        assert actual['result']['actual_native_updates'] == 2 and actual['result']['actual_smoke_updates'] == 6
        assert actual['result']['amp_skipped_steps'] == 0 and actual['exit_code'] == 0
    parents = review['directly_reused_sources_sha256']
    ready = json.loads(remote_python('2026', f'''import hashlib,json
from pathlib import Path
project=Path({REMOTE!r})
assert not Path({root!r}).exists()
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==digest for name,digest in {parents!r}.items())
anchor=json.loads((Path({ANCHOR!r})/'result.json').read_text())
assert anchor['status']=='COMPLETE' and anchor['arguments']['variant']=='demo'
assert anchor['arguments']['dataset']=='RGBNT201' and anchor['epochs']==50
assert anchor['optimizer_steps']==2647 and anchor['amp_skipped_steps']==0
assert (Path({M8!r})/'controller_train_launch.json').exists()
print(json.dumps({{'anchor_best':anchor['best']['epoch'],'existing_M8_preserved':True}}))'''))
    for file in sources:
        if not file.startswith('results/'):
            command(['scp', *OPTIONS, str(PROJECT / file), '2026:' + REMOTE + '/' + file])
    deployed = {file: digest for file, digest in sources.items() if not file.startswith('results/')}
    assert json.loads(remote_python('2026', f'''import hashlib,json
from pathlib import Path
root=Path({REMOTE!r})
assert all(hashlib.sha256((root/file).read_bytes()).hexdigest()==digest for file,digest in {deployed!r}.items())
print(json.dumps({deployed!r}))''')) == deployed
    argv = [PYTHON, '-u', 'launch_rgbnt201_original_anchor.py', '--mode', args.mode,
        '--data-root', '/data/gaob/Re-ID/dataset', '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-run-dir', ANCHOR, '--m8-root', M8, '--output', root]
    print('R201A_REVIEWED_DEPLOYMENT', args.mode, json.dumps(ready), flush=True)
    if args.mode == 'train':
        argv += ['--preflight-root', PREFLIGHT]
        archive_session(argv, Path(TRAIN), Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/rgbnt201_original_anchor_r201a_20261005'), proof)
    else:
        process = subprocess.Popen(['ssh', *OPTIONS, '2026', 'cd ' + shlex.quote(REMOTE) + ' && ' + shlex.join(argv)])
        assert process.wait() == 0
        result = json.loads(remote_python('2026', f'''import json
from pathlib import Path
root=Path({PREFLIGHT!r})
assert not list(root.rglob('*.pth'))
print((root/'preflight_result.json').read_text())'''))
        assert result['status'] == 'PASS_RGBNT201_ORIGINAL_ANCHOR_NATIVE2_SMOKE6'
        actual = dict(status='ACTUAL_R201A_NATIVE2_SMOKE6_PASS_NOT_FULL50', result=result,
                      exit_code=0, root=PREFLIGHT, new_weight_files=0, sources_sha256=sources)
        (pf / 'rgbnt201_original_anchor_actual_preflight_20261005.json').write_text(json.dumps(actual, indent=2) + '\n', encoding='utf-8')
        proof.write_text(json.dumps(dict(status='R201A_PREFLIGHT_SESSION_EXIT0', mode=args.mode, exit_code=0), indent=2) + '\n', encoding='utf-8')
        print('R201A_ACTUAL_PREFLIGHT_PASS', flush=True)


if __name__ == '__main__':
    main()
