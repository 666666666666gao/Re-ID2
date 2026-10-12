"""Explicit warm load of the strong shared-DeMo teacher into the existing 5120D model."""
import hashlib,json
from pathlib import Path
import torch
from official_training_data import full_records
from original_identity_anchor import IDENTITY_MODULES,assert_anchor_unchanged


def install(model,args,classes,cameras):
    assert args.dataset=='MSVR310' and args.seed==42 and model.freeze_identity_encoder
    anchor=Path(args.anchor_run_dir).parent/'MSVR310_demo_shared_s42'
    trained=json.loads((anchor/'result.json').read_text())
    assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['descriptor_dim']==5632
    assert trained['arguments']['dataset']=='MSVR310' and trained['arguments']['variant']=='demo_shared' and trained['arguments']['seed']==42
    assert trained['classes']==classes==155 and trained['camera_embeddings']==cameras
    assert trained['steps']==trained['optimizer_steps'] and trained['amp_skipped_steps']==trained['training_heldout_identities']==0
    assert json.loads((anchor.parent/(anchor.name+'_exit.json')).read_text())['exit_code']==0
    assert json.loads((anchor/'official_split_manifest.json').read_text())==full_records(args.data_root,'MSVR310')[-1]
    saved=torch.load(anchor/'best.pth',map_location='cpu',weights_only=True)
    # The trained 5632D fused head has a different input width. Keep the current
    # 5120D fused head's matched fresh initialization; every other teacher tensor
    # maps directly, with no partial or non-strict load.
    excluded={'fused_neck','fused_classifier'}
    selected={k:v for k,v in saved.items() if k.split('.')[0] not in excluded}
    current=model.state_dict();fresh={k:v.detach().cpu().clone() for k,v in current.items() if k not in selected}
    assert set(selected)<=set(current) and all(current[k].shape==v.shape and current[k].dtype==v.dtype for k,v in selected.items())
    current.update(selected);model.load_state_dict(current,strict=True)
    assert all(torch.equal(model.state_dict()[k].detach().cpu(),v) for k,v in selected.items())
    assert all(torch.equal(model.state_dict()[k].detach().cpu(),v) for k,v in fresh.items())
    model.anchor_encoder_state={k:v.clone() for k,v in selected.items() if k.split('.')[0] in IDENTITY_MODULES}
    model.anchor_record.update(original_run=str(anchor),original_best_epoch=trained['best']['epoch'],
        selected_anchor_full_metrics=trained['full_metrics'],
        teacher_variant='demo_shared',teacher_descriptor_dim=5632,deployed_descriptor_dim=5120,
        strong_teacher=dict(weight_sha256=hashlib.sha256((anchor/'best.pth').read_bytes()).hexdigest(),
            copied_tensors=len(selected),excluded_teacher_prefixes=sorted(excluded),fresh_student_tensors=len(fresh),
            private_reference='Same teacher private5120 block, not its full5632 .75/.25 score',
            limits='Full teacher49.426/69.036 is an existing baseline, not this5120 student score or expert gain. '
                'Warm private reference47.763432/65.482234 must be independently reloaded/checked before training. '
                'Only warm initialization changes from Y; no gradual unfreezing is implemented here.'))
    assert_anchor_unchanged(model)
    return model
