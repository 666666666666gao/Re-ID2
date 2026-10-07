"""Fixed selected L K8 checkpoints through the unchanged full49 installed-GT path."""
import evaluate_identity_coordinate_missing49_stream as evaluator
from relation_local_identity_outlet import RelationLocalPIAxis
from run_r201l_uniform_k8 import REVISION,build as uniform_build,configuration

def build(args,cfg,classes,cameras):
    assert args.seed==42 and args.dataset in ('MSVR310','RGBNT100')
    assert cfg.DATALOADER.NUM_INSTANCE==8
    model=uniform_build(args,cfg,classes,cameras)
    assert type(model) is RelationLocalPIAxis and model.anchor_record['method']==REVISION
    return model

if __name__=='__main__':
    evaluator.configuration=configuration
    evaluator.build=build
    evaluator.main()
