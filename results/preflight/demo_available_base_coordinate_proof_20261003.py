import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path

project=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
tree=ast.parse((project/'dual_axis.py').read_text(encoding='utf-8'))
definition=next(node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='RELATIONS' for target in node.targets))
relations=ast.literal_eval(definition.value)
sets={'R':{0},'N':{1},'T':{2},'RN':{0,1},'RT':{0,2},'NT':{1,2},'RNT':{0,1,2}}
supports={name:sorted(value | {3+index for index,relation in enumerate(relations) if set(relation)<=value}) for name,value in sets.items()}
disjoint=[dict(query=q,gallery=g,query_blocks=supports[q],gallery_blocks=supports[g]) for q in sets for g in sets if not set(supports[q])&set(supports[g])]
assert len(disjoint)==12
result=dict(status='SOURCE_GEOMETRY_DEDUCTION_NOT_NEURAL_EVALUATION',observed_at=datetime.now().isoformat(timespec='seconds'),
            source_sha256={name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in ('dual_axis.py','availability_base_intervention.py','verify_availability_base_intervention.py','modeling/make_model.py','full_evaluation.py')},
            model='DeMo frozen descriptor has3 modality blocks plus7 relation blocks, each512 coordinates; current intervention zeros unavailable global/invalid relation blocks',
            supports=supports,disjoint_pairs=disjoint,
            implication='For these12/49 availability pairs, correctly masked fixed5120D DeMo descriptors have disjoint support and their inner product is identicallyzero, even if all surviving individual features are informative. Masking alone cannot create cross-set comparability.',
            limits='Deduction conditional on actual zero-coordinate tensor contracts, not a claim that frozen evaluation ran or a measured accuracy. V5 has a shared frequency tail and is not necessarily orthogonal in these pairs.',
            next='Use a learned shared identity coordinate path and asymmetric partial/full reference training, alongside source-specific private evidence. Match augmentation and capacity controls; all3+2 goal unchanged.')
path=project/'results/preflight/availability_base_coordinate_geometry.json';assert not path.exists()
path.write_bytes(json.dumps(result,indent=2).encode())
print('MASKED_DEMO_DISJOINT_COORDINATE_PAIRS',len(disjoint),flush=True)
