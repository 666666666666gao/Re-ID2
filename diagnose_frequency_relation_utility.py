"""The same read-only eight-stage protocol, now labeling the fresh M2 weights."""
import diagnose_trained_outlet_utility as utility


original_write = utility.ORIGINAL_WRITE


def write_result(path, value):
    value['scope'] = 'Frozen M2 original_mean fresh50; no optimizer or deployment changes in this diagnostic.'
    value['learning_revision'] = 'weight0.1 full-view actual F_pre -> PF normalized pair-distance preservation; other M1 training unchanged'
    original_write(path, value)


utility.ORIGINAL_WRITE = write_result

if __name__ == '__main__':
    utility.framework.main()
