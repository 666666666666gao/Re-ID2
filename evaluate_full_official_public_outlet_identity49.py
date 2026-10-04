"""Use the public-identity builder with the unchanged full official evaluator."""
from run_full_official_public_outlet_identity_experiment import builder
import evaluate_full_official49 as evaluation


if __name__ == '__main__':
    evaluation.build = builder
    evaluation.main()
