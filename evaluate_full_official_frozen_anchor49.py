"""The unchanged complete official evaluator with the common-anchor builder."""
from frozen_identity_anchor_axis import build
import evaluate_full_official49 as evaluation


if __name__ == '__main__':
    evaluation.build = build
    evaluation.main()
