"""Explicit M3a/M3b state builder for complete official frozen 49 evaluation."""
from independent_control_axis import build
import evaluate_full_official49 as evaluation


if __name__ == '__main__':
    evaluation.build = build
    evaluation.main()
