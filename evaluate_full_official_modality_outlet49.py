"""Strict M4 builder; unchanged complete official all49 frozen evaluation."""
from modality_outlet_alignment_axis import build
import evaluate_full_official49 as evaluation


if __name__ == '__main__':
    evaluation.build = build
    evaluation.main()
