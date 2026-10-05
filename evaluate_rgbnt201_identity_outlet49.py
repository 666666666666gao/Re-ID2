"""Entire official query/gallery all49 with the new saved original-coordinate builder."""
import evaluate_full_official49 as evaluator
from rgbnt201_identity_outlet import build


if __name__ == '__main__':
    evaluator.build = build
    evaluator.main()
