"""All49 evaluator with the original-DeMo-initialized RGBNT201 builder."""
import evaluate_full_official49 as evaluator
from original_identity_anchor import build


if __name__ == '__main__':
    evaluator.build = build
    evaluator.main()
