"""Use the recorded graph builder with the unchanged official frozen49 evaluator."""
from run_full_official_availability_graph_experiment import builder
import evaluate_full_official49 as evaluation


if __name__ == '__main__':
    evaluation.build = builder
    evaluation.main()
