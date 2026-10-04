"""Use the graph builder for the unchanged enhanced-model six-state diagnostics."""
from run_full_official_availability_graph_experiment import builder
import diagnose_full_official_control_states as diagnosis


if __name__ == '__main__':
    diagnosis.build = builder
    diagnosis.main()
