"""Use the public-identity builder with the unchanged six-state diagnostics."""
from run_full_official_public_outlet_identity_experiment import builder
import diagnose_full_official_control_states as diagnosis


if __name__ == '__main__':
    diagnosis.build = builder
    diagnosis.main()
