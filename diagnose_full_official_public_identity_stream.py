"""Validate streaming against the completed M6 public-identity model."""
from run_full_official_public_outlet_identity_experiment import builder
import diagnose_full_official_streaming_states as diagnosis


if __name__ == '__main__':
    diagnosis.build = builder
    diagnosis.main()
