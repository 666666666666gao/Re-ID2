"""Complete official states with the common-anchor builder and bounded raw archive."""
from frozen_identity_anchor_axis import build
import diagnose_full_official_streaming_states as diagnosis


if __name__ == '__main__':
    diagnosis.build = build
    diagnosis.main()
