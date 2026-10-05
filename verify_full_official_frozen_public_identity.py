"""Run M7's actual native contract with only M8's build and update functions."""
import verify_full_official_frozen_anchor as parent
from run_full_official_frozen_public_identity_experiment import VARIANTS, build, step


if __name__ == '__main__':
    parent.VARIANTS = VARIANTS
    parent.build = build
    parent.step = step
    parent.main()
