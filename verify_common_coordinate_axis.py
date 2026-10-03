"""Inherited availability contracts plus private invariance and old-frequency parity."""
from common_coordinate_axis import build
from verify_shared_identity_axis import main


if __name__ == '__main__':
    main(build, common_increments=True)
