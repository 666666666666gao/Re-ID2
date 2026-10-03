"""V12 fresh50: inherited V11 learning with a matched common-coordinate interface."""
from common_coordinate_axis import build
from run_shared_identity_experiment import main


if __name__ == '__main__':
    main(build,
         method_revision='common_coordinate_v12: same V11 backbone, expert access, FFT, routing, losses and partial training; all augmented M/F/I increments use the existing ordinary-frequency common-coordinate interface',
         metric='5632D: sqrt(.75) normalized unchanged private5120 + sqrt(.25) normalized common512 with M/F/I increments; V11 ordinary-frequency and DeMo behaviors are controls')
