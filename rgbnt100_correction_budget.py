"""Bind the existing geometric budget only in the isolated AM RGB process."""
from functools import partial

import bounded_identity_descriptor as descriptor
import run_r201n_bounded_identity_shift as bounded_runner

EPSILON=.20
assert descriptor.MAX_UNIT_SHIFT==bounded_runner.MAX_UNIT_SHIFT==.10
# Existing bound_descriptor already exposes epsilon; the installer and fixture
# resolve it through this module global. Explicit binding overrides its .10
# default without modifying any shared source or adding model parameters.
descriptor.bound_descriptor=partial(descriptor.bound_descriptor,epsilon=EPSILON)
descriptor.MAX_UNIT_SHIFT=EPSILON
bounded_runner.MAX_UNIT_SHIFT=EPSILON
