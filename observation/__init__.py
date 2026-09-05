# OBSERVATION package -- MDMARD Phase 1
# Constructs the normalized state vector observed by the RL agent.
#
# Modules (to be created in Stage 2):
#   builder.py      : ObservationBuilder -- assembles the obs vector
#   normalizer.py   : Per-component normalization (min-max, clip)
#   space.py        : gymnasium.spaces.Box definition for obs space