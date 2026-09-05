# UTILS package -- MDMARD Phase 1
# Shared helpers used across all packages.
# Nothing in utils/ should import from envs/, physics/, reward/ etc.
#
# Modules (to be created in Stage 10):
#   config.py       : YAML config loader --> typed dataclasses
#   seeding.py      : Global seed utility (numpy + torch + env)
#   logging.py      : Structured logger (file + console + TensorBoard)
#   geometry.py     : 2D geometry helpers (distance, angle, clamp)
#   io.py           : File I/O helpers (JSON, HDF5, YAML read/write)