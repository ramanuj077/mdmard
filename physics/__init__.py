# PHYSICS package -- MDMARD Phase 1
# Aircraft and environment dynamics modules.
#
# Modules (to be created in Stage 1 refactor):
#   dynamics.py   : Aircraft equations of motion (RK4 integrator)
#   state.py      : AircraftState and PhysicsConfig dataclasses
#   actions.py    : Action decoders (discrete and continuous)
#   forces.py     : Force computation (thrust, drag, lift, gravity)
#   wind.py       : Wind disturbance models (Stage 13+)