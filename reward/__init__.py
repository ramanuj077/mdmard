# REWARD package -- MDMARD Phase 1
# Pluggable reward components. Ablation = swap class in config.
#
# Modules (to be created in Stage 4):
#   base.py         : BaseReward abstract base class
#   components.py   : Individual reward components
#   composite.py    : CompositeReward -- combines components with weights
#   shaping.py      : Potential-based shaping utilities (Ng et al. 1999)