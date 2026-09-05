# EVALUATION package -- MDMARD Phase 1
# Tools for benchmarking trained policies and generating research metrics.
#
# Modules (to be created in Stage 15):
#   evaluator.py    : PolicyEvaluator -- run N episodes, collect metrics
#   benchmark.py    : BenchmarkSuite -- compare all agents on same seed
#   report.py       : Auto-generate matplotlib figures and LaTeX tables
#   timing.py       : Inference latency profiler (ms per step)