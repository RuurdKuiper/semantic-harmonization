# Experiment 2: provisional test and future independent hold-out

The current test input is type 1 diabetes. It is excluded from selection in
Experiment 1. Earlier development used this EDF, so the experiment is described
as retrospective. A new independent hold-out set remains unavailable.

The freeze step fixes retrieval weight, decision model, stopping rule, reviewer,
review threshold and cap, and the CodeMapper inclusion decision for both branches.
The runner verifies the freeze checksum and validation-source hashes before
testing. It reports decision, post-review, and final TP/TN/FP/FN, recall, precision,
and F1 separately for each branch in `test_metrics.csv`. No test result changes
the frozen selections. A new independent dataset should use a separate output
directory with its split declared before selection.
