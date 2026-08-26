"""
src package - reusable framework code shared across Problems 1-5.

See course_context/AI_AGENT_INSTRUCTIONS.md for how this framework is
meant to be used, and course_context/EXPERIMENT_PLAN.md for how each
problem will use these modules.

Modules:
    utils               - random seeds, GPU device detection, small helpers.
    data_loader          - load the Excel dataset into DataFrames.
    cleaning              - detect/report missing values, duplicates, anomalies.
    preprocessing         - feature/target separation, scaling, categorical encoding.
    feature_engineering   - Clear-Sky Index, time encodings, lag features.
    splitting             - chronological, cross-city, and random-subset splits.
    evaluation            - classification and regression metrics, multi-seed aggregation.
    visualization          - plots, always saved to file.
    experiment_runner      - config loading, experiment IDs, results/leaderboard.
    torch_utils            - generic PyTorch training loop (any nn.Module).
"""
