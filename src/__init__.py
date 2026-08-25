"""
src package - reusable framework code shared across Problems 1-5.

See course_context/AI_AGENT_INSTRUCTIONS.md for how this framework is
meant to be used, and course_context/EXPERIMENT_PLAN.md for how each
problem will use these modules.

Modules:
    utils               - random seeds, GPU device detection, small helpers.
    data_loader          - load the Excel dataset into DataFrames.
    preprocessing         - missing values, scaling, categorical encoding.
    feature_engineering   - Clear-Sky Index, time encodings, lag features.
    splitting             - chronological, cross-city, and random-subset splits.
    evaluation            - classification and regression metrics.
    visualization          - plots, always saved to file.
    experiment_runner      - config loading and result logging.
"""
