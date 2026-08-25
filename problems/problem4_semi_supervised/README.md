# Problem 4 — Semi-Supervised Learning

**Status: NOT STARTED.**

This folder will eventually contain the code for Problem 4: comparing
a supervised-only baseline against a semi-supervised method at
labeled fractions of 10%/30%/50%.

Before writing any code here, read:
- `course_context/TEACHER_EXPECTATIONS.md` (Problem 4 section) - note
  the SSL algorithm choice is still an open decision (course-taught
  methods vs. pseudo-labeling).
- `course_context/EXPERIMENT_PLAN.md` (Problem 4 section).

Use `src/splitting.py`'s `random_labeled_subset()` for the
labeled/unlabeled split - it's built specifically for this and keeps
the split's randomness separate from the chronological train/test
split.
