# Presentation Checklist

Presentation file:
`presentation/PV_ML_Final_Presentation.pptx`

Number of slides:
14

Estimated speaking time:
~11.5-13.3 minutes (1,725 spoken words, at 130-150 words/minute) — slightly over the 8-12 minute target; a brisk, confident delivery should land in range, and Slides 2/5 are flagged as trimmable if needed

Figures:
6 actual project figures used (Output Power by city, Problem 1 confusion matrix, Problem 2 predicted-vs-actual, Problem 3 explained variance, Problem 4 label-efficiency curve, Problem 5 transfer curve) — all embedded directly from `figures/`, none recreated or invented. One slide (Slide 2, motivation) intentionally has no figure, since no existing project figure fits a purely conceptual/motivational slide — flagged rather than fabricating one.

Results verified:
YES — every number on every slide was checked against `results/FINAL_EXPERIMENT_TABLE.csv` and the underlying per-problem results files before writing slide content, and re-verified after the deck was built (via `markitdown` text extraction + grep against the actual rendered file).

Speaker script:
YES — `presentation/SPEAKER_SCRIPT.md`, generated directly from the pptx's own embedded speaker notes (not retyped separately), so the two can never drift out of sync.

Professor questions:
YES — `presentation/PROFESSOR_QUESTIONS.md` (20 questions across 8 categories: General Project, Problems 1-5, Results, Limitations).

Recording guide:
YES — `presentation/VIDEO_RECORDING_GUIDE.md`.

Remaining issues:
- Total estimated time (~11.5-13.3 min) runs slightly over the 8-12 min target at a slower speaking pace — addressed with a specific, named trim recommendation (Slides 2 and/or 5) rather than left unresolved.
- Title slide has placeholder fields (`[YOUR NAME]`, `[PROFESSOR NAME]`, `[DATE]`) — none of these are recorded anywhere in the project's files, so none were invented; must be filled in before recording.
- The presentation's `[PROFESSOR NAME]` placeholder was deliberately left generic rather than reusing "Prof. Puchovsky" from an earlier presentation draft — that name doesn't appear anywhere in this project's actual files and was almost certainly pulled in error from an unrelated course.
