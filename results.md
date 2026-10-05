# Final Datathon Status

## Current incumbent

Leaderboard score: `0.81534`

The exact file responsible for this score could not be fully reconstructed from local metadata. No script, JSON, Markdown, notebook, or filename explicitly maps a submission to `0.81534`.

Most likely local incumbent candidate:

`submission_mlp128_xgb45_lr15.csv`

Confidence: `LOW`

Reason: the filename is the most suggestive model-recipe clue, but this is inference rather than proof.

## Incumbent preservation

Preservation directory:

`artifacts/incumbent_081534/`

Contents:

- `candidate_submissions/`: copies of all local `submission*.csv`
- `probability_references/`: copies of key compatible `(97731, 7)` probability arrays
- `incumbent_audit.json`
- `final_push_report.json`
- `cheap_blend_grid.csv`

Likely incumbent distribution:

| Class | Count |
| --- | ---: |
| 1 | 11747 |
| 2 | 1197 |
| 3 | 22884 |
| 4 | 22388 |
| 5 | 15018 |
| 6 | 24101 |
| 7 | 396 |

## Existing validation evidence

These are from the existing 80/20 validation artifacts, not an honest 5-fold CV:

| Model | Local Macro F1 |
| --- | ---: |
| LGBM | 0.802625 |
| XGB | 0.755838 |
| CatBoost | 0.743161 |
| Existing ensemble | 0.804109 |
| Existing best 3-way | 0.804546 |
| Next LGBM | 0.803290 |
| Next XGB | 0.799351 |
| Next CatBoost | 0.779618 |

## Cheap blend candidate

Generated:

`submission_final_blend.csv`

Weights:

| Model | Weight |
| --- | ---: |
| next_lgbm | 0.775 |
| next_xgb | 0.200 |
| next_catboost | 0.025 |

Validation Macro F1: `0.807114`

Candidate distribution:

| Class | Count |
| --- | ---: |
| 1 | 11670 |
| 2 | 1053 |
| 3 | 23045 |
| 4 | 22748 |
| 5 | 14549 |
| 6 | 24334 |
| 7 | 332 |

Agreement with likely incumbent candidate: `90270 / 97731 = 92.37%`

## Recommendation

`KEEP INCUMBENT`

Only submit `submission_final_blend.csv` as an extra attempt if there is a spare submission slot and the incumbent has already been safely submitted. The candidate improves old local validation artifacts, but the exact `0.81534` recipe was not reconstructed and there is not enough evidence to replace the incumbent.
