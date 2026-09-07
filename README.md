# Reported satellite rice-mapping accuracy: extraction table and meta-analytic code

Companion deposit to *"Has algorithmic innovation improved reported satellite rice-mapping
accuracy? A global meta-analysis of 417 studies"*. Licence: CC BY 4.0.

This archive contains the two things the paper's data availability statement names: the
provenance-audited extraction table, and the code that performs the meta-analysis.

## Contents

| Folder | What is in it |
|---|---|
| `data/` | `meta_study_table.csv` - the extraction table, one row per included study, every accuracy value accompanied by the verbatim sentence it was taken from. Plus the poolable-F1 subset and the two extraction intermediates the analysis code reads |
| `code/` | The 15 analysis scripts: pooled estimation and heterogeneity, moderator and subgroup tests, equivalence testing, effect sizes, cluster-robust meta-regression, and the robustness and sensitivity analyses |
| `results/` | The JSON the code reads and writes. Several scripts take another's output as input, so these ship with the code rather than as a separate product |

## Running it

Python 3.10 or later with NumPy, SciPy, statsmodels and pandas. Paths resolve relative to
the archive root, so no configuration is needed:

    python code/meta_formal.py        # pooled estimates and heterogeneity
    python code/effect_sizes.py       # effect size for every rank-based test
    python code/cluster_robust_tost.py

Pooled estimates were cross-checked independently in R with `metafor`.

Fourteen of the nineteen scripts run from this archive as it stands. Five do not, because
they read the cached full texts of the primary studies, which cannot be redistributed:
`metric_census.py`, `secondary_f1.py`, `coding_confusion.py`, `recode_validation.py` and
`covariate_rule.py`. They are included because they document exactly how the extraction and
the reliability checks were performed, and their outputs are in `results/`.

## What is not here

**The full texts of the reviewed studies.** They belong to their publishers. The extraction
table quotes the source sentence for every value, so any number can be checked against the
original article by anyone with access to it.

**Figure and table scripts, and the screening-validation materials.** This deposit is
scoped to the extraction table and the meta-analytic code. The screening validation
reported in the paper - one independent reviewer, blind, on a stratified 500-record sample,
giving 43% precision and 18.7% recall against the criteria as written and 42% and 63.2%
against the criterion the harvest implements - is described in full in the paper and its
supplementary material, and the underlying sheets are available from the corresponding
author on request.
