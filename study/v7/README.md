# V7 execution evidence

The frozen protocol and completed results are in
[artifacts/v7](../../artifacts/v7/REPORT.md). The supplied patch is preserved
with author attribution under `patches`; its private external session URL was
removed from the commit message only. Claims supplied with it are hypotheses,
with their independently executed evidence separated in the final report.

These scripts record the isolated study topology used for execution. They are
not an installer: `runtime.py` expects this directory to contain a `kit` copy,
`venv`, frozen `inputs`, and sibling v3/v6 result directories. Raw data are a
separate read-only mount. The publication copy intentionally includes none of
the raw dataset. Do not run `setup_remote.py` or `controller.py` over an existing
study: the completed controller is not a retry or resume service.

Core reusable tools live under `kit/tools`: `decision_fusion.py`,
`fusion_breakdown.py`, `subset_bootstrap.py`, `sync_audit.py` and `score_runs.py`.
Run their `--help` for path arguments. Scoring and breakdown need separately
obtained original IR annotations. Published gzip predictions can be read by
the hardened reader; fusion breakdown requires decompressing its CSV log.

The publication and execution layouts differ. The additional local checks can
be run from the repository root with:

```bash
python -m unittest discover -s study/v7 -p 'test_*.py' -v
python -m unittest discover -s kit/tests
```

`geometry_review.py` and `media_timing_audit.py` are explanatory diagnostics
added after registration; they did not change fusion inputs, exclusions or the
gate. `finalize_report.py` validates the completed portable export and report;
its `finish()` function can validate already integrated artifacts. Export and
report generation do not start inference or training. The final source/tests
record and pre-scoring input hashes identify the actual executed versions.
