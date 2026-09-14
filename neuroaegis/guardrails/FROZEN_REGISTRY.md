# Frozen Model Registry

Models that will touch Siena data or clinician-annotated data MUST be registered here with a SHA256 hash and a timestamp.
This is enforced by the `check_frozen()` guardrail to prevent repeated peeking or unintended tuning.

| Model Name       | SHA256 Hash                                                      | Frozen At            | Purpose                |
|------------------|------------------------------------------------------------------|----------------------|------------------------|
| mock_model_v1    | d2a84f4b8b650937ec8f73cd8be2c74add5a911ba64df27458ed8229da804a26 | 2026-09-13T12:00:00Z | Track D Annotation     |

| fake_model | 1eec943f3fbf69947176e7c711415ad88a08184169f72cd31dca0ab071e14939 | 2026-09-13 | Test eval |

| fake_model | 1eec943f3fbf69947176e7c711415ad88a08184169f72cd31dca0ab071e14939 | 2026-09-13 | Test eval |
